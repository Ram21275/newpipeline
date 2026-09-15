"""Training-free layer decoding for frozen multimodal language models.

The visual encoder is not modified here.  The method operates on language
states *after* visual tokens and the question have entered the decoder, so the
candidate and mature predictions are conditioned on the same image--text
input.  This keeps the intervention close to DoLa while making the measurement
valid for LLaVA- and Qwen-like architectures.  In addition to the DoLa
contrast, this module implements DeCo's additive preceding-layer correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MultimodalDoLaOutput:
    """Auditable outputs from dynamic premature-layer selection."""

    contrasted_logits: torch.Tensor
    premature_indices: torch.Tensor
    js_divergences: torch.Tensor
    mature_log_probabilities: torch.Tensor
    premature_log_probabilities: torch.Tensor
    plausibility_mask: torch.Tensor


@dataclass(frozen=True)
class MultimodalDeCoOutput:
    """Auditable outputs from DeCo-style dynamic anchor selection."""

    corrected_logits: torch.Tensor
    anchor_indices: torch.Tensor
    anchor_probabilities: torch.Tensor
    candidate_token_mask: torch.Tensor
    layer_candidate_max_probabilities: torch.Tensor


def project_query_hidden_states(
    hidden_states: torch.Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    query_positions: torch.Tensor | int,
) -> torch.Tensor:
    """Project one query position per example through the shared final head."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    batch, sequence, _ = hidden_states.shape
    if isinstance(query_positions, int):
        positions = torch.full(
            (batch,), query_positions, dtype=torch.long, device=hidden_states.device
        )
    else:
        positions = query_positions.to(device=hidden_states.device, dtype=torch.long)
    if positions.shape != (batch,):
        raise ValueError("query_positions must be one integer per batch element")
    positions = torch.where(positions < 0, positions + sequence, positions)
    if bool(((positions < 0) | (positions >= sequence)).any()):
        raise ValueError("query position is outside the active sequence")
    selected = hidden_states[
        torch.arange(batch, device=hidden_states.device), positions
    ]
    projection_device = next(lm_head.parameters()).device
    selected = selected.to(projection_device)
    return lm_head(final_norm(selected))


def stack_projected_layers(
    layer_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    query_positions: torch.Tensor | int,
) -> torch.Tensor:
    """Return candidate logits with shape [batch, layers, vocabulary]."""

    if not layer_hidden_states:
        raise ValueError("at least one candidate layer is required")
    projected = [
        project_query_hidden_states(
            state,
            final_norm=final_norm,
            lm_head=lm_head,
            query_positions=query_positions,
        )
        for state in layer_hidden_states
    ]
    if any(values.shape != projected[0].shape for values in projected[1:]):
        raise ValueError("all projected layers must share batch and vocabulary dimensions")
    return torch.stack(projected, dim=1)


def _js_divergence(mature_logp: torch.Tensor, candidate_logp: torch.Tensor) -> torch.Tensor:
    mature = mature_logp.exp().unsqueeze(1)
    candidate = candidate_logp.exp()
    mixture = 0.5 * (mature + candidate)
    log_mixture = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    mature_kl = (mature * (mature_logp.unsqueeze(1) - log_mixture)).sum(dim=-1)
    candidate_kl = (candidate * (candidate_logp - log_mixture)).sum(dim=-1)
    return 0.5 * (mature_kl + candidate_kl)


def dynamic_layer_contrast(
    mature_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    plausibility_alpha: float | None = None,
    masked_value: float = -1000.0,
) -> MultimodalDoLaOutput:
    """Select the maximum-JSD layer and subtract its log probabilities.

    Inputs may contain the full vocabulary (the paper-faithful option) or only
    the answer alternatives (the constrained binary-classification option).
    Candidate selection is performed independently for every example.
    """

    if mature_logits.ndim != 2 or candidate_logits.ndim != 3:
        raise ValueError("expected mature [batch,vocab] and candidates [batch,layers,vocab]")
    if (
        candidate_logits.shape[0] != mature_logits.shape[0]
        or candidate_logits.shape[2] != mature_logits.shape[1]
        or candidate_logits.shape[1] == 0
    ):
        raise ValueError("candidate logits do not align with mature logits")
    if plausibility_alpha is not None and not 0.0 < plausibility_alpha <= 1.0:
        raise ValueError("plausibility_alpha must be in (0,1] or None")
    if not bool(torch.isfinite(mature_logits).all()) or not bool(
        torch.isfinite(candidate_logits).all()
    ):
        raise ValueError("DoLa logits must be finite")

    mature_logp = F.log_softmax(mature_logits.float(), dim=-1)
    candidate_logp = F.log_softmax(candidate_logits.float(), dim=-1)
    divergences = _js_divergence(mature_logp, candidate_logp)
    if candidate_mask is None:
        valid = torch.ones_like(divergences, dtype=torch.bool)
    else:
        valid = candidate_mask.to(device=divergences.device, dtype=torch.bool)
        if valid.shape != divergences.shape:
            raise ValueError("candidate_mask must have shape [batch,layers]")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every example needs at least one valid candidate layer")
    selected_indices = divergences.masked_fill(~valid, -torch.inf).argmax(dim=1)
    row = torch.arange(mature_logits.shape[0], device=mature_logits.device)
    premature_logp = candidate_logp[row, selected_indices]
    contrasted = mature_logp - premature_logp

    if plausibility_alpha is None:
        plausible = torch.ones_like(contrasted, dtype=torch.bool)
    else:
        mature_probabilities = mature_logp.exp()
        threshold = plausibility_alpha * mature_probabilities.max(dim=-1, keepdim=True).values
        plausible = mature_probabilities >= threshold
        contrasted = contrasted.masked_fill(~plausible, masked_value)
    return MultimodalDoLaOutput(
        contrasted_logits=contrasted,
        premature_indices=selected_indices,
        js_divergences=divergences,
        mature_log_probabilities=mature_logp,
        premature_log_probabilities=premature_logp,
        plausibility_mask=plausible,
    )


def top_p_candidate_mask(
    logits: torch.Tensor, *, top_p: float, top_k: int | None = None
) -> torch.Tensor:
    """Return a nucleus set, optionally capped to the highest ``top_k`` tokens."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch,vocabulary]")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0,1]")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("candidate logits must be finite")
    probabilities = F.softmax(logits.float(), dim=-1)
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative_before = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
    sorted_mask = cumulative_before < top_p
    if top_k is not None:
        ranks = torch.arange(
            sorted_mask.shape[-1], device=sorted_mask.device
        ).unsqueeze(0)
        sorted_mask = sorted_mask & (ranks < min(top_k, sorted_mask.shape[-1]))
    mask = torch.zeros_like(sorted_mask)
    return mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)


def dynamic_layer_correction(
    mature_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    alpha: float,
    top_p: float = 0.9,
    top_k: int | None = None,
    candidate_token_mask: torch.Tensor | None = None,
) -> MultimodalDeCoOutput:
    """Apply DeCo's dynamically modulated additive logit correction.

    The final-layer top-p set is the paper-faithful candidate vocabulary.  A
    caller may instead supply a fixed candidate mask, which is useful for the
    preregistered yes/no-constrained CUB arm.  For each example, the anchor is
    the layer with the largest probability assigned to any candidate token.
    Its raw projected logits are added to the final logits with DeCo's
    ``alpha * max_prob`` soft-modulation coefficient.
    """

    if mature_logits.ndim != 2 or candidate_logits.ndim != 3:
        raise ValueError("expected mature [batch,vocab] and candidates [batch,layers,vocab]")
    if (
        candidate_logits.shape[0] != mature_logits.shape[0]
        or candidate_logits.shape[2] != mature_logits.shape[1]
        or candidate_logits.shape[1] == 0
    ):
        raise ValueError("candidate logits do not align with mature logits")
    if not 0.0 <= alpha:
        raise ValueError("alpha must be non-negative")
    if not bool(torch.isfinite(mature_logits).all()) or not bool(
        torch.isfinite(candidate_logits).all()
    ):
        raise ValueError("DeCo logits must be finite")

    if candidate_token_mask is None:
        token_mask = top_p_candidate_mask(mature_logits, top_p=top_p, top_k=top_k)
    else:
        token_mask = candidate_token_mask.to(device=mature_logits.device, dtype=torch.bool)
        if token_mask.shape != mature_logits.shape:
            raise ValueError("candidate_token_mask must have shape [batch,vocabulary]")
        if not bool(token_mask.any(dim=-1).all()):
            raise ValueError("every example needs at least one candidate token")

    candidate_probabilities = F.softmax(candidate_logits.float(), dim=-1)
    layer_scores = candidate_probabilities.masked_fill(
        ~token_mask.unsqueeze(1), -torch.inf
    ).amax(dim=-1)
    anchor_indices = layer_scores.argmax(dim=-1)
    row = torch.arange(mature_logits.shape[0], device=mature_logits.device)
    anchor_logits = candidate_logits.float()[row, anchor_indices]
    anchor_probabilities = candidate_probabilities[row, anchor_indices].amax(dim=-1)
    corrected_logits = (
        mature_logits.float()
        + alpha * anchor_probabilities.unsqueeze(-1) * anchor_logits
    )
    return MultimodalDeCoOutput(
        corrected_logits=corrected_logits,
        anchor_indices=anchor_indices,
        anchor_probabilities=anchor_probabilities,
        candidate_token_mask=token_mask,
        layer_candidate_max_probabilities=layer_scores,
    )


def semantic_binary_margin(
    logits: torch.Tensor,
    *,
    positive_index: int,
    negative_index: int,
) -> torch.Tensor:
    """Return positive-minus-negative evidence without target-sign recoding."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch,vocabulary]")
    width = int(logits.shape[1])
    if not 0 <= positive_index < width or not 0 <= negative_index < width:
        raise ValueError("answer index is outside the vocabulary")
    if positive_index == negative_index:
        raise ValueError("positive and negative answer indices must differ")
    return logits[:, positive_index] - logits[:, negative_index]
