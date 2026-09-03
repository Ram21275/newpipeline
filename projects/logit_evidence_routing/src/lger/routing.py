"""Unified routing API for all patch-selection comparisons."""

from __future__ import annotations

import torch

from .scoring import (
    aggregate_layer_scores,
    logits_to_evidence,
    normalize_patch_scores,
    reduce_attention,
    stable_topk,
)
from .types import RouterConfig, SelectionResult


class EvidenceRouter:
    """Select visual evidence without receiving labels or class names."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    def __call__(
        self,
        candidate_hidden_states: torch.Tensor,
        candidate_logits: torch.Tensor,
        patch_coords: torch.Tensor,
        layer_ids: torch.Tensor,
        *,
        attention_scores: torch.Tensor | None = None,
        k: int | None = None,
    ) -> SelectionResult:
        self._validate_inputs(
            candidate_hidden_states,
            candidate_logits,
            patch_coords,
            layer_ids,
            attention_scores,
        )
        patch_coords = patch_coords.to(candidate_hidden_states.device)
        layer_ids = layer_ids.to(candidate_hidden_states.device)
        if attention_scores is not None:
            attention_scores = attention_scores.to(candidate_logits.device)
        patch_count = candidate_hidden_states.shape[1]
        evidence_k = self.config.k if k is None else k
        context_k = self.config.context_k if self.config.context == "random" else 0
        if evidence_k <= 0 or evidence_k + context_k > patch_count:
            raise ValueError("k plus context_k must fit within the candidate patch count")

        raw_layer_scores = logits_to_evidence(candidate_logits, self.config.score)
        lger_scores, normalized_layer_scores = aggregate_layer_scores(
            raw_layer_scores,
            self.config.aggregate,
            self.config.persistent_lambda,
        )
        selection_scores, evidence_indices = self._select_evidence(
            raw_layer_scores,
            lger_scores,
            attention_scores,
            evidence_k,
            patch_count,
        )
        context_indices = self._sample_context(
            evidence_indices,
            patch_count,
            context_k,
            candidate_hidden_states.device,
        )
        selected_indices = torch.cat((evidence_indices, context_indices))
        context_mask = torch.cat(
            (
                torch.zeros(evidence_indices.numel(), dtype=torch.bool),
                torch.ones(context_indices.numel(), dtype=torch.bool),
            )
        ).to(candidate_hidden_states.device)

        representation_layer = candidate_hidden_states.shape[0] - 1
        representations = candidate_hidden_states[representation_layer]
        selected_layer_id = layer_ids[representation_layer].expand(selected_indices.numel())
        layer_statistics = {
            "raw_scores": raw_layer_scores,
            "normalized_scores": normalized_layer_scores,
            "aggregate_scores": lger_scores,
            "mean_score": normalized_layer_scores.mean(dim=0),
            "score_std": normalized_layer_scores.std(dim=0, unbiased=False),
        }
        return SelectionResult(
            hidden_states=representations.index_select(0, selected_indices),
            patch_indices=selected_indices,
            patch_coords=patch_coords.index_select(0, selected_indices),
            scores=selection_scores.index_select(0, selected_indices),
            layer_ids=selected_layer_id,
            context_mask=context_mask,
            layer_statistics=layer_statistics,
        )

    def _select_evidence(
        self,
        raw_layer_scores: torch.Tensor,
        lger_scores: torch.Tensor,
        attention_scores: torch.Tensor | None,
        k: int,
        patch_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selector = self.config.selector
        if selector == "random":
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
            indices = torch.randperm(patch_count, generator=generator)[:k].to(
                raw_layer_scores.device
            )
            return torch.zeros(patch_count, device=raw_layer_scores.device), indices

        if selector == "attention":
            if attention_scores is None:
                raise ValueError("attention selector requires attention_scores")
            scores = reduce_attention(attention_scores)
        elif selector == "logit":
            scores = raw_layer_scores[-1]
        elif selector == "attention_logit":
            if attention_scores is None:
                raise ValueError("attention_logit selector requires attention_scores")
            scores = 0.5 * (
                normalize_patch_scores(reduce_attention(attention_scores))
                + normalize_patch_scores(raw_layer_scores[-1])
            )
        elif selector == "lger":
            scores = lger_scores
        else:  # RouterConfig validates this; retained as a defensive boundary.
            raise RuntimeError(f"unsupported selector: {selector}")
        return scores, stable_topk(scores, k)

    def _sample_context(
        self,
        evidence_indices: torch.Tensor,
        patch_count: int,
        context_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        if context_k == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        available = torch.ones(patch_count, dtype=torch.bool)
        available[evidence_indices.detach().cpu()] = False
        candidates = torch.arange(patch_count)[available]
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 1)
        order = torch.randperm(candidates.numel(), generator=generator)[:context_k]
        return candidates[order].to(device)

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        patch_coords: torch.Tensor,
        layer_ids: torch.Tensor,
        attention_scores: torch.Tensor | None,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("candidate_hidden_states must be [layers, patches, hidden]")
        if logits.ndim != 3:
            raise ValueError("candidate_logits must be [layers, patches, vocabulary]")
        if hidden_states.shape[:2] != logits.shape[:2]:
            raise ValueError("hidden states and logits must align by layer and patch")
        layers, patches = hidden_states.shape[:2]
        if patch_coords.shape != (patches, 2):
            raise ValueError("patch_coords must have shape [patches, 2]")
        if layer_ids.shape != (layers,):
            raise ValueError("layer_ids must have shape [layers]")
        if not hidden_states.is_floating_point() or not logits.is_floating_point():
            raise TypeError("hidden states and logits must be floating point")
        if hidden_states.device != logits.device:
            raise ValueError("hidden states and logits must be on the same device")
        if attention_scores is not None and tuple(attention_scores.shape) not in {
            (patches,),
            (layers, patches),
        }:
            raise ValueError("attention_scores must be [patches] or [layers, patches]")
