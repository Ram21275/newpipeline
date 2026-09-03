"""Class-agnostic vocabulary evidence scores and layer aggregation."""

import torch


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def logits_to_evidence(logits: torch.Tensor, method: str) -> torch.Tensor:
    """Convert ``[..., vocabulary]`` logits into one score per visual patch."""

    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("logits must have at least two vocabulary entries")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    _require_finite("logits", logits)

    if method == "margin":
        top_two = torch.topk(logits, k=2, dim=-1).values
        return top_two[..., 0] - top_two[..., 1]
    if method == "maxprob":
        return torch.softmax(logits.float(), dim=-1).amax(dim=-1).to(logits.dtype)
    if method == "negentropy":
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return (probs * log_probs).sum(dim=-1).to(logits.dtype)
    raise ValueError("method must be one of: margin, maxprob, negentropy")


def normalize_within_layers(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-normalize patch scores independently within each layer."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [layers, patches]")
    _require_finite("scores", scores)
    mean = scores.mean(dim=-1, keepdim=True)
    scale = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (scores - mean) / scale


def normalize_patch_scores(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-normalize a single score per patch."""

    if scores.ndim != 1:
        raise ValueError("scores must have shape [patches]")
    _require_finite("scores", scores)
    return (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(eps)


def aggregate_layer_scores(
    layer_scores: torch.Tensor,
    method: str,
    persistent_lambda: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align patches by spatial index and aggregate evidence across layers."""

    normalized = normalize_within_layers(layer_scores)
    if method == "last":
        aggregate = normalized[-1]
    elif method == "mean_layers":
        aggregate = normalized.mean(dim=0)
    elif method == "persistent":
        aggregate = normalized.mean(dim=0) - persistent_lambda * normalized.std(
            dim=0, unbiased=False
        )
    else:
        raise ValueError("method must be one of: last, mean_layers, persistent")
    return aggregate, normalized


def reduce_attention(attention: torch.Tensor) -> torch.Tensor:
    """Produce one documented score per patch from one or more attention layers."""

    if attention.ndim == 1:
        _require_finite("attention", attention)
        return attention
    if attention.ndim == 2:
        return normalize_within_layers(attention).mean(dim=0)
    raise ValueError("attention must have shape [patches] or [layers, patches]")


def stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return deterministic descending indices, using spatial index for ties."""

    if scores.ndim != 1:
        raise ValueError("scores must have shape [patches]")
    if not 0 < k <= scores.numel():
        raise ValueError("k must be between 1 and the number of patches")
    _require_finite("scores", scores)
    return torch.argsort(scores, descending=True, stable=True)[:k]
