"""Patch-localization scores and grid-level evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _validate_vision_attentions(attentions: Sequence[torch.Tensor]) -> int:
    if not attentions:
        raise ValueError("at least one vision attention layer is required")
    token_count = attentions[0].shape[-1]
    for layer in attentions:
        if layer.ndim != 4 or layer.shape[0] != 1:
            raise ValueError("vision attentions must have shape [1, heads, tokens, tokens]")
        if layer.shape[-2:] != (token_count, token_count):
            raise ValueError("vision attention layers must use the same square token grid")
        if not torch.isfinite(layer).all():
            raise ValueError("vision attentions contain NaN or infinite values")
    if token_count <= 1:
        raise ValueError("vision attentions must contain CLS and patch tokens")
    return token_count


def final_cls_attention(attentions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average final-layer vision CLS-to-patch attention over heads."""

    _validate_vision_attentions(attentions)
    return attentions[-1][0, :, 0, 1:].float().mean(dim=0)


def attention_rollout(attentions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Propagate final CLS relevance through residual-aware vision attention."""

    token_count = _validate_vision_attentions(attentions)
    device = attentions[0].device
    relevance = torch.zeros(token_count, device=device, dtype=torch.float32)
    relevance[0] = 1.0
    identity = torch.eye(token_count, device=device, dtype=torch.float32)
    for layer in reversed(attentions):
        transition = layer[0].float().mean(dim=0) + identity
        transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        relevance = relevance @ transition
    return relevance[1:]


def patch_centers_in_box(
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    bbox_xyxy: tuple[float, float, float, float],
) -> torch.Tensor:
    """Return a flat boolean mask for patch centers inside an image-space box."""

    rows, columns = grid_size
    height, width = image_size
    if rows <= 0 or columns <= 0 or height <= 0 or width <= 0:
        raise ValueError("grid and image dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox must be a non-empty xyxy box inside the image")
    x_centers = (torch.arange(columns, dtype=torch.float32) + 0.5) * width / columns
    y_centers = (torch.arange(rows, dtype=torch.float32) + 0.5) * height / rows
    inside_x = (x_centers >= x1) & (x_centers < x2)
    inside_y = (y_centers >= y1) & (y_centers < y2)
    return (inside_y[:, None] & inside_x[None, :]).flatten()


def selection_localization_metrics(
    selected_indices: torch.Tensor,
    bbox_patch_mask: torch.Tensor,
) -> dict[str, float]:
    """Evaluate an ordered Top-K selection against a patch-grid bounding box."""

    if selected_indices.ndim != 1 or bbox_patch_mask.ndim != 1:
        raise ValueError("selection and bounding-box mask must be one-dimensional")
    if selected_indices.numel() == 0:
        raise ValueError("selection must contain at least one patch")
    if selected_indices.min() < 0 or selected_indices.max() >= bbox_patch_mask.numel():
        raise ValueError("selection contains an out-of-range patch index")
    if selected_indices.unique().numel() != selected_indices.numel():
        raise ValueError("selection contains duplicate patch indices")
    bbox_patch_mask = bbox_patch_mask.bool()
    bbox_count = int(bbox_patch_mask.sum())
    if bbox_count == 0:
        raise ValueError("bounding box does not contain any patch centers")
    selected_mask = torch.zeros_like(bbox_patch_mask)
    selected_mask[selected_indices.long()] = True
    intersection = int((selected_mask & bbox_patch_mask).sum())
    union = int((selected_mask | bbox_patch_mask).sum())
    return {
        "inside_fraction": intersection / selected_indices.numel(),
        "bbox_patch_recall": intersection / bbox_count,
        "bbox_patch_iou": intersection / union,
        "pointing_game": float(bbox_patch_mask[int(selected_indices[0])]),
    }
