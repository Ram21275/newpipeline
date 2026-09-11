"""Exact spatial summaries for mean-pooled linear attribute probes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .localization import (
    patch_centers_in_box,
    selection_localization_metrics,
    selection_part_metrics,
)
from .scoring import stable_topk


def exact_patch_margin(
    patch_states: torch.Tensor,
    *,
    normalization_mean: torch.Tensor,
    normalization_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | float,
    threshold: torch.Tensor | float,
) -> torch.Tensor:
    """Return patch terms whose sum is the threshold-centered probe margin."""

    if patch_states.ndim != 2 or patch_states.shape[0] == 0:
        raise ValueError("patch states must have shape [patches, feature_dim]")
    feature_dim = patch_states.shape[1]
    vectors = (normalization_mean, normalization_scale, weight)
    if any(value.ndim != 1 or value.numel() != feature_dim for value in vectors):
        raise ValueError("probe vectors do not match the patch feature dimension")
    scale = normalization_scale.float()
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("normalization scale must be finite and positive")
    normalized = (patch_states.float() - normalization_mean.float()) / scale
    patch_count = patch_states.shape[0]
    linear = normalized @ weight.float() / patch_count
    offset = (float(bias) - float(threshold)) / patch_count
    result = linear + offset
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("patch margin contains non-finite values")
    return result


def landmark_patch_mask(
    points: Sequence[tuple[float, float]],
    *,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> torch.Tensor:
    rows, columns = grid_size
    height, width = image_size
    if min(rows, columns, height, width) <= 0:
        raise ValueError("grid and image dimensions must be positive")
    mask = torch.zeros(rows * columns, dtype=torch.bool)
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("landmark point falls outside the processed image")
        column = min(int(x * columns / width), columns - 1)
        row = min(int(y * rows / height), rows - 1)
        mask[row * columns + column] = True
    return mask


def patch_evidence_summary(
    patch_margin: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    bbox_xyxy: tuple[float, float, float, float],
    landmark_points: Sequence[tuple[float, float]],
    k_values: Sequence[int] = (16, 32),
) -> dict[str, float | int | list[int] | None]:
    """Summarize concentration and localization without treating it as causality."""

    if patch_margin.ndim != 1 or patch_margin.numel() != grid_size[0] * grid_size[1]:
        raise ValueError("patch margin does not align with the patch grid")
    if not k_values or min(k_values) <= 0 or max(k_values) > patch_margin.numel():
        raise ValueError("invalid Top-K values")
    probabilities = torch.softmax(patch_margin.float(), dim=0)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    box_mask = patch_centers_in_box(grid_size, image_size, bbox_xyxy)
    landmark_mask = landmark_patch_mask(
        landmark_points, grid_size=grid_size, image_size=image_size
    )
    output: dict[str, float | int | list[int] | None] = {
        "probe_margin_reconstructed": float(patch_margin.sum()),
        "evidence_entropy": entropy,
        "evidence_effective_tokens": math.exp(entropy),
        "bbox_evidence_mass": float(probabilities[box_mask].sum()),
        "landmark_evidence_mass": (
            float(probabilities[landmark_mask].sum())
            if bool(landmark_mask.any())
            else None
        ),
        "relevant_landmark_patches": int(landmark_mask.sum()),
    }
    for k in k_values:
        selected = stable_topk(patch_margin, int(k))
        box = selection_localization_metrics(selected, box_mask)
        prefix = f"top{k}"
        output[f"{prefix}_indices"] = [int(value) for value in selected.tolist()]
        for name, value in box.items():
            output[f"{prefix}_{name}"] = value
        if landmark_points:
            parts = selection_part_metrics(
                selected,
                landmark_points,
                grid_size=grid_size,
                image_size=image_size,
            )
            for name, value in parts.items():
                output[f"{prefix}_{name}"] = value
        else:
            for name in (
                "part_patch_recall",
                "any_part_hit",
                "top1_part_hit",
                "top1_nearest_part_distance_patches",
                "visible_part_patches",
            ):
                output[f"{prefix}_{name}"] = None
    return output
