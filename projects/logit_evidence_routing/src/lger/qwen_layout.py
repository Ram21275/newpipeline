"""Validation helpers for Qwen2.5-VL's dynamic visual-token layout."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def expected_merged_visual_tokens(
    image_grid_thw: torch.Tensor | Sequence[Sequence[int]],
    *,
    spatial_merge_size: int,
) -> tuple[int, ...]:
    """Compute language-visible token counts from Qwen's pre-merge THW grid."""

    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    values = torch.as_tensor(image_grid_thw, dtype=torch.long)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("image_grid_thw must have shape [images,3]")
    if bool((values <= 0).any()):
        raise ValueError("all temporal and spatial grid dimensions must be positive")
    divisor = spatial_merge_size**2
    premerge = values.prod(dim=1)
    if bool((premerge % divisor != 0).any()):
        raise ValueError("pre-merge patch count is not divisible by spatial merge area")
    return tuple(int(value) for value in (premerge // divisor).tolist())


def qwen_layout_assertions(
    *,
    input_image_tokens: int,
    expected_image_tokens: Sequence[int],
    visual_output_tokens: int,
    input_sequence_tokens: int,
    hidden_sequence_tokens: int,
) -> dict[str, bool]:
    """Return named invariants and raise if Qwen token alignment is unsafe."""

    expected_total = sum(int(value) for value in expected_image_tokens)
    checks = {
        "processor_matches_grid": input_image_tokens == expected_total,
        "visual_output_matches_processor": visual_output_tokens == input_image_tokens,
        "language_hidden_matches_input": hidden_sequence_tokens == input_sequence_tokens,
        "nonempty_visual_sequence": input_image_tokens > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Qwen layout validation failed: {failed}")
    return checks
