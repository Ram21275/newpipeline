"""Deterministic multi-scale regions over an existing visual-token grid."""

from __future__ import annotations

from typing import Sequence


def _grid_shape(grid_shape: Sequence[int]) -> tuple[int, int]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain height and width")
    height, width = (int(value) for value in grid_shape)
    if height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive")
    return height, width


def valid_window_centers(
    grid_shape: Sequence[int], *, window_width: int, dilation: int = 1
) -> list[int]:
    """Return centers whose complete odd dilated window fits inside the grid."""

    height, width = _grid_shape(grid_shape)
    if window_width <= 0 or window_width % 2 == 0:
        raise ValueError("window_width must be a positive odd integer")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    radius = (window_width // 2) * dilation
    return [row * width + col
            for row in range(radius, height - radius)
            for col in range(radius, width - radius)]


def window_indices(
    grid_shape: Sequence[int], center_index: int, *, window_width: int,
    dilation: int = 1,
) -> tuple[int, ...]:
    """Return row-major indices for one complete odd dilated token window."""

    height, width = _grid_shape(grid_shape)
    if center_index < 0 or center_index >= height * width:
        raise IndexError("center_index is outside the visual-token grid")
    if center_index not in set(valid_window_centers(
        grid_shape, window_width=window_width, dilation=dilation
    )):
        raise ValueError("the complete window does not fit around center_index")
    center_row, center_col = divmod(center_index, width)
    radius = window_width // 2
    return tuple(
        (center_row + row_offset * dilation) * width
        + center_col + col_offset * dilation
        for row_offset in range(-radius, radius + 1)
        for col_offset in range(-radius, radius + 1)
    )


def pooled_grid_regions(
    grid_shape: Sequence[int], pooled_shape: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Partition a token grid into equal non-overlapping pooled regions."""

    height, width = _grid_shape(grid_shape)
    pooled_height, pooled_width = _grid_shape(pooled_shape)
    if height % pooled_height or width % pooled_width:
        raise ValueError("pooled_shape must divide the visual-token grid exactly")
    cell_height, cell_width = height // pooled_height, width // pooled_width
    regions = []
    for pooled_row in range(pooled_height):
        for pooled_col in range(pooled_width):
            regions.append(tuple(
                row * width + col
                for row in range(pooled_row * cell_height, (pooled_row + 1) * cell_height)
                for col in range(pooled_col * cell_width, (pooled_col + 1) * cell_width)
            ))
    return tuple(regions)
