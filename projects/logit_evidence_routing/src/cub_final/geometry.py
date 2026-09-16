"""Processor-aware point/box mapping and crop construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TransformTrace:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[float, float, float, float]
    output_size: tuple[int, int]
    pad_left: float = 0.0
    pad_top: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_point(point: tuple[float, float], trace: TransformTrace) -> tuple[float, float] | None:
    ow, oh = trace.original_size
    rw, rh = trace.resized_size
    x1, y1, _, _ = trace.crop_box
    out_w, out_h = trace.output_size
    x = point[0] * rw / ow - x1 + trace.pad_left
    y = point[1] * rh / oh - y1 + trace.pad_top
    if not (0 <= x < out_w and 0 <= y < out_h):
        return None
    return x, y


def map_box(
    box: tuple[float, float, float, float], trace: TransformTrace
) -> tuple[float, float, float, float] | None:
    x, y, width, height = box
    first = map_point((x, y), trace)
    second = map_point((x + width, y + height), trace)
    if first is None or second is None:
        # Clip partially visible boxes rather than pretending they disappeared.
        ow, oh = trace.original_size
        rw, rh = trace.resized_size
        cx, cy, _, _ = trace.crop_box
        out_w, out_h = trace.output_size
        values = (
            x * rw / ow - cx + trace.pad_left,
            y * rh / oh - cy + trace.pad_top,
            (x + width) * rw / ow - cx + trace.pad_left,
            (y + height) * rh / oh - cy + trace.pad_top,
        )
        clipped = (
            min(max(values[0], 0.0), float(out_w)),
            min(max(values[1], 0.0), float(out_h)),
            min(max(values[2], 0.0), float(out_w)),
            min(max(values[3], 0.0), float(out_h)),
        )
        return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None
    return first[0], first[1], second[0], second[1]


def token_indices_for_box(
    xyxy: tuple[float, float, float, float],
    *,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, ...]:
    x1, y1, x2, y2 = xyxy
    cols, rows = grid_size
    width, height = image_size
    output: list[int] = []
    for row in range(rows):
        cy = (row + 0.5) * height / rows
        for col in range(cols):
            cx = (col + 0.5) * width / cols
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                output.append(row * cols + col)
    return tuple(output)


def padded_union_crop(
    points: Sequence[tuple[float, float]],
    *,
    image_size: tuple[int, int],
    padding_fraction: float,
    minimum_fraction: float,
) -> tuple[int, int, int, int]:
    """Create one answer-label-independent crop from queried part geometry."""

    if not points:
        raise ValueError("oracle crop requires at least one visible queried part")
    if padding_fraction < 0 or not 0 < minimum_fraction <= 1:
        raise ValueError("invalid crop fractions")
    width, height = image_size
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    min_width, min_height = width * minimum_fraction, height * minimum_fraction
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span_x = max(max(xs) - min(xs), min_width) * (1 + 2 * padding_fraction)
    span_y = max(max(ys) - min(ys), min_height) * (1 + 2 * padding_fraction)
    side = min(max(span_x, span_y), float(max(width, height)))
    x1 = max(0.0, min(center_x - side / 2, width - side))
    y1 = max(0.0, min(center_y - side / 2, height - side))
    x2, y2 = min(float(width), x1 + side), min(float(height), y1 + side)
    return round(x1), round(y1), round(x2), round(y2)

