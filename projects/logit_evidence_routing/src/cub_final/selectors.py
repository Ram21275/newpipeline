"""Named, architecture-scoped visual-token selectors."""

from __future__ import annotations

import math
import random
from typing import Sequence


def stable_topk(scores: Sequence[float], k: int) -> tuple[int, ...]:
    if not 0 < k <= len(scores):
        raise ValueError("k must be within the score vector")
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("selector scores must be finite")
    return tuple(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k])


def decoder_attention_selector(scores: Sequence[float], k: int) -> tuple[int, ...]:
    return stable_topk(scores, k)


def logit_concept_selector(scores: Sequence[float], k: int) -> tuple[int, ...]:
    return stable_topk(scores, k)


def vision_cls_selector(
    scores: Sequence[float], k: int, *, architecture: str
) -> tuple[int, ...]:
    if architecture != "llava":
        raise ValueError("Vision-CLS is LLaVA-specific and is not a Qwen3 replication selector")
    return stable_topk(scores, k)


def random_selector(token_count: int, k: int, *, seed: int) -> tuple[int, ...]:
    if not 0 < k <= token_count:
        raise ValueError("k must be within token_count")
    return tuple(sorted(random.Random(seed).sample(range(token_count), k)))


def predicted_square_crop(
    selected_token: int,
    *,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    crop_fraction: float,
) -> tuple[int, int, int, int]:
    """Map a selected merged token to one crop in original-image coordinates."""

    cols, rows = grid_size
    width, height = image_size
    if not 0 <= selected_token < cols * rows or not 0 < crop_fraction <= 1:
        raise ValueError("invalid token index or crop fraction")
    row, col = divmod(selected_token, cols)
    center_x = (col + 0.5) * width / cols
    center_y = (row + 0.5) * height / rows
    side = min(width, height) * crop_fraction
    x1 = max(0.0, min(center_x - side / 2, width - side))
    y1 = max(0.0, min(center_y - side / 2, height - side))
    return round(x1), round(y1), round(x1 + side), round(y1 + side)

