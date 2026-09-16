"""Image-clustered paired bootstrap with simultaneous max-t intervals."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Iterable


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile needs values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def paired_image_cluster_bootstrap(
    rows: Iterable[dict[str, Any]],
    *,
    effect_field: str,
    image_field: str = "image_id",
    family_field: str | None = None,
    resamples: int = 10_000,
    seed: int = 20260916,
    weighting: str = "image",
) -> dict[str, Any]:
    """Bootstrap complete image clusters, preserving all within-image rows."""

    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if weighting not in {"image", "decision"}:
        raise ValueError("weighting must be image or decision")
    materialized = list(rows)
    if not materialized:
        raise ValueError("bootstrap input is empty")
    by_family_image: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in materialized:
        family = str(row[family_field]) if family_field else "overall"
        value = float(row[effect_field])
        if not math.isfinite(value):
            raise ValueError("bootstrap effects must be finite")
        by_family_image[family][str(row[image_field])].append(value)
    families = sorted(by_family_image)
    image_ids = sorted({image for values in by_family_image.values() for image in values})
    if len(image_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two images")

    def estimate(family: str, sampled: list[str] | None = None) -> float:
        mapping = by_family_image[family]
        keys = sampled if sampled is not None else sorted(mapping)
        if weighting == "image":
            values = [mean(mapping[key]) for key in keys if key in mapping]
        else:
            values = [value for key in keys if key in mapping for value in mapping[key]]
        return mean(values)

    point = {family: estimate(family) for family in families}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {family: [] for family in families}
    for _ in range(resamples):
        sampled = [rng.choice(image_ids) for _ in image_ids]
        for family in families:
            if any(key in by_family_image[family] for key in sampled):
                draws[family].append(estimate(family, sampled))
    intervals: dict[str, Any] = {}
    standardized: list[list[float]] = []
    scales: dict[str, float] = {}
    for family in families:
        values = draws[family]
        scale = pstdev(values)
        scales[family] = scale
        intervals[family] = {
            "estimate": point[family],
            "ci_low": _quantile(values, 0.025),
            "ci_high": _quantile(values, 0.975),
            "bootstrap_se": scale,
            "image_count": len(by_family_image[family]),
            "decision_count": sum(len(values) for values in by_family_image[family].values()),
        }
    if len(families) > 1 and all(scales[family] > 0 for family in families):
        for index in range(resamples):
            standardized.append(
                [
                    abs(draws[family][index] - point[family]) / scales[family]
                    for family in families
                ]
            )
        critical = _quantile([max(values) for values in standardized], 0.95)
        for family in families:
            intervals[family]["simultaneous_ci_low"] = point[family] - critical * scales[family]
            intervals[family]["simultaneous_ci_high"] = point[family] + critical * scales[family]
        simultaneous = {"method": "bootstrap_max_t", "critical_value": critical}
    else:
        simultaneous = {"method": "not_applicable", "critical_value": None}
    return {
        "resamples": resamples,
        "seed": seed,
        "cluster_unit": image_field,
        "weighting": weighting,
        "families": intervals,
        "simultaneous": simultaneous,
    }

