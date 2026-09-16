"""Predeclared matched-control construction for every intervention dose."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MatchResult:
    selected: tuple[int, ...]
    control: tuple[int, ...]
    feasible: bool
    policy: str
    selected_mean_norm: float
    control_mean_norm: float
    relative_norm_gap: float


def _mean(values: Sequence[float], indices: Sequence[int]) -> float:
    return sum(float(values[index]) for index in indices) / len(indices)


def norm_matched_random(
    selected: Sequence[int],
    norms: Sequence[float],
    *,
    seed: int,
    spatial_bins: Sequence[str] | None = None,
    strict_spatial: bool = False,
) -> MatchResult:
    """Match selected tokens without dose-specific policy fallbacks.

    A deterministic global assignment greedily matches closest log norms.  The
    primary series uses ``strict_spatial=False`` at every K.  The sensitivity
    series uses ``strict_spatial=True`` and reports infeasible decisions rather
    than silently changing policy.
    """

    chosen = tuple(sorted({int(value) for value in selected}))
    if not chosen or len(chosen) != len(selected):
        raise ValueError("selected indices must be unique and non-empty")
    if any(index < 0 or index >= len(norms) for index in chosen):
        raise ValueError("selected index is out of range")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in norms):
        raise ValueError("norms must be finite and non-negative")
    if strict_spatial and (spatial_bins is None or len(spatial_bins) != len(norms)):
        raise ValueError("strict spatial matching needs one bin per token")

    rng = random.Random(seed)
    candidates = [index for index in range(len(norms)) if index not in chosen]
    rng.shuffle(candidates)
    matched: list[int] = []
    for source in sorted(chosen, key=lambda index: (float(norms[index]), index)):
        eligible = [
            index
            for index in candidates
            if not strict_spatial or spatial_bins[index] == spatial_bins[source]
        ]
        if not eligible:
            return MatchResult(
                selected=chosen,
                control=tuple(),
                feasible=False,
                policy="spatial_and_norm" if strict_spatial else "norm",
                selected_mean_norm=_mean(norms, chosen),
                control_mean_norm=float("nan"),
                relative_norm_gap=float("nan"),
            )
        target = math.log1p(float(norms[source]))
        best = min(
            eligible,
            key=lambda index: (abs(math.log1p(float(norms[index])) - target), index),
        )
        matched.append(best)
        candidates.remove(best)
    selected_mean = _mean(norms, chosen)
    control_mean = _mean(norms, matched)
    gap = abs(selected_mean - control_mean) / max(abs(selected_mean), 1e-12)
    return MatchResult(
        selected=chosen,
        control=tuple(sorted(matched)),
        feasible=True,
        policy="spatial_and_norm" if strict_spatial else "norm",
        selected_mean_norm=selected_mean,
        control_mean_norm=control_mean,
        relative_norm_gap=gap,
    )

