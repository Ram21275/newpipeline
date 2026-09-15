"""Label-free proposal rules and geometry for the CUB development bridge.

Selectors accept model maps only. Annotations belong to the separate evaluator.
These rules contain no fitted parameters and make no information-absence claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class Proposal:
    scores: torch.Tensor
    status: str
    top_rank: int = 0
    anchor_projection: float = 0.0


def normalized_maps(values: torch.Tensor) -> torch.Tensor:
    maps = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if maps.ndim != 2 or min(maps.shape) < 2:
        raise ValueError("attention must have shape [layers>=2, patches>=2]")
    if not bool(torch.isfinite(maps).all()) or bool((maps < 0).any()):
        raise ValueError("attention must be finite and nonnegative")
    mass = maps.sum(1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("each attention layer must have positive visual mass")
    return maps / mass


def select_depth_map(real: torch.Tensor, null: torch.Tensor, method: str = "spectral") -> Proposal:
    """Query-anchor orientation removes eigenvector sign/basis ambiguity.

    X contains centered, unit-length real-query maps. d is the real-minus-null
    layer mean. P projects onto the top eigenspace of XX^T (all eigenvalues
    within 5% of the largest). s = X^T P X d / lambda_max. Using the projector,
    rather than one arbitrary eigenvector, handles repeated top eigenvalues.
    A weak anchor or zero map variance falls back to the real layer mean.
    This identifies query-aligned variance, NOT ground-truth relevance.
    """
    a, b = normalized_maps(real), normalized_maps(null)
    if a.shape != b.shape:
        raise ValueError("real and null maps must share layers and spatial layout")
    mean, delta = a.mean(0), (a - b).mean(0)
    if method == "mean":
        return Proposal(mean, "mean")
    if method == "contrast":
        return Proposal(delta, "contrast")
    if method.startswith("fixed_"):
        numerator = int(method.split("_")[1])
        if numerator not in (1, 2, 3):
            raise ValueError("fixed layer must be one of the prespecified quartiles")
        return Proposal(a[min(len(a) - 1, numerator * len(a) // 4)], method)
    x = a - a.mean(1, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if method == "medoid":
        c = x @ x.T
        # Choose a medoid using agreement only; no target, location, or answer.
        medoid = int(c.median(dim=1).values.argmax())
        agree = c[medoid] >= 0.0
        return Proposal(a[agree].median(dim=0).values, "medoid")
    if method != "spectral":
        raise ValueError(f"unknown selector: {method}")
    values, vectors = torch.linalg.eigh(x @ x.T)
    largest = float(values[-1])
    if largest < 1e-10 or float(delta.norm()) < 1e-10:
        return Proposal(mean, "fallback_unidentified")
    top = vectors[:, values >= 0.95 * largest]
    score = x.T @ top @ (top.T @ (x @ delta)) / largest
    ratio = float(score.norm() / delta.norm())
    if ratio < 1e-6:
        return Proposal(mean, "fallback_orthogonal_anchor", top.shape[1], ratio)
    return Proposal(score, "spectral", top.shape[1], ratio)


def crop_at(cx: float, cy: float, width: float) -> tuple[float, float, float, float]:
    if not 0 < width <= 1 or not all(math.isfinite(x) for x in (cx, cy)):
        raise ValueError("invalid crop geometry")
    left = min(max(cx - width / 2, 0.0), 1 - width)
    top = min(max(cy - width / 2, 0.0), 1 - width)
    return left, top, left + width, top + width


def grid_centers(grid: tuple[int, int]) -> torch.Tensor:
    h, w = grid
    if h <= 0 or w <= 0:
        raise ValueError("grid dimensions must be positive")
    y, x = torch.meshgrid((torch.arange(h) + 0.5) / h,
                          (torch.arange(w) + 0.5) / w, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=1).double()


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))


def choose_crops(scores: torch.Tensor, grid: tuple[int, int], *, width: float,
                 count: int = 1) -> list[tuple[float, float, float, float]]:
    """Greedy new score mass: a cell covered once has no further utility.

    Tie-breaking is stable row-major. No outer-ring mask or center prior.
    count=2 is a secondary fixed-budget context/coverage ablation.
    """
    centers = grid_centers(grid)
    score = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if len(score) != len(centers) or not bool(torch.isfinite(score).all()):
        raise ValueError("score does not match the finite spatial grid")
    if count not in (1, 2):
        raise ValueError("the bridge supports one or two crops")
    # Positive signed mass; if absent, a deterministic uniform map.
    utility = score.clamp_min(0)
    if float(utility.sum()) <= 1e-12:
        utility = torch.ones_like(utility)
    windows = [crop_at(float(x), float(y), width) for x, y in centers]
    masks = torch.stack([inside(centers, box) for box in windows])
    selected, covered = [], torch.zeros(len(score), dtype=torch.bool)
    for _ in range(count):
        gains = ((masks & ~covered) * utility).sum(1)
        # Soft overlap penalty breaks ties toward distinct evidence regions.
        if selected:
            overlap = torch.tensor([max(intersection_area(w, s) / width**2
                                       for s in selected) for w in windows])
            gains = gains - 0.25 * utility.sum() * overlap
        best = int(gains.argmax())
        selected.append(windows[best])
        covered |= masks[best]
    return selected


def inside(points: torch.Tensor, box: Sequence[float]) -> torch.Tensor:
    points = torch.as_tensor(points, dtype=torch.float64)
    return ((points[:, 0] >= box[0]) & (points[:, 0] <= box[2]) &
            (points[:, 1] >= box[1]) & (points[:, 1] <= box[3]))


def random_crops(grid: tuple[int, int], *, width: float, count: int,
                 image_id: str, attribute_id: str, seed: int) -> list[tuple[float, ...]]:
    key = f"{image_id}:{attribute_id}:{seed}".encode()
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    order = torch.randperm(grid[0] * grid[1], generator=generator)
    centers = grid_centers(grid)
    return [crop_at(*centers[int(i)].tolist(), width) for i in order[:count]]


def pixel_box(box: Sequence[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    left, top = math.floor(box[0] * w), math.floor(box[1] * h)
    right, bottom = math.ceil(box[2] * w), math.ceil(box[3] * h)
    return max(0, left), max(0, top), min(w, right), min(h, bottom)


def realized_box(box: Sequence[float], size: tuple[int, int]) -> tuple[float, ...]:
    l, t, r, b = pixel_box(box, size)
    return l / size[0], t / size[1], r / size[0], b / size[1]


def proxy_token_coverage(points: torch.Tensor, boxes: Sequence[Sequence[float]],
                         grids: Sequence[tuple[int, int]], radius: float) -> list[float]:
    """Area-integrated tokens in landmark squares, not true attribute extent.

    Returns per-landmark exposure summed over views, including repeated pixels.
    Radius is in original image fractions. This is not a segmentation estimate.
    """
    if len(boxes) != len(grids) or not 0 < radius < 0.5:
        raise ValueError("invalid proxy geometry")
    result = []
    for x, y in torch.as_tensor(points).tolist():
        region = (max(0, x-radius), max(0, y-radius), min(1, x+radius), min(1, y+radius))
        count = 0.0
        for box, (h, w) in zip(boxes, grids):
            area = (box[2]-box[0]) * (box[3]-box[1])
            count += h * w * intersection_area(region, box) / area
        result.append(count)
    return result


def cluster_interval(image_ids: Sequence[str], differences: Sequence[float], *,
                     seed: int = 1609, draws: int = 2000, alpha: float = 0.05) -> dict[str, float | int | list]:
    """Decision-weighted paired estimate; resample entire image clusters."""
    if len(image_ids) != len(differences) or not image_ids or not 0 < alpha < 1:
        raise ValueError("need aligned nonempty paired differences")
    ids = sorted(set(image_ids))
    values = torch.tensor(differences, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or len(ids) < 2:
        raise ValueError("need finite differences and at least two image clusters")
    sums = torch.tensor([sum(d for i, d in zip(image_ids, differences) if i == key) for key in ids])
    counts = torch.tensor([image_ids.count(key) for key in ids])
    rng = torch.Generator().manual_seed(seed)
    samples = torch.randint(len(ids), (draws, len(ids)), generator=rng)
    bootstrap = sums[samples].sum(1) / counts[samples].sum(1)
    return {"estimate": float(values.mean()), "interval": torch.quantile(
        bootstrap.double(), torch.tensor([alpha/2, 1-alpha/2], dtype=torch.float64)).tolist(),
        "confidence": 1-alpha, "images": len(ids), "decisions": len(image_ids), "draws": draws}
