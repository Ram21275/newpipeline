#!/usr/bin/env python3
"""Plot Phase 01B same-backbone localization maps and CUB boxes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.localization import (  # noqa: E402
    patch_centers_in_box,
    selection_localization_metrics,
)
from lger.phase1b import DETERMINISTIC_LOCALIZERS, feature_key  # noqa: E402
from lger.probe import jaccard  # noqa: E402


def resized_heatmap(
    scores: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> torch.Tensor:
    scores = scores.float()
    scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-8)
    grid = scores.reshape(1, 1, *grid_size)
    return F.interpolate(
        grid,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def add_bbox(axis: object, bbox: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = bbox
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=1.8,
        )
    )


def plot_record(record: dict[str, object], destination: Path, k: int) -> None:
    image = record["processed_image"].permute(1, 2, 0)
    height, width = image.shape[:2]
    grid_size = tuple(int(value) for value in record["grid_size"])
    bbox = tuple(float(value) for value in record["bbox_xyxy_model"])
    bbox_mask = patch_centers_in_box(grid_size, (height, width), bbox)
    rows, columns = grid_size

    figure, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
    axes_flat = axes.flatten()
    for axis in axes_flat:
        axis.axis("off")
    axes_flat[0].imshow(image)
    add_bbox(axes_flat[0], bbox)
    axes_flat[0].set_title("Model input + CUB box")

    for axis, selector in zip(axes_flat[1:], DETERMINISTIC_LOCALIZERS):
        scores = record["score_maps"][selector]
        selected = record["selections"][feature_key(selector, k)]
        metrics = selection_localization_metrics(selected, bbox_mask)
        heatmap = resized_heatmap(scores, grid_size, (height, width))
        axis.imshow(image)
        axis.imshow(heatmap, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        add_bbox(axis, bbox)
        selected_rows = selected // columns
        selected_columns = selected % columns
        axis.scatter(
            (selected_columns.float() + 0.5) * width / columns,
            (selected_rows.float() + 0.5) * height / rows,
            s=10,
            c="cyan",
            marker="o",
            linewidths=0,
        )
        axis.set_title(
            f"{selector}\n"
            f"inside={metrics['inside_fraction']:.2f}, "
            f"point={metrics['pointing_game']:.0f}"
        )

    attention = record["selections"][feature_key("llm_attention", k)]
    concept = record["selections"][feature_key("logit_concept", k)]
    figure.suptitle(
        f"{record['class_name']} | image {record['image_id']} | Top-{k} | "
        f"attention/concept Jaccard={jaccard(attention, concept):.3f}",
        fontsize=14,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.k <= 0 or args.count <= 0:
        raise ValueError("K and count must be positive")

    ranked: list[tuple[float, dict[str, object]]] = []
    for path in sorted((args.cache_dir / "records").glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if int(record.get("schema_version", 0)) != 2 or record["split"] != "val":
            continue
        attention = record["selections"][feature_key("llm_attention", args.k)]
        concept = record["selections"][feature_key("logit_concept", args.k)]
        ranked.append((jaccard(attention, concept), record))
    ranked.sort(key=lambda item: (item[0], int(item[1]["image_id"])))
    selected_records = ranked[: args.count]
    if not selected_records:
        raise RuntimeError("No compatible validation cache records were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for overlap, record in selected_records:
        destination = args.output_dir / f"{int(record['image_id']):05d}.png"
        plot_record(record, destination, args.k)
        index_rows.append(
            {
                "image_id": record["image_id"],
                "class_name": record["class_name"],
                "llm_attention_logit_concept_jaccard": overlap,
                "figure": destination.name,
            }
        )
    with (args.output_dir / "index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(selected_records)} localization figures to {args.output_dir}")


if __name__ == "__main__":
    main()
