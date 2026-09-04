#!/usr/bin/env python3
"""Create the required attention-vs-logit disagreement figures from the cache."""

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

from lger.probe import jaccard  # noqa: E402


def feature_key(selector: str, k: int) -> str:
    return f"{selector}_k{k}"


def resized_heatmap(scores: torch.Tensor, height: int, width: int) -> torch.Tensor:
    scores = scores.float()
    scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-8)
    side = int(scores.numel() ** 0.5)
    grid = scores.reshape(1, 1, side, side)
    return F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)[0, 0]


def plot_record(record: dict[str, object], destination: Path, k: int) -> None:
    image = record["processed_image"].permute(1, 2, 0)
    height, width = image.shape[:2]
    attention_scores = record["attention_scores"]
    logit_scores = record["evidence_scores"]["maxprob"]
    attention_map = resized_heatmap(attention_scores, height, width)
    logit_map = resized_heatmap(logit_scores, height, width)
    logit_indices = record["selections"][feature_key("logit", k)]
    rows, columns = record["grid_size"]

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(image)
    axes[0].set_title("Model input")
    axes[1].imshow(image)
    axes[1].imshow(attention_map, cmap="magma", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Attention evidence")
    axes[2].imshow(image)
    axes[2].imshow(logit_map, cmap="viridis", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Logit max-probability")
    axes[3].imshow(image)
    token_lookup = {
        int(index): tokens
        for index, tokens in zip(
            record["decoded_logit_patch_indices"].tolist(),
            record["decoded_logit_top_tokens"],
        )
    }
    cell_width = width / columns
    cell_height = height / rows
    for rank, patch_index in enumerate(logit_indices.tolist()):
        row, column = divmod(int(patch_index), columns)
        axes[3].add_patch(
            Rectangle(
                (column * cell_width, row * cell_height),
                cell_width,
                cell_height,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
            )
        )
        if rank < 6:
            tokens = ", ".join(token_lookup.get(int(patch_index), [])[:2])
            axes[3].text(
                column * cell_width,
                row * cell_height,
                tokens,
                color="white",
                fontsize=6,
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 1},
            )
    axes[3].set_title(f"Logit Top-{k} + decoded tokens")
    figure.suptitle(
        f"{record['class_name']} | image {record['image_id']} | "
        f"attention/logit Jaccard={jaccard(record['selections'][feature_key('attention', k)], logit_indices):.3f}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    ranked: list[tuple[float, dict[str, object]]] = []
    for path in sorted((args.cache_dir / "records").glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record["split"] != "val":
            continue
        attention = record["selections"][feature_key("attention", args.k)]
        logit = record["selections"][feature_key("logit", args.k)]
        ranked.append((jaccard(attention, logit), record))
    ranked.sort(key=lambda item: (item[0], int(item[1]["image_id"])))
    selected = ranked[: args.count]
    if not selected:
        raise RuntimeError("No validation cache records were found")

    index_rows: list[dict[str, object]] = []
    for overlap, record in selected:
        destination = args.output_dir / f"{int(record['image_id']):05d}.png"
        plot_record(record, destination, args.k)
        index_rows.append(
            {
                "image_id": record["image_id"],
                "class_name": record["class_name"],
                "attention_logit_jaccard": overlap,
                "figure": destination.name,
            }
        )
    with (args.output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(selected)} disagreement figures to {args.output_dir}")


if __name__ == "__main__":
    main()
