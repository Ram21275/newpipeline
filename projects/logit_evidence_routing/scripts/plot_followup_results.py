#!/usr/bin/env python3
"""Plot Phase 9 mechanism and cross-model VQA follow-up confidence intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "vision_cls_attention": "#0072B2",
    "logit_concept": "#D55E00",
    "llava": "#0072B2",
    "qwen": "#009E73",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise RuntimeError(f"expected a passing JSON report: {path}")
    return value


def save(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def forest(axis: Any, rows: list[dict[str, Any]], labels: list[str], colors: list[str]) -> None:
    positions = list(range(len(rows)))[::-1]
    for position, row, color in zip(positions, rows, colors):
        estimate = float(row["estimate"])
        low, high = float(row["ci_low"]), float(row["ci_high"])
        axis.errorbar(
            estimate, position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o", color=color, capsize=3, linewidth=1.5,
        )
    axis.axvline(0, color="#333333", linestyle=":", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)


def plot_phase9(report: dict[str, Any], output_dir: Path) -> list[Path]:
    selectors = report["selector_specific_contrasts"]
    interactions = report["stage_interactions"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    forest(
        axes[0], selectors,
        [f"{row['stage']} · {row['selection_method'].replace('_', ' ')}" for row in selectors],
        [COLORS.get(str(row["selection_method"]), "#666666") for row in selectors],
    )
    axes[0].set_title("Top evidence minus matched random")
    axes[0].set_xlabel("Additional correct-answer margin drop")
    forest(
        axes[1], interactions,
        [row["selection_method"].replace("_", " ") for row in interactions],
        [COLORS.get(str(row["selection_method"]), "#666666") for row in interactions],
    )
    axes[1].set_title("Vision late minus projector interaction")
    axes[1].set_xlabel("Difference in selector-specific effect")
    figure.suptitle("Phase 9 development mechanism follow-up (95% image-cluster bootstrap CIs)")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return save(figure, output_dir, "figure_6_phase9_selector_interventions")


def plot_replication(report: dict[str, Any], output_dir: Path) -> list[Path]:
    rows = [row for row in report["paired_contrasts"]
            if row["metric"] == "correct_answer_margin"]
    order = {"prompt_only": 0, "image_shuffled": 1, "opposite_label_image": 2}
    rows.sort(key=lambda row: (
        report["model_labels"].index(row["model_label"]),
        order[row["contrast"].replace("image_minus_", "")],
    ))
    labels = [
        f"{row['model_label']} · image − "
        f"{row['contrast'].replace('image_minus_', '').replace('_', ' ')}"
        for row in rows
    ]
    colors = [COLORS.get(str(row["model_label"]).casefold(), "#666666") for row in rows]
    figure, axis = plt.subplots(figsize=(9.2, max(4.4, 0.55 * len(rows) + 1.5)))
    forest(axis, rows, labels, colors)
    axis.set_xlabel("Correct-answer teacher-forced margin difference")
    axis.set_title("Correct-image advantage across models and controls\n"
                   "95% image-cluster bootstrap confidence intervals")
    figure.tight_layout()
    return save(figure, output_dir, "figure_7_cross_model_visual_utilization")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase9-analysis", type=Path, required=True)
    parser.add_argument("--replication-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    phase9 = read_json(args.phase9_analysis)
    replication = read_json(args.replication_analysis)
    if phase9.get("official_test_images_used") != 0 \
            or replication.get("official_test_images_used") != 0:
        raise RuntimeError("this plotting script accepts development results only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = plot_phase9(phase9, args.output_dir)
    outputs += plot_replication(replication, args.output_dir)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "paper_figures_from_phase9_and_cross_model_development_results",
        "official_test_images_used": 0,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in outputs
        },
    }
    target = args.output_dir / "followup_figure_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
