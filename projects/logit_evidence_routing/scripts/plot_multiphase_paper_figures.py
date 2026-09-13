#!/usr/bin/env python3
"""Create paper-ready development figures from saved multiphase artifacts.

This script never loads a model.  It verifies the compact package manifest,
uses only recorded summary estimates, and labels panels without sampling
intervals as descriptive.  When the reviewed Phase 4 bundle is supplied, the
localization figure includes every selector and a fixed qualitative gallery.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


COLORS = {
    "primary": "#0072B2",
    "prevalence": "#666666",
    "shuffled_labels": "#D55E00",
    "random_projection": "#009E73",
    "image": "#0072B2",
    "prompt_only": "#E69F00",
    "image_shuffled": "#999999",
    "logit_concept": "#D55E00",
    "vision_cls_attention": "#0072B2",
    "attention_logit_fusion": "#CC79A7",
    "llm_attention": "#56B4E9",
    "random": "#999999",
    "dense_object": "#009E73",
    "dense_attribute": "#F0E442",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"empty CSV: {path}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_paper_package(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "paper_package_manifest.json")
    require(manifest.get("status") == "PASS", "paper package is not PASS")
    for name, expected in manifest.get("artifacts", {}).items():
        path = root / name
        require(path.is_file(), f"manifest artifact is missing: {name}")
        require(path.stat().st_size == int(expected["bytes"]), f"size mismatch: {name}")
        require(sha256(path) == expected["sha256"], f"SHA-256 mismatch: {name}")
    return manifest


def mean(values: Iterable[float]) -> float:
    values = list(values)
    require(bool(values), "cannot average an empty sequence")
    return float(sum(values) / len(values))


def clean_axes(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return outputs


def plot_phase3(rows: Sequence[Mapping[str, str]], output_dir: Path) -> list[Path]:
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    labels = [stage.replace(".", "\n") for stage in stages]
    x = np.arange(len(stages))
    figure, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    for axis, metric, ylabel in (
        (axes[0], "macro_auroc", "Macro AUROC"),
        (axes[1], "macro_f1", "Macro F1"),
    ):
        for control in ("primary", "random_projection", "shuffled_labels", "prevalence"):
            centers, lows, highs = [], [], []
            for stage in stages:
                values = [float(row[metric]) for row in rows
                          if row["stage"] == stage and row["control"] == control]
                centers.append(mean(values))
                lows.append(min(values))
                highs.append(max(values))
            label = control.replace("_", " ")
            axis.plot(x, centers, marker="o", linewidth=2 if control == "primary" else 1.4,
                      color=COLORS[control], label=label, zorder=3)
            if len(values) > 1 and control != "prevalence":
                axis.fill_between(x, lows, highs, color=COLORS[control], alpha=0.12)
        axis.axvline(3.5, color="#444444", linewidth=0.8, linestyle=":")
        axis.axvline(4.5, color="#444444", linewidth=0.8, linestyle=":")
        axis.set_ylabel(ylabel)
        clean_axes(axis)
    axes[0].legend(ncol=2, frameon=False, loc="lower right")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Measured frozen stage")
    figure.suptitle("Phase 3R: linear accessibility across the model")
    figure.text(
        0.5, 0.01,
        "Lines show means across three probe seeds (prevalence has one row); bands show seed ranges, not sampling CIs.\n"
        "vision.final is an auxiliary measured state; vision.late supplies the projector.",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.96))
    return save_figure(figure, output_dir, "figure_1_phase3_accessibility")


def _macro_phase4(
    rows: Sequence[Mapping[str, str]], selector: str, metric: str
) -> float | None:
    selected = [row for row in rows if row["split"] == "val"
                and row["attribute_group"] == "ALL_SELECTED_ATTRIBUTES"
                and row["selector"] == selector and row["K"] == "32"
                and row["metric"] == metric]
    return float(selected[0]["mean_across_evaluable_attributes"]) if selected else None


def _object_phase4(
    rows: Sequence[Mapping[str, str]], selector: str, metric: str
) -> float | None:
    selected = [row for row in rows if row["split"] == "val"
                and row["selector"] == selector and row["K"] == "32"]
    if not selected:
        return None
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        by_image[row["image_id"]].append(float(row[metric]))
    return mean(mean(values) for values in by_image.values())


def plot_phase4(
    paper_rows: Sequence[Mapping[str, str]], output_dir: Path,
    phase4_dir: Path | None,
) -> list[Path]:
    if phase4_dir is not None:
        macro = read_csv(phase4_dir / "attribute_macro_summary.csv")
        objects = read_csv(phase4_dir / "object_metrics.csv")
        selectors = ["random", "llm_attention", "dense_object", "dense_attribute",
                     "attention_logit_fusion", "logit_concept", "vision_cls_attention"]
    else:
        macro = list(paper_rows)
        objects = []
        selectors = ["logit_concept", "vision_cls_attention"]
    shown = [selector for selector in selectors
             if _macro_phase4(macro, selector, "part_patch_recall") is not None]
    x = np.arange(len(shown))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    landmark = [_macro_phase4(macro, selector, "part_patch_recall") for selector in shown]
    any_hit = [_macro_phase4(macro, selector, "any_part_hit") for selector in shown]
    width = 0.36
    axes[0].bar(x - width / 2, landmark, width, color=[COLORS[s] for s in shown],
                edgecolor="white", label="landmark-patch recall")
    axes[0].bar(x + width / 2, any_hit, width, color=[COLORS[s] for s in shown],
                alpha=0.45, edgecolor="white", label="any landmark hit")
    axes[0].set_ylabel("Attribute macro score")
    axes[0].set_ylim(0, max(any_hit) * 1.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Attribute landmark coverage")
    axes[0].set_xticks(x, [s.replace("_", "\n") for s in shown], fontsize=7.5)
    clean_axes(axes[0])
    object_shown = [selector for selector in shown
                    if _object_phase4(objects, selector, "inside_fraction") is not None]
    if object_shown:
        ox = np.arange(len(object_shown))
        inside = [_object_phase4(objects, selector, "inside_fraction") for selector in object_shown]
        pointing = [_object_phase4(objects, selector, "pointing_game") for selector in object_shown]
        axes[1].bar(ox - width / 2, inside, width, color=[COLORS[s] for s in object_shown],
                    edgecolor="white", label="Top-32 inside box")
        axes[1].bar(ox + width / 2, pointing, width, color=[COLORS[s] for s in object_shown],
                    alpha=0.45, edgecolor="white", label="Top-1 in box")
        axes[1].set_xticks(ox, [s.replace("_", "\n") for s in object_shown], fontsize=7.5)
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Supply --phase4-dir for the full object-localization panel.",
                     ha="center", va="center", wrap=True)
    axes[1].set_ylabel("Object score")
    axes[1].set_title("Bird-object concentration")
    clean_axes(axes[1])
    figure.suptitle("Phase 4: localization is selector- and metric-dependent")
    figure.text(0.5, 0.015,
                "Development validation, K=32. Bars are descriptive means; no sampling CIs are available in the source bundle.",
                ha="center", fontsize=8.5, color="#444444")
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    return save_figure(figure, output_dir, "figure_2_phase4_localization")


def plot_phase5(rows: Sequence[Mapping[str, str]], output_dir: Path) -> list[Path]:
    controls = ["image", "prompt_only", "image_shuffled"]
    cohorts = ["grounded_positive", "grounded_negative"]
    labels = ["Correct image", "Prompt only", "Shuffled image"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(len(cohorts))
    width = 0.24
    for index, control in enumerate(controls):
        selected = [{row["cohort"]: row for row in rows if row["control"] == control}[cohort]
                    for cohort in cohorts]
        axes[0].bar(x + (index - 1) * width,
                    [100 * float(row["margin_accuracy"]) for row in selected], width,
                    color=COLORS[control], label=labels[index])
        axes[1].bar(x + (index - 1) * width,
                    [float(row["mean_signed_answer_margin"]) for row in selected], width,
                    color=COLORS[control], label=labels[index])
    axes[0].axhline(50, color="#333333", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Teacher-forced accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Binary decision accuracy")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_ylabel("Mean correct-answer margin")
    axes[1].set_title("Signed likelihood margin")
    for axis in axes:
        axis.set_xticks(x, ["Attribute present", "Attribute absent"])
        clean_axes(axis)
    figure.suptitle("Phase 5: the correct image helps, amid a strong yes bias")
    figure.text(0.5, 0.015,
                "Equal-attribute macro means over 417 decisions per label cohort. No paired sampling CIs are available in the compact package.",
                ha="center", fontsize=8.5, color="#444444")
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    return save_figure(figure, output_dir, "figure_3_phase5_utilization")


def _forest(
    rows: Sequence[Mapping[str, str]], labels: Sequence[str], output_dir: Path,
    stem: str, title: str, xlabel: str,
) -> list[Path]:
    require(len(rows) == len(labels) and rows, "forest plot rows and labels must align")
    estimates = np.array([float(row["estimate"]) for row in rows])
    low = np.array([float(row["ci_low"]) for row in rows])
    high = np.array([float(row["ci_high"]) for row in rows])
    y = np.arange(len(rows))[::-1]
    height = max(3.2, 0.48 * len(rows) + 1.6)
    figure, axis = plt.subplots(figsize=(9.5, height))
    axis.axvline(0, color="#333333", linewidth=1)
    colors = ["#D55E00" if not (lo <= 0 <= hi) else "#777777"
              for lo, hi in zip(low, high)]
    for estimate, lower, upper, y_value, color in zip(estimates, low, high, y, colors):
        axis.errorbar(
            estimate,
            y_value,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
        )
    axis.scatter(estimates, y, c=colors, s=34, zorder=3)
    axis.set_yticks(y, labels)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    clean_axes(axis)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    axis.grid(axis="y", visible=False)
    figure.text(0.5, 0.015, "Points are estimates; bars are recorded 95% image-clustered intervals.",
                ha="center", fontsize=8.5, color="#444444")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    return save_figure(figure, output_dir, stem)


def plot_phase6(rows: Sequence[Mapping[str, str]], output_dir: Path) -> list[Path]:
    paired_order = [
        ("vision_minus_language_probe", "Vision − language probe correctness"),
        ("language_probe_minus_answer", "Language probe − answer correctness"),
        ("discriminative_minus_semantic_localization", "Vision-CLS − Logit Lens recall"),
        ("language_minus_vision_evidence_entropy", "Language − vision entropy"),
        ("spatial_evidence_redistribution", "Top-32 redistribution (1 − Jaccard)"),
    ]
    paired = {row["contrast"]: row for row in rows if row["kind"] == "paired"}
    outputs = _forest(
        [paired[name] for name, _ in paired_order], [label for _, label in paired_order],
        output_dir, "figure_4_phase6_gate",
        "Phase 6: preregistered development-gate diagnostics", "Recorded effect",
    )
    failure = [row for row in rows if row["kind"] == "failure_success"]
    failure_labels = [row["contrast"].replace("failure_minus_success__", "").replace("_", " ")
                      for row in failure]
    outputs += _forest(
        failure, failure_labels, output_dir, "figure_s1_phase6_failure_diagnostics",
        "Phase 6: failures minus successes", "Failure − success effect",
    )
    return outputs


def plot_phase7(rows: Sequence[Mapping[str, str]], output_dir: Path) -> list[Path]:
    ordered = sorted(rows, key=lambda row: (
        0 if row["stage_role"] == "selected" else 1,
        0 if row["metric"] == "correct_answer_teacher_forced_margin_drop" else 1,
        0 if row["reference"] == "matched_random" else 1,
    ))
    labels = [
        f"{row['stage']} · {row['reference'].replace('_', ' ')} · "
        f"{'margin' if row['metric'].startswith('correct') else 'generated accuracy'}"
        for row in ordered
    ]
    return _forest(
        ordered, labels, output_dir, "figure_5_phase7_interventions",
        "Phase 7: top-evidence replacement minus control replacement",
        "Extra drop caused by top-evidence replacement",
    )


def qualitative_gallery(phase4_dir: Path, output_dir: Path) -> list[Path]:
    paths = sorted((phase4_dir / "qualitative").glob("*.png"))
    if not paths:
        return []
    columns = 3
    rows = math.ceil(len(paths) / columns)
    thumb_size = (900, 620)
    figure, axes = plt.subplots(rows, columns, figsize=(13.5, 4.1 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    for axis in axes.flat:
        axis.axis("off")
    for axis, path in zip(axes.flat, paths):
        with Image.open(path) as opened:
            image = ImageOps.contain(opened.convert("RGB"), thumb_size)
            axis.imshow(image)
        axis.set_title(path.stem.replace("_", " · "), fontsize=8)
    figure.suptitle("Phase 4 fixed qualitative outputs (all returned panels)")
    figure.text(0.5, 0.008,
                "Panels are displayed in filename order; no example was selected by outcome.",
                ha="center", fontsize=8.5, color="#444444")
    figure.tight_layout(rect=(0, 0.025, 1, 0.975))
    return save_figure(figure, output_dir, "figure_s2_phase4_qualitative_gallery")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-package", type=Path, required=True)
    parser.add_argument("--phase4-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_paper_package(args.paper_package)
    if args.phase4_dir is not None:
        require(args.phase4_dir.is_dir(), "--phase4-dir is not a directory")
    outputs: list[Path] = []
    outputs += plot_phase3(read_csv(args.paper_package / "table_phase3_accessibility.csv"), args.output_dir)
    outputs += plot_phase4(read_csv(args.paper_package / "table_phase4_localization.csv"),
                           args.output_dir, args.phase4_dir)
    outputs += plot_phase5(read_csv(args.paper_package / "table_phase5_vqa.csv"), args.output_dir)
    outputs += plot_phase6(read_csv(args.paper_package / "table_phase6_joint_inference.csv"), args.output_dir)
    outputs += plot_phase7(read_csv(args.paper_package / "table_phase7_causal.csv"), args.output_dir)
    if args.phase4_dir is not None:
        outputs += qualitative_gallery(args.phase4_dir, args.output_dir)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "paper_figures_from_saved_development_results",
        "source_package": str(args.paper_package.resolve()),
        "source_package_selected_transition": manifest["selected_transition"],
        "phase4_source": str(args.phase4_dir.resolve()) if args.phase4_dir else None,
        "official_test_images_used": 0,
        "model_extraction_performed": False,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(outputs)
        },
    }
    report_path = args.output_dir / "figure_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
