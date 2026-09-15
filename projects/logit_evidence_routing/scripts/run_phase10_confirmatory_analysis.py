#!/usr/bin/env python3
"""Run the no-GPU Phase 10 raw/correct-margin confirmatory analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase10_confirmatory import run_confirmatory_analysis  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"input CSV is empty: {path}")
    return rows


def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        found = False
        for row in reader:
            found = True
            yield row
    if not found:
        raise RuntimeError(f"input CSV is empty: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _error_bar(ax: Any, y: float, row: dict[str, Any], color: str, marker: str) -> None:
    estimate = float(row["estimate"])
    low = float(row["simultaneous_ci_low"])
    high = float(row["simultaneous_ci_high"])
    ax.plot([low, high], [y, y], color=color, linewidth=1.5, alpha=0.85)
    ax.scatter([estimate], [y], color=color, marker=marker, s=34, zorder=3)


def _plot_results_pillow(
    output_dir: Path,
    inference: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    accuracy: list[dict[str, Any]],
) -> list[Path]:
    """Dependency-light raster/PDF fallback for clean Kaggle/local figures."""

    from PIL import Image, ImageDraw, ImageFont

    regular_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

    def font(size: int, *, bold: bool = False) -> Any:
        path = bold_path if bold else regular_path
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:  # pragma: no cover - platform font fallback
            return ImageFont.load_default()

    title_font, label_font, tick_font = font(30, bold=True), font(22), font(18)
    navy, orange, ink, grid = "#2563EB", "#D97706", "#111827", "#D1D5DB"

    def save_pair(image: Any, stem: str) -> list[Path]:
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        image.save(png, format="PNG", dpi=(240, 240))
        image.save(pdf, format="PDF", resolution=240.0)
        return [pdf, png]

    def domain(values: list[float], *, include_zero: bool = True) -> tuple[float, float]:
        if include_zero:
            values = [*values, 0.0]
        low, high = min(values), max(values)
        padding = max((high - low) * 0.12, 0.01)
        return low - padding, high + padding

    paths: list[Path] = []
    focus = [row for row in inference if row["contrast_type"] == "selector_difference_by_target"]
    image = Image.new("RGB", (2200, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 35), "Phase 10 selector differential by target (simultaneous 95% intervals)",
              fill=ink, font=title_font)
    for panel, scale in enumerate(("raw_yes_minus_no", "correct_answer_margin")):
        rows = [row for row in focus if row["metric_scale"] == scale]
        x0, x1 = 410 + panel * 1060, 1020 + panel * 1060
        y0, y1 = 170, 730
        values = [float(row[key]) for row in rows
                  for key in ("simultaneous_ci_low", "simultaneous_ci_high")]
        low, high = domain(values)
        to_x = lambda value: x0 + (float(value) - low) / (high - low) * (x1 - x0)
        zero = to_x(0.0)
        draw.line((zero, y0, zero, y1), fill="#6B7280", width=3)
        for tick_index in range(5):
            tick = low + tick_index * (high - low) / 4
            x = to_x(tick)
            draw.line((x, y1, x, y1 + 12), fill=ink, width=2)
            draw.text((x - 30, y1 + 18), f"{tick:.2f}", fill=ink, font=tick_font)
        draw.text((x0, 105), "Raw yes-minus-no" if panel == 0 else "Correct-answer margin",
                  fill=ink, font=label_font)
        for row_index, (stage, target) in enumerate(
            (stage_target for stage_target in ((stage, target)
             for stage in ("vision.late", "projector.output") for target in (0, 1)))
        ):
            row = next(item for item in rows
                       if item["stage"] == stage and int(item["target"]) == target)
            y = y0 + 75 + row_index * 120
            draw.text((80 + panel * 1060, y - 13), f"{stage}; target {target}",
                      fill=ink, font=tick_font)
            lo, estimate, hi = (to_x(row[key]) for key in
                                ("simultaneous_ci_low", "estimate", "simultaneous_ci_high"))
            draw.line((lo, y, hi, y), fill=navy if target else orange, width=6)
            radius = 10
            draw.ellipse((estimate - radius, y - radius, estimate + radius, y + radius),
                         fill=navy if target else orange)
        draw.line((x0, y1, x1, y1), fill=ink, width=2)
        draw.text((x0, 780), "Vision-CLS minus Logit-Concept selector effect",
                  fill=ink, font=tick_font)
    paths.extend(save_pair(image, "phase10_raw_correct_selector_effects"))

    blocked = [row for row in robustness
               if row["metric_scale"] == "raw_yes_minus_no"
               and row["contrast_type"] == "selector_difference_target_difference"]
    image = Image.new("RGB", (1800, 850), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Blocked robustness and leave-one-group-out range", fill=ink,
              font=title_font)
    values = [float(row[key]) for row in blocked for key in (
        "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high",
        "leave_one_group_out_min", "leave_one_group_out_max")]
    low, high = domain(values)
    x0, x1, y0 = 610, 1700, 170
    to_x = lambda value: x0 + (float(value) - low) / (high - low) * (x1 - x0)
    draw.line((to_x(0), y0 - 40, to_x(0), 700), fill="#6B7280", width=3)
    for tick_index in range(5):
        tick = low + tick_index * (high - low) / 4
        x = to_x(tick)
        draw.line((x, 700, x, 712), fill=ink, width=2)
        draw.text((x - 30, 720), f"{tick:.2f}", fill=ink, font=tick_font)
    for index, row in enumerate(blocked):
        y = y0 + index * 125
        grouping = "attribute" if row["grouping"] == "attribute_id" else "species"
        draw.text((70, y - 13), f"{row['stage']}; {grouping} blocked", fill=ink,
                  font=tick_font)
        draw.line((to_x(row["leave_one_group_out_min"]), y,
                   to_x(row["leave_one_group_out_max"]), y), fill="#9CA3AF", width=18)
        draw.line((to_x(row["cluster_bootstrap_ci_low"]), y,
                   to_x(row["cluster_bootstrap_ci_high"]), y), fill=navy, width=6)
        point = to_x(row["equal_group_estimate"])
        draw.ellipse((point - 9, y - 9, point + 9, y + 9), fill=ink)
    draw.text((x0, 755), "Target 1 minus target 0 of raw selector differential",
              fill=ink, font=tick_font)
    paths.extend(save_pair(image, "phase10_blocked_robustness"))

    selected = [row for row in accuracy
                if row["source"] == "retained_behavior" and row["control"] == "image"
                and row["target"] == "all"
                and row["metric"] in ("teacher_forced_accuracy", "generation_accuracy_all")]
    models = sorted({str(row["model"]) for row in selected})
    metrics = ("teacher_forced_accuracy", "generation_accuracy_all")
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Balanced CUB development behavior", fill=ink, font=title_font)
    x0, x1, y0, y1 = 180, 1400, 140, 750
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y1 - fraction * (y1 - y0)
        draw.line((x0, y, x1, y), fill=grid, width=2)
        draw.text((95, y - 10), f"{fraction:.2f}", fill=ink, font=tick_font)
    bar_width = 150
    group_width = (x1 - x0) / max(len(models), 1)
    for model_index, model in enumerate(models):
        center = x0 + group_width * (model_index + 0.5)
        for metric_index, metric in enumerate(metrics):
            row = next(item for item in selected
                       if item["model"] == model and item["metric"] == metric)
            value = float(row["estimate"])
            bx0 = center + (metric_index - 1) * bar_width
            bx1 = bx0 + bar_width
            by = y1 - value * (y1 - y0)
            draw.rectangle((bx0, by, bx1, y1), fill=navy if metric_index == 0 else orange)
            cx = (bx0 + bx1) / 2
            ci_low = y1 - float(row["ci_low"]) * (y1 - y0)
            ci_high = y1 - float(row["ci_high"]) * (y1 - y0)
            draw.line((cx, ci_low, cx, ci_high), fill=ink, width=4)
        draw.text((center - 45, 780), model, fill=ink, font=label_font)
    draw.line((x0, y1 - 0.5 * (y1 - y0), x1, y1 - 0.5 * (y1 - y0)),
              fill="#6B7280", width=3)
    draw.rectangle((930, 80, 960, 110), fill=navy)
    draw.text((975, 82), "Teacher-forced", fill=ink, font=tick_font)
    draw.rectangle((1165, 80, 1195, 110), fill=orange)
    draw.text((1210, 82), "Generated", fill=ink, font=tick_font)
    paths.extend(save_pair(image, "phase10_absolute_accuracy"))
    return paths


def plot_results(
    output_dir: Path,
    inference: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    accuracy: list[dict[str, Any]],
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - exercised on minimal local runtimes
        return _plot_results_pillow(output_dir, inference, robustness, accuracy)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    paths: list[Path] = []

    focus = [row for row in inference
             if row["contrast_type"] == "selector_difference_by_target"]
    stages = ["vision.late", "projector.output"]
    scales = ["raw_yes_minus_no", "correct_answer_margin"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for axis, scale in zip(axes, scales):
        rows = [row for row in focus if row["metric_scale"] == scale]
        labels = []
        for index, stage in enumerate(stages):
            for target in (0, 1):
                row = next(item for item in rows
                           if item["stage"] == stage and int(item["target"]) == target)
                y = 3 - (index * 2 + target)
                _error_bar(axis, y, row, "#2563EB" if target else "#D97706", "o")
                labels.append((y, f"{stage}; target {target}"))
        axis.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--")
        axis.set_title("Raw yes-minus-no" if scale == "raw_yes_minus_no"
                       else "Correct-answer margin")
        axis.set_xlabel("Vision-CLS minus Logit-Concept selector effect")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        axis.set_yticks([item[0] for item in labels], [item[1] for item in labels])
    fig.suptitle("Phase 10 selector differential by target (simultaneous 95% intervals)")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"phase10_raw_correct_selector_effects.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    blocked = [row for row in robustness
               if row["metric_scale"] == "raw_yes_minus_no"
               and row["contrast_type"] == "selector_difference_target_difference"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    positions = list(range(len(blocked)))
    for position, row in zip(positions, blocked):
        estimate = float(row["equal_group_estimate"])
        low = float(row["cluster_bootstrap_ci_low"])
        high = float(row["cluster_bootstrap_ci_high"])
        ax.plot([low, high], [position, position], color="#2563EB", linewidth=1.5)
        ax.plot([float(row["leave_one_group_out_min"]),
                 float(row["leave_one_group_out_max"])], [position, position],
                color="#9CA3AF", linewidth=5, alpha=0.45)
        ax.scatter([estimate], [position], color="#111827", s=30, zorder=3)
    ax.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--")
    ax.set_yticks(positions, [
        f"{row['stage']}; {'attribute' if row['grouping']=='attribute_id' else 'species'} blocked"
        for row in blocked
    ])
    ax.set_xlabel("Target 1 minus target 0 of raw selector differential")
    ax.set_title("Blocked robustness and leave-one-group-out range")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"phase10_blocked_robustness.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    accuracy_rows = [row for row in accuracy
                     if row["source"] == "retained_behavior"
                     and row["control"] == "image" and row["target"] == "all"
                     and row["metric"] in ("teacher_forced_accuracy", "generation_accuracy_all")]
    models = sorted({str(row["model"]) for row in accuracy_rows})
    metrics = ["teacher_forced_accuracy", "generation_accuracy_all"]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    width = 0.34
    colors = ["#2563EB", "#D97706"]
    for metric_index, metric in enumerate(metrics):
        values = [next(row for row in accuracy_rows
                       if row["model"] == model and row["metric"] == metric)
                  for model in models]
        xs = [index + (metric_index - 0.5) * width for index in range(len(models))]
        estimates = [float(row["estimate"]) for row in values]
        errors = [[estimate - float(row["ci_low"]) for estimate, row in zip(estimates, values)],
                  [float(row["ci_high"]) - estimate for estimate, row in zip(estimates, values)]]
        ax.bar(xs, estimates, width=width, color=colors[metric_index],
               label="Teacher-forced" if metric_index == 0 else "Generated (all decisions)")
        ax.errorbar(xs, estimates, yerr=errors, fmt="none", ecolor="#111827",
                    elinewidth=0.8, capsize=2)
    ax.axhline(0.5, color="#4B5563", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(models)), models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Absolute accuracy")
    ax.set_title("Balanced CUB development behavior")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"phase10_absolute_accuracy.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decision-metadata", type=Path, required=True,
                        help="Manifest or repeated selector-token metadata containing class_id")
    parser.add_argument("--behavior", nargs=2, action="append", metavar=("MODEL", "CSV"),
                        default=[])
    parser.add_argument("--config", type=Path,
                        default=PROJECT / "configs/phase10_confirmatory_analysis.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--confidence-level", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not (config.get("schema_version") == 1
            and config.get("purpose") == "phase10_confirmatory_raw_and_correct_margin_analysis"
            and config.get("development_only") is True
            and config.get("official_test_images_used") == 0):
        raise RuntimeError("unsupported confirmatory analysis config")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if int(plan.get("balanced_k", -1)) != int(config["balanced_k"]):
        raise RuntimeError("analysis config and Phase 10 balanced K differ")
    behavior_paths = {label: Path(path) for label, path in args.behavior}
    if len(behavior_paths) != len(args.behavior):
        raise RuntimeError("behavior model labels must be unique")
    samples = args.bootstrap_samples or int(config["bootstrap_replicates"])
    confidence = args.confidence_level or float(config["confidence_level"])
    seed = args.seed if args.seed is not None else int(config["seed"])
    result = run_confirmatory_analysis(
        read_csv(args.outcomes), plan=plan,
        decision_metadata=iter_csv(args.decision_metadata),
        behavior_by_model={label: read_csv(path) for label, path in behavior_paths.items()},
        bootstrap_samples=samples, seed=seed, confidence_level=confidence,
    )
    for field in ("outcome_count", "decision_count", "image_count"):
        if int(result[field]) != int(config[field]):
            raise RuntimeError(f"observed {field} differs from locked config")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "phase10_selector_effects.csv": result.pop("selector_effect_rows"),
        "phase10_selector_differences.csv": result.pop("selector_difference_rows"),
        "phase10_baseline_margins.csv": result.pop("baseline_rows"),
        "phase10_main_inference.csv": result["main_inference"],
        "phase10_blocked_robustness.csv": result["blocked_robustness"],
        "phase10_leave_one_group_out.csv": result.pop("leave_one_group_out"),
        "phase10_absolute_accuracy.csv": result["absolute_accuracy"],
    }
    for name, rows in tables.items():
        write_csv(args.output_dir / name, rows)
    figure_paths = [] if args.no_figures else plot_results(
        args.output_dir, result["main_inference"], result["blocked_robustness"],
        result["absolute_accuracy"],
    )
    result["source_hashes"] = {
        "outcomes": sha256(args.outcomes),
        "plan": sha256(args.plan),
        "decision_metadata": sha256(args.decision_metadata),
        "analysis_config": sha256(args.config),
        **{f"behavior_{label}": sha256(path) for label, path in behavior_paths.items()},
    }
    result["artifacts"] = {
        **{name: sha256(args.output_dir / name) for name in tables},
        **{path.name: sha256(path) for path in figure_paths},
    }
    report_path = args.output_dir / "phase10_confirmatory_analysis.json"
    atomic_json_write(result, report_path)
    print(json.dumps({
        "status": result["status"],
        "output_dir": str(args.output_dir.resolve()),
        "decisions": result["decision_count"],
        "images": result["image_count"],
        "attributes": result["attribute_count"],
        "species": result["species_count"],
        "main_estimands": len(result["main_inference"]),
        "figures": len(figure_paths),
        "official_test_images_used": result["official_test_images_used"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
