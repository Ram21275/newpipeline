#!/usr/bin/env python3
"""Analyze full LLaVA/Qwen VQA replications with image-clustered intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase7 import clustered_paired_bootstrap  # noqa: E402


CONTROLS = ("image", "prompt_only", "image_shuffled", "opposite_label_image")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"replication table is empty: {path}")
    return rows


def finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{field} must be finite")
    return result


def stable_seed(seed: int, *parts: str) -> int:
    payload = json.dumps([seed, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def normalize(label: str, path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    decisions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_csv(path):
        decision_id = str(row["decision_id"])
        control = str(row["control"])
        if control not in CONTROLS or control in decisions[decision_id]:
            raise RuntimeError(f"{label} has an unknown or duplicate control")
        target = int(row["target"])
        tie = int(row["margin_tie"])
        correct = int(row["margin_correct"])
        if target not in (0, 1) or tie not in (0, 1) or correct not in (0, 1):
            raise RuntimeError(f"{label} has malformed binary outcomes")
        if tie and correct:
            raise RuntimeError("an exact margin tie must be scored incorrect")
        decisions[decision_id][control] = {
            "decision_id": decision_id,
            "image_id": str(row["image_id"]),
            "attribute_id": str(row["attribute_id"]),
            "target": target,
            "tie": tie,
            "correct": correct,
            "correct_margin": finite(row["correct_answer_margin"], "correct_answer_margin"),
            "generation_correct": (
                None if row.get("generation_correct", "") == ""
                else int(row["generation_correct"])
            ),
        }
    for decision_id, controls in decisions.items():
        if set(controls) != set(CONTROLS):
            raise RuntimeError(f"{label}/{decision_id} lacks one of the four controls")
        identity = {(row["image_id"], row["attribute_id"], row["target"])
                    for row in controls.values()}
        if len(identity) != 1:
            raise RuntimeError(f"{label}/{decision_id} identity changes across controls")
    return decisions


def interval(
    values: list[tuple[str, float]], *, samples: int, seed: int, confidence: float
) -> dict[str, Any]:
    return clustered_paired_bootstrap(
        [item[0] for item in values], [item[1] for item in values],
        samples=samples, seed=seed, confidence_level=confidence,
    )


def analyze_model(
    label: str,
    decisions: dict[str, dict[str, dict[str, Any]]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    summaries = []
    for control in CONTROLS:
        rows = [values[control] for values in decisions.values()]
        summaries.append({
            "model_label": label,
            "control": control,
            "decisions": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "margin_accuracy": statistics.mean(row["correct"] for row in rows),
            "mean_correct_answer_margin": statistics.mean(row["correct_margin"] for row in rows),
            "margin_ties": sum(row["tie"] for row in rows),
            "margin_tie_rate": statistics.mean(row["tie"] for row in rows),
            "generation_accuracy": (
                statistics.mean(row["generation_correct"] for row in rows)
                if all(row["generation_correct"] is not None for row in rows) else None
            ),
        })
    contrasts = []
    for reference in CONTROLS[1:]:
        for metric, field in (("correct_answer_margin", "correct_margin"),
                              ("margin_accuracy", "correct")):
            values = [
                (controls["image"]["image_id"],
                 float(controls["image"][field]) - float(controls[reference][field]))
                for controls in decisions.values()
            ]
            result = interval(
                values, samples=samples,
                seed=stable_seed(seed, label, reference, metric), confidence=confidence,
            )
            contrasts.append({
                "model_label": label,
                "contrast": f"image_minus_{reference}",
                "metric": metric,
                **result,
                "interpretation": (
                    "null_compatible" if result["ci_low"] <= 0 <= result["ci_high"]
                    else "correct_image_higher" if result["ci_low"] > 0
                    else "correct_image_lower"
                ),
            })
    return {"summaries": summaries, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.bootstrap_samples < 100 or not 0.5 < args.confidence_level < 1:
        raise RuntimeError("invalid bootstrap settings")
    labels = [item[0] for item in args.input]
    if len(labels) != len(set(labels)):
        raise RuntimeError("input labels must be unique")
    models = {label: normalize(label, Path(path)) for label, path in args.input}
    summaries, contrasts = [], []
    for label, decisions in models.items():
        result = analyze_model(
            label, decisions, samples=args.bootstrap_samples,
            seed=args.seed, confidence=args.confidence_level,
        )
        summaries.extend(result["summaries"])
        contrasts.extend(result["contrasts"])

    cross_model = []
    if len(models) == 2:
        left_label, right_label = labels
        left, right = models[left_label], models[right_label]
        if set(left) != set(right):
            raise RuntimeError("cross-model comparison requires identical decisions")
        for reference in CONTROLS[1:]:
            values = []
            for decision_id in sorted(left):
                left_advantage = (left[decision_id]["image"]["correct_margin"]
                                  - left[decision_id][reference]["correct_margin"])
                right_advantage = (right[decision_id]["image"]["correct_margin"]
                                   - right[decision_id][reference]["correct_margin"])
                values.append((left[decision_id]["image"]["image_id"],
                               left_advantage - right_advantage))
            result = interval(
                values, samples=args.bootstrap_samples,
                seed=stable_seed(args.seed, left_label, right_label, reference),
                confidence=args.confidence_level,
            )
            cross_model.append({
                "left_model": left_label,
                "right_model": right_label,
                "contrast": f"difference_in_image_minus_{reference}_correct_margin",
                **result,
                "interpretation": "null_compatible" if result["ci_low"] <= 0 <= result["ci_high"]
                else "left_larger" if result["ci_low"] > 0 else "right_larger",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "replication_vqa_summary.csv"
    contrast_path = args.output_dir / "replication_vqa_contrasts.csv"
    for path, rows in ((summary_path, summaries), (contrast_path, contrasts)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "cross_model_vqa_replication_analysis",
        "model_labels": labels,
        "bootstrap_unit": "image_id",
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_level": args.confidence_level,
        "tie_policy": "retain exact zero margins as abstentions and score incorrect",
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "cross_model_contrasts": cross_model,
        "official_test_images_used": 0,
    }
    report_path = args.output_dir / "replication_vqa_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(report_path.resolve()),
                      "models": labels}, sort_keys=True))


if __name__ == "__main__":
    main()
