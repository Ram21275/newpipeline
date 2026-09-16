"""Final per-example validation and clustered analysis exports."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .core import atomic_write_json, atomic_write_jsonl, read_jsonl
from .scoring import fixed_binary_parser, polarity_effects
from .statistics import paired_image_cluster_bootstrap


REQUIRED_FIELDS = {
    "question_id",
    "image_id",
    "attribute_id",
    "target",
    "model",
    "condition",
    "positive_log_probability",
    "negative_log_probability",
    "generated_text",
    "visual_token_count",
    "runtime_seconds",
}


def validate_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, int]] = set()
    for row in rows:
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"per-example record lacks fields: {sorted(missing)}")
        identity = (
            str(row["model"]),
            str(row["question_id"]),
            str(row["condition"]),
            int(row.get("repeat", 0)),
        )
        if identity in identities:
            raise ValueError(f"duplicate per-example condition: {identity}")
        identities.add(identity)
        positive = float(row["positive_log_probability"])
        negative = float(row["negative_log_probability"])
        if not math.isfinite(positive) or not math.isfinite(negative):
            raise ValueError(f"non-finite candidate score: {identity}")
        normalized = dict(row)
        normalized["raw_margin"] = positive - negative
        normalized["parsed_generation"] = fixed_binary_parser(
            str(row["generated_text"]),
            positive=str(row.get("positive_answer", "yes")),
            negative=str(row.get("negative_answer", "no")),
        )
        output.append(normalized)
    return output


def derive_paired_effects(
    rows: Iterable[dict[str, Any]], *, baseline_condition: str = "full"
) -> list[dict[str, Any]]:
    index = {
        (str(row["model"]), str(row["question_id"]), str(row["condition"]), int(row.get("repeat", 0))): row
        for row in rows
    }
    effects: list[dict[str, Any]] = []
    for key, row in sorted(index.items()):
        model, question_id, condition, repeat = key
        if condition == baseline_condition:
            continue
        baseline = index.get((model, question_id, baseline_condition, repeat))
        if baseline is None:
            raise ValueError(f"missing paired baseline for {key}")
        values = polarity_effects(
            baseline_margin=float(baseline["raw_margin"]),
            intervention_margin=float(row["raw_margin"]),
            target=int(row["target"]),
        )
        effects.append(
            {
                "model": model,
                "question_id": question_id,
                "image_id": row["image_id"],
                "attribute_id": row["attribute_id"],
                "attribute_group": row.get("attribute_group"),
                "target": int(row["target"]),
                "condition": condition,
                "repeat": repeat,
                **values,
            }
        )
    return effects


def analyze_per_example(
    source: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 20260916,
) -> dict[str, Any]:
    source_rows = read_jsonl(source)
    rows = validate_records(
        [
            dict(row["payload"])
            if "payload" in row and row.get("status") == "complete"
            else row
            for row in source_rows
            if "payload" not in row or row.get("status") == "complete"
        ]
    )
    effects = derive_paired_effects(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_dir / "validated_per_example.jsonl", rows)
    atomic_write_jsonl(output_dir / "paired_effects.jsonl", effects)
    families: list[dict[str, Any]] = []
    for row in effects:
        for scale in ("delta_raw", "delta_correct"):
            families.append(
                {
                    **row,
                    "family": f"{row['model']}::{row['condition']}::{scale}",
                    "effect": row[scale],
                }
            )
    bootstrap = paired_image_cluster_bootstrap(
        families,
        effect_field="effect",
        family_field="family",
        resamples=bootstrap_resamples,
        seed=seed,
        weighting="image",
    )
    generated: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = f"{row['model']}::{row['condition']}"
        parsed = row["parsed_generation"]
        generated[key]["total"] += 1
        generated[key][parsed] += 1
        prediction = 1 if parsed == "positive" else 0 if parsed == "negative" else None
        if prediction is not None and prediction == int(row["target"]):
            generated[key]["correct"] += 1
    report = {
        "status": "COMPLETE_FROM_PROVIDED_PER_EXAMPLE_OUTPUTS",
        "record_count": len(rows),
        "effect_count": len(effects),
        "models": sorted({str(row["model"]) for row in rows}),
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "image_count": len({str(row["image_id"]) for row in rows}),
        "generated_counts": {key: dict(value) for key, value in sorted(generated.items())},
        "cluster_bootstrap": bootstrap,
    }
    atomic_write_json(output_dir / "final_analysis.json", report)
    return report
