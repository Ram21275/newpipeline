"""Paired no-GPU analysis for multimodal DeCo and DoLa extractions."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics
from typing import Any, Mapping, Sequence


METHOD_PREFIXES = (
    "ordinary",
    "binary_dola",
    "full_vocab_dola",
    "binary_deco",
    "full_vocab_deco",
)


class LayerDecodingAnalysisError(RuntimeError):
    """Raised when a DeCo/DoLa result table violates the locked schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerDecodingAnalysisError(message)


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise LayerDecodingAnalysisError(f"{name} must be numeric") from error
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value: str,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    _require(samples >= 100, "bootstrap requires at least 100 samples")
    _require(0.5 < confidence_level < 1.0, "invalid confidence level")
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(_number(row[value], value))
    _require(bool(by_image), "inference cell has no images")
    images = sorted(by_image)
    estimate = statistics.mean(value for values in by_image.values() for value in values)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [images[rng.randrange(len(images))] for _ in images]
        draws.append(
            statistics.mean(value for image in chosen for value in by_image[image])
        )
    draws.sort()
    alpha = (1.0 - confidence_level) / 2.0

    def quantile(fraction: float) -> float:
        position = fraction * (len(draws) - 1)
        low = int(position)
        high = min(low + 1, len(draws) - 1)
        weight = position - low
        return draws[low] * (1.0 - weight) + draws[high] * weight

    return {
        "estimate": estimate,
        "ci_low": quantile(alpha),
        "ci_high": quantile(1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "bootstrap_unit": "image_id",
        "estimand_weighting": "decision",
        "n_images": len(images),
        "n_decisions": len(rows),
    }


def analyze_layer_decoding(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10000,
    seed: int = 20260915,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarize methods and paired image-dependent gains over final scoring."""

    _require(bool(rows), "layer-decoding table is empty")
    required = {
        "decision_id",
        "image_id",
        "class_id",
        "attribute_id",
        "target",
        "control",
        "binary_premature_layer",
        "full_vocab_premature_layer",
        "binary_deco_anchor_layer",
        "full_vocab_deco_anchor_layer",
    }
    for prefix in METHOD_PREFIXES:
        required.update(
            {
                f"{prefix}_semantic_margin",
                f"{prefix}_correct_margin",
                f"{prefix}_correct",
            }
        )
    _require(all(required <= set(row) for row in rows), "layer-decoding table lacks fields")
    keys = [(str(row["decision_id"]), str(row["control"])) for row in rows]
    _require(len(keys) == len(set(keys)), "layer-decoding rows are duplicated")
    decisions = sorted({str(row["decision_id"]) for row in rows})
    controls = sorted({str(row["control"]) for row in rows})
    _require(
        set(keys) == {(decision, control) for decision in decisions for control in controls},
        "layer-decoding result coverage is incomplete",
    )
    _require(
        {"image", "prompt_only", "image_shuffled"} <= set(controls),
        "visual controls are incomplete",
    )

    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["target"] = int(source["target"])
        _require(row["target"] in (0, 1), "target must be binary")
        for prefix in METHOD_PREFIXES:
            row[f"{prefix}_semantic_margin"] = _number(
                source[f"{prefix}_semantic_margin"], f"{prefix} semantic margin"
            )
            row[f"{prefix}_correct_margin"] = _number(
                source[f"{prefix}_correct_margin"], f"{prefix} correct margin"
            )
            row[f"{prefix}_correct"] = float(int(source[f"{prefix}_correct"]))
        normalized.append(row)

    summary: list[dict[str, Any]] = []
    for control in controls:
        subset = [row for row in normalized if row["control"] == control]
        for prefix in METHOD_PREFIXES:
            summary.append(
                {
                    "control": control,
                    "method": prefix,
                    "mean_semantic_margin": statistics.mean(
                        row[f"{prefix}_semantic_margin"] for row in subset
                    ),
                    "mean_correct_margin": statistics.mean(
                        row[f"{prefix}_correct_margin"] for row in subset
                    ),
                    "accuracy": statistics.mean(
                        row[f"{prefix}_correct"] for row in subset
                    ),
                    "n_images": len({str(row["image_id"]) for row in subset}),
                    "n_decisions": len(subset),
                }
            )

    indexed = {
        (str(row["decision_id"]), str(row["control"])): row for row in normalized
    }
    paired_rows: list[dict[str, Any]] = []
    for decision in decisions:
        for control in controls:
            row = indexed[(decision, control)]
            for method in METHOD_PREFIXES[1:]:
                for comparator in ("ordinary", "full_vocab_dola"):
                    if method == comparator:
                        continue
                    paired_rows.append(
                        {
                            "decision_id": decision,
                            "image_id": row["image_id"],
                            "attribute_id": row["attribute_id"],
                            "class_id": row["class_id"],
                            "target": row["target"],
                            "control": control,
                            "method": method,
                            "comparator": comparator,
                            "correct_margin_difference": (
                                row[f"{method}_correct_margin"]
                                - row[f"{comparator}_correct_margin"]
                            ),
                            "accuracy_difference": (
                                row[f"{method}_correct"] - row[f"{comparator}_correct"]
                            ),
                        }
                    )

    paired_inference: list[dict[str, Any]] = []
    inference_index = 0
    for control in controls:
        for method in METHOD_PREFIXES[1:]:
            for comparator in ("ordinary", "full_vocab_dola"):
                if method == comparator:
                    continue
                subset = [
                    row
                    for row in paired_rows
                    if row["control"] == control
                    and row["method"] == method
                    and row["comparator"] == comparator
                ]
                for metric in ("correct_margin_difference", "accuracy_difference"):
                    paired_inference.append(
                        {
                            "contrast": f"{method}_minus_{comparator}",
                            "control": control,
                            "metric": metric,
                            **_cluster_interval(
                                subset,
                                value=metric,
                                samples=samples,
                                seed=seed + inference_index,
                                confidence_level=confidence_level,
                            ),
                        }
                    )
                    inference_index += 1

    visual_rows: list[dict[str, Any]] = []
    for decision in decisions:
        image_row = indexed[(decision, "image")]
        for method in METHOD_PREFIXES[1:]:
            image_gain = (
                image_row[f"{method}_correct_margin"]
                - image_row["ordinary_correct_margin"]
            )
            for control in ("prompt_only", "image_shuffled"):
                control_row = indexed[(decision, control)]
                control_gain = (
                    control_row[f"{method}_correct_margin"]
                    - control_row["ordinary_correct_margin"]
                )
                visual_rows.append(
                    {
                        "decision_id": decision,
                        "image_id": image_row["image_id"],
                        "attribute_id": image_row["attribute_id"],
                        "class_id": image_row["class_id"],
                        "target": image_row["target"],
                        "method": method,
                        "contrast": f"image_gain_minus_{control}_gain",
                        "difference_in_differences": image_gain - control_gain,
                    }
                )
    visual_inference = []
    for method in METHOD_PREFIXES[1:]:
        for contrast in (
            "image_gain_minus_prompt_only_gain",
            "image_gain_minus_image_shuffled_gain",
        ):
            subset = [
                row
                for row in visual_rows
                if row["method"] == method and row["contrast"] == contrast
            ]
            visual_inference.append(
                {
                    "method": method,
                    "contrast": contrast,
                    **_cluster_interval(
                        subset,
                        value="difference_in_differences",
                        samples=samples,
                        seed=seed + 1000 + len(visual_inference),
                        confidence_level=confidence_level,
                    ),
                }
            )

    layer_columns = (
        ("binary_deco", "binary_deco_anchor_layer"),
        ("full_vocab_deco", "full_vocab_deco_anchor_layer"),
        ("binary_dola", "binary_premature_layer"),
        ("full_vocab_dola", "full_vocab_premature_layer"),
    )
    layer_selection = []
    for control in controls:
        subset = [row for row in normalized if row["control"] == control]
        for method, column in layer_columns:
            counts = Counter(int(row[column]) for row in subset)
            for layer, count in sorted(counts.items()):
                layer_selection.append(
                    {
                        "control": control,
                        "method": method,
                        "layer": layer,
                        "count": count,
                        "proportion": count / len(subset),
                        "n_decisions": len(subset),
                    }
                )

    image_full_deco = [
        row
        for row in paired_inference
        if row["control"] == "image"
        and row["contrast"] == "full_vocab_deco_minus_ordinary"
    ]
    return {
        "schema_version": 1,
        "status": "PASS",
        "decision_count": len(decisions),
        "image_count": len({str(row["image_id"]) for row in normalized}),
        "controls": controls,
        "summary": summary,
        "paired_inference": paired_inference,
        "visual_specificity_inference": visual_inference,
        "layer_selection": layer_selection,
        "primary_promotion_snapshot": image_full_deco,
        "bootstrap_samples": samples,
        "confidence_level": confidence_level,
        "official_test_images_used": 0,
        "claim_boundary": (
            "A paired DeCo gain shows that preceding fused language-layer logits can "
            "improve frozen CUB readout. It does not uniquely establish language-prior "
            "suppression, natural model routing, or generalization beyond this cohort."
        ),
    }
