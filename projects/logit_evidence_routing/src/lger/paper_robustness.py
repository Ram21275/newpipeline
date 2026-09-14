"""Submission-oriented robustness analyses over retained development outcomes.

The functions here operate only on saved CSV/JSON artifacts.  They preserve
the image-clustered estimand used by Phase 9, add family-wise bootstrap
intervals, expose per-image influence, and recompute class- and
attribute-balanced estimates without executing a VLM.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


class PaperRobustnessError(RuntimeError):
    """Raised when retained artifacts cannot support an auditable analysis."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperRobustnessError(message)


def finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PaperRobustnessError(f"{field} must be numeric") from error
    require(math.isfinite(result), f"{field} must be finite")
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "cannot take a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _manifest_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        decision_id = str(source.get("decision_id", ""))
        require(decision_id and decision_id not in result, "manifest decision IDs must be unique")
        require(source.get("split") == "val" and source.get("official_split") == "train",
                "robustness analysis accepts development validation rows from official train only")
        result[decision_id] = {
            "decision_id": decision_id,
            "image_id": str(source["image_id"]),
            "class_id": str(source["class_id"]),
            "attribute_id": str(source["attribute_id"]),
            "target": int(source["target"]),
        }
    require(bool(result), "manifest is empty")
    return result


def phase9_family_rows(
    outcomes: Iterable[Mapping[str, Any]],
    plan: Mapping[str, Any],
    manifest: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return decision-level values for four selector effects and two comparisons."""

    records = plan.get("records")
    require(plan.get("status") == "PASS" and isinstance(records, list) and records,
            "a passing intervention plan with records is required")
    require(plan.get("development_only") is True and plan.get("official_test_images_used") == 0,
            "only development intervention plans with zero official-test use are accepted")
    index = {(str(row["decision_id"]), str(row["intervention_id"])): dict(row)
             for row in records}
    require(len(index) == len(records), "plan intervention identities are not unique")
    seen: set[tuple[str, str]] = set()
    by_design: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in outcomes:
        key = (str(source.get("decision_id", "")), str(source.get("intervention_id", "")))
        require(key in index and key not in seen, "outcome identity is absent or duplicated")
        seen.add(key)
        planned = index[key]
        before = finite(source.get("correct_answer_teacher_forced_margin_before"), "margin before")
        after = finite(source.get("correct_answer_teacher_forced_margin_after"), "margin after")
        if str(planned["selection_method"]) != "global_control":
            by_design[(str(planned["decision_id"]), str(planned["stage"]),
                       str(planned["selection_method"]))].append(
                {**planned, "damage": before - after}
            )
    require(seen == set(index), "outcomes do not cover the complete plan")

    decision_effects: dict[tuple[str, str, str], float] = {}
    for key, rows in by_design.items():
        top = [row for row in rows if row["intervention_type"] == "top_evidence"]
        matched = [row for row in rows if row["intervention_type"] == "matched_random"]
        require(len(top) == 1 and matched, f"incomplete top/random design: {key}")
        decision_effects[key] = float(top[0]["damage"]) - statistics.mean(
            float(row["damage"]) for row in matched
        )

    methods = [str(value) for value in plan["selector_methods"]]
    stages = [str(plan["selected_stage"]), str(plan["neighbor_stage"])]
    require(len(methods) == 2, "the simultaneous Phase 9 family requires exactly two selectors")
    manifest_by_id = _manifest_index(manifest)
    decision_ids = sorted({key[0] for key in decision_effects})
    require(set(decision_ids) <= set(manifest_by_id), "Phase 9 decisions are missing from the manifest")
    result: list[dict[str, Any]] = []
    for decision_id in decision_ids:
        identity = manifest_by_id[decision_id]
        for stage in stages:
            values = {}
            for method in methods:
                key = (decision_id, stage, method)
                require(key in decision_effects, f"missing decision effect: {key}")
                values[method] = decision_effects[key]
                result.append({
                    **identity,
                    "estimand": f"selector_effect::{method}::{stage}",
                    "value": values[method],
                })
            result.append({
                **identity,
                "estimand": f"selector_difference::{methods[0]}_minus_{methods[1]}::{stage}",
                "value": values[methods[0]] - values[methods[1]],
            })
    require(len({row["estimand"] for row in result}) == 6,
            "the Phase 9 simultaneous family must contain six estimands")
    return result


def replication_family_rows(
    sources: Mapping[str, Iterable[Mapping[str, Any]]],
    manifest: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return paired visual-control effects for each retained model."""

    manifest_by_id = _manifest_index(manifest)
    controls = ("image", "prompt_only", "image_shuffled", "opposite_label_image")
    result: list[dict[str, Any]] = []
    for label, source_rows in sources.items():
        grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in source_rows:
            decision_id = str(row["decision_id"])
            control = str(row["control"])
            require(control in controls and control not in grouped[decision_id],
                    f"{label} has an unknown or duplicate control")
            grouped[decision_id][control] = row
        require(set(grouped) == set(manifest_by_id),
                f"{label} decisions do not exactly match the manifest")
        for decision_id, rows in grouped.items():
            require(set(rows) == set(controls), f"{label}/{decision_id} lacks a control")
            identity = manifest_by_id[decision_id]
            for reference in controls[1:]:
                for metric, field in (("correct_answer_margin", "correct_answer_margin"),
                                      ("margin_accuracy", "margin_correct")):
                    value = finite(rows["image"][field], field) - finite(rows[reference][field], field)
                    result.append({
                        **identity,
                        "estimand": f"replication::{label}::image_minus_{reference}::{metric}",
                        "value": value,
                    })
    return result


def _image_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(finite(row["value"], "effect value"))
    require(bool(grouped), "estimand has no image values")
    return {key: statistics.mean(values) for key, values in grouped.items()}


def simultaneous_image_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Max-t simultaneous intervals over an image-clustered estimand family."""

    require(samples >= 100 and 0.5 < confidence_level < 1.0, "invalid bootstrap settings")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["estimand"])].append(row)
    image_maps = {estimand: _image_values(values) for estimand, values in grouped.items()}
    image_sets = {tuple(sorted(values)) for values in image_maps.values()}
    require(len(image_sets) == 1, "simultaneous family must share one complete image universe")
    images = list(next(iter(image_sets)))
    estimands = sorted(image_maps)
    estimates = {name: statistics.mean(image_maps[name].values()) for name in estimands}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in estimands}
    for _ in range(samples):
        chosen = [images[rng.randrange(len(images))] for _ in images]
        for name in estimands:
            draws[name].append(statistics.mean(image_maps[name][image_id] for image_id in chosen))
    scales = {name: statistics.stdev(draws[name]) for name in estimands}
    variable = [name for name in estimands if scales[name] > 0]
    maxima = [
        max(abs(draws[name][index] - estimates[name]) / scales[name] for name in variable)
        for index in range(samples)
    ] if variable else [0.0] * samples
    critical = percentile(maxima, confidence_level)
    return [{
        "estimand": name,
        "estimate": estimates[name],
        "simultaneous_ci_low": estimates[name] - critical * scales[name],
        "simultaneous_ci_high": estimates[name] + critical * scales[name],
        "familywise_confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "bootstrap_unit": "image_id",
        "family_size": len(estimands),
        "critical_max_t": critical,
        "n_images": len(images),
    } for name in estimands]


def group_robustness(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = ("class_id", "attribute_id"),
    samples: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Cluster-balanced bootstrap and leave-one-cluster-out diagnostics."""

    by_estimand: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_estimand[str(row["estimand"])].append(row)
    output: list[dict[str, Any]] = []
    for estimand, values in sorted(by_estimand.items()):
        for group_field in group_fields:
            by_group_image: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for row in values:
                by_group_image[str(row[group_field])][str(row["image_id"])].append(
                    finite(row["value"], "effect value")
                )
            group_values = {
                group: statistics.mean(
                    statistics.mean(image_rows) for image_rows in images.values()
                )
                for group, images in by_group_image.items()
            }
            groups = sorted(group_values)
            require(len(groups) >= 2, f"{estimand} needs at least two {group_field} groups")
            estimate = statistics.mean(group_values.values())
            rng = random.Random(seed + sum(ord(char) for char in estimand + group_field))
            draws = [statistics.mean(group_values[groups[rng.randrange(len(groups))]]
                                     for _ in groups) for _ in range(samples)]
            alpha = (1.0 - confidence_level) / 2.0
            loo = {
                group: statistics.mean(value for key, value in group_values.items() if key != group)
                for group in groups
            }
            minimum_group = min(loo, key=loo.get)
            maximum_group = max(loo, key=loo.get)
            output.append({
                "estimand": estimand,
                "grouping": group_field,
                "balanced_estimate": estimate,
                "cluster_bootstrap_ci_low": percentile(draws, alpha),
                "cluster_bootstrap_ci_high": percentile(draws, 1.0 - alpha),
                "confidence_level": confidence_level,
                "bootstrap_samples": samples,
                "n_groups": len(groups),
                "leave_one_group_out_min": loo[minimum_group],
                "leave_one_group_out_min_omitted": minimum_group,
                "leave_one_group_out_max": loo[maximum_group],
                "leave_one_group_out_max_omitted": maximum_group,
            })
    return output


def image_influence(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-image signs and leave-one-image-out ranges for every estimand."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["estimand"])].append(row)
    per_image: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for estimand, values in sorted(grouped.items()):
        image_values = _image_values(values)
        for image_id, value in sorted(image_values.items()):
            per_image.append({"estimand": estimand, "image_id": image_id, "effect": value})
        images = sorted(image_values)
        require(len(images) >= 2, "leave-one-image-out analysis needs at least two images")
        estimate = statistics.mean(image_values.values())
        loo = {
            image_id: statistics.mean(value for key, value in image_values.items() if key != image_id)
            for image_id in images
        }
        minimum_image = min(loo, key=loo.get)
        maximum_image = max(loo, key=loo.get)
        summaries.append({
            "estimand": estimand,
            "estimate": estimate,
            "n_images": len(images),
            "positive_images": sum(value > 0 for value in image_values.values()),
            "zero_images": sum(value == 0 for value in image_values.values()),
            "negative_images": sum(value < 0 for value in image_values.values()),
            "fraction_positive": statistics.mean(value > 0 for value in image_values.values()),
            "leave_one_image_out_min": loo[minimum_image],
            "leave_one_image_out_min_omitted": minimum_image,
            "leave_one_image_out_max": loo[maximum_image],
            "leave_one_image_out_max_omitted": maximum_image,
            "max_absolute_leave_one_out_change": max(abs(value - estimate) for value in loo.values()),
        })
    return per_image, summaries
