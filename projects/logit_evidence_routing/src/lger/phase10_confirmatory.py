"""Confirmatory retained-artifact analysis for the balanced Phase 10 cohort.

The Phase 10 intervention CSV stores correct-answer margins.  This module keeps
that scale and reconstructs the underlying yes-minus-no margin so the polarity
analysis is not driven by the ``(2y-1)`` sign convention.  All uncertainty is
computed from retained tables; no model or image data are required.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PURPOSE = "phase10_confirmatory_raw_and_correct_margin_analysis"


class Phase10ConfirmatoryError(RuntimeError):
    """Raised when a retained artifact is incomplete or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase10ConfirmatoryError(message)


def finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Phase10ConfirmatoryError(f"{field} must be numeric") from error
    require(math.isfinite(result), f"{field} must be finite")
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stable_seed(seed: int, *parts: Any) -> int:
    # Avoid process-randomized hash(), while keeping the implementation stdlib-only.
    text = "::".join([str(int(seed)), *(str(part) for part in parts)])
    value = 1469598103934665603
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def decision_metadata_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collapse repeated token/manifest rows to one audited decision identity."""

    output: dict[str, dict[str, Any]] = {}
    image_classes: dict[str, str] = {}
    for source in rows:
        decision_id = str(source.get("decision_id", ""))
        if not decision_id:
            continue
        image_id = str(source.get("image_id", ""))
        class_id = str(source.get("class_id", ""))
        require(image_id and class_id, "decision metadata requires image_id and class_id")
        identity = {
            "decision_id": decision_id,
            "image_id": image_id,
            "class_id": class_id,
            "attribute_id": str(source.get("attribute_id", "")),
            "target": int(source.get("target")),
        }
        if source.get("attribute_name") not in (None, ""):
            identity["attribute_name"] = str(source["attribute_name"])
        previous = output.setdefault(decision_id, identity)
        require(previous == identity, f"decision metadata differs for {decision_id}")
        previous_class = image_classes.setdefault(image_id, class_id)
        require(previous_class == class_id, f"class_id differs within image {image_id}")
    require(bool(output), "decision metadata is empty")
    return output


def build_effect_tables(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selector effects, paired selector differences, and unique baselines.

    Positive ``effect`` means the selected replacement is more damaging than its
    matched-random controls.  On the raw scale, damage is measured as the drop in
    ``log P(yes) - log P(no)``.  On the correct-answer scale it is the corresponding
    signed-margin drop used by the original Phase 10 analysis.
    """

    require(plan.get("status") == "PASS", "a passing Phase 10 plan is required")
    require(plan.get("purpose") == "phase10_development_balanced_target_and_k_dose_plan",
            "unexpected Phase 10 plan purpose")
    require(plan.get("development_only") is True and plan.get("official_test_images_used") == 0,
            "confirmatory retained analysis must remain development-only")
    records = plan.get("records")
    require(isinstance(records, list) and bool(records), "Phase 10 plan records are missing")
    plan_index = {
        (str(row["decision_id"]), str(row["intervention_id"])): dict(row)
        for row in records
    }
    require(len(plan_index) == len(records), "Phase 10 plan identities are duplicated")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    baseline_values: dict[str, float] = {}
    for source in outcomes:
        key = (str(source.get("decision_id", "")), str(source.get("intervention_id", "")))
        require(key in plan_index, f"outcome is absent from the Phase 10 plan: {key}")
        require(key not in seen, f"outcome is duplicated: {key}")
        seen.add(key)
        planned = plan_index[key]
        decision_id = str(planned["decision_id"])
        require(decision_id in metadata, f"decision metadata is missing: {decision_id}")
        meta = dict(metadata[decision_id])
        require(str(planned["image_id"]) == str(meta["image_id"]),
                f"image identity differs for {decision_id}")
        require(int(planned["attribute_id"]) == int(meta["attribute_id"]),
                f"attribute identity differs for {decision_id}")
        require(int(planned["target"]) == int(meta["target"]),
                f"target differs for {decision_id}")
        before = finite(source.get("correct_answer_teacher_forced_margin_before"),
                        "correct margin before")
        after = finite(source.get("correct_answer_teacher_forced_margin_after"),
                       "correct margin after")
        prior = baseline_values.setdefault(decision_id, before)
        require(math.isclose(prior, before, rel_tol=0.0, abs_tol=1e-9),
                f"baseline margin changes within {decision_id}")
        target = int(planned["target"])
        sign = 1.0 if target == 1 else -1.0
        correct_damage = before - after
        normalized.append({
            **source,
            **planned,
            **meta,
            "target": target,
            "correct_damage": correct_damage,
            "raw_yes_minus_no_before": sign * before,
            "raw_yes_minus_no_after": sign * after,
            "raw_yes_minus_no_damage": sign * correct_damage,
        })
    require(seen == set(plan_index), "outcomes do not exactly cover the Phase 10 plan")

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        if str(row["selection_method"]) == "global_control":
            continue
        grouped[(str(row["decision_id"]), str(row["stage"]),
                 str(row["selection_method"]), int(row["selection_k"]))].append(row)

    effects: list[dict[str, Any]] = []
    expected_random = len(plan.get("matched_random_seeds", []))
    for (decision_id, stage, selector, selection_k), values in sorted(grouped.items()):
        selected = [row for row in values if row["intervention_type"] == "top_evidence"]
        random_rows = [row for row in values if row["intervention_type"] == "matched_random"]
        require(len(selected) == 1, f"{decision_id}/{stage}/{selector}/{selection_k} lacks selected row")
        require(len(random_rows) == expected_random and expected_random > 0,
                f"{decision_id}/{stage}/{selector}/{selection_k} lacks matched-random rows")
        identity = selected[0]
        for scale, field in (
            ("correct_answer_margin", "correct_damage"),
            ("raw_yes_minus_no", "raw_yes_minus_no_damage"),
        ):
            effect = float(identity[field]) - statistics.mean(float(row[field]) for row in random_rows)
            effects.append({
                "decision_id": decision_id,
                "image_id": str(identity["image_id"]),
                "class_id": str(identity["class_id"]),
                "attribute_id": str(identity["attribute_id"]),
                "target": int(identity["target"]),
                "stage": stage,
                "selection_method": selector,
                "selection_k": selection_k,
                "metric_scale": scale,
                "effect": effect,
            })

    balanced_k = int(plan["balanced_k"])
    selectors = [str(value) for value in plan["selector_methods"]]
    require(len(selectors) == 2, "confirmatory selector difference requires exactly two selectors")
    balanced = [row for row in effects if int(row["selection_k"]) == balanced_k]
    effect_index = {
        (row["decision_id"], row["stage"], row["metric_scale"], row["selection_method"]): row
        for row in balanced
    }
    differences: list[dict[str, Any]] = []
    for decision_id in sorted({row["decision_id"] for row in balanced}):
        for stage in (str(plan["selected_stage"]), str(plan["neighbor_stage"])):
            for scale in ("correct_answer_margin", "raw_yes_minus_no"):
                left = effect_index[(decision_id, stage, scale, selectors[0])]
                right = effect_index[(decision_id, stage, scale, selectors[1])]
                differences.append({
                    **{key: left[key] for key in (
                        "decision_id", "image_id", "class_id", "attribute_id", "target",
                        "stage", "selection_k", "metric_scale"
                    )},
                    "left_selector": selectors[0],
                    "right_selector": selectors[1],
                    "effect": float(left["effect"]) - float(right["effect"]),
                })

    baselines = [{
        "decision_id": decision_id,
        "image_id": str(metadata[decision_id]["image_id"]),
        "class_id": str(metadata[decision_id]["class_id"]),
        "attribute_id": str(metadata[decision_id]["attribute_id"]),
        "target": int(metadata[decision_id]["target"]),
        "correct_answer_margin": margin,
        "raw_yes_minus_no_margin": margin if int(metadata[decision_id]["target"]) else -margin,
    } for decision_id, margin in sorted(baseline_values.items())]
    return effects, differences, baselines


def main_estimand_specs(
    *, stages: Sequence[str], selectors: Sequence[str], scales: Sequence[str]
) -> list[dict[str, Any]]:
    """Declare the full stage/polarity family before reading effect values."""

    require(len(selectors) == 2, "main estimands require two ordered selectors")
    specs: list[dict[str, Any]] = []
    for scale in scales:
        for stage in stages:
            for selector in selectors:
                for target in (0, 1):
                    specs.append({
                        "estimand": f"selector_effect::{scale}::{stage}::{selector}::target_{target}",
                        "metric_scale": scale,
                        "stage": stage,
                        "contrast_type": "selector_effect_by_target",
                        "source": "effects",
                        "selection_method": selector,
                        "target": target,
                        "target_difference": False,
                    })
                specs.append({
                    "estimand": f"selector_target_difference::{scale}::{stage}::{selector}",
                    "metric_scale": scale,
                    "stage": stage,
                    "contrast_type": "selector_target_difference",
                    "source": "effects",
                    "selection_method": selector,
                    "target": "",
                    "target_difference": True,
                })
            for target in (0, 1):
                specs.append({
                    "estimand": f"selector_difference::{scale}::{stage}::target_{target}",
                    "metric_scale": scale,
                    "stage": stage,
                    "contrast_type": "selector_difference_by_target",
                    "source": "differences",
                    "selection_method": f"{selectors[0]}_minus_{selectors[1]}",
                    "target": target,
                    "target_difference": False,
                })
            specs.append({
                "estimand": f"selector_difference_target_difference::{scale}::{stage}",
                "metric_scale": scale,
                "stage": stage,
                "contrast_type": "selector_difference_target_difference",
                "source": "differences",
                "selection_method": f"{selectors[0]}_minus_{selectors[1]}",
                "target": "",
                "target_difference": True,
            })
    return specs


def _rows_for_spec(
    spec: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    source = effects if spec["source"] == "effects" else differences
    rows = [row for row in source
            if str(row["metric_scale"]) == str(spec["metric_scale"])
            and str(row["stage"]) == str(spec["stage"])]
    if spec["source"] == "effects":
        rows = [row for row in rows
                if str(row["selection_method"]) == str(spec["selection_method"])]
    if spec["target"] != "":
        rows = [row for row in rows if int(row["target"]) == int(spec["target"])]
    require(bool(rows), f"estimand has no rows: {spec['estimand']}")
    return rows


def _image_target_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["image_id"])][int(row["target"])].append(finite(row["effect"], "effect"))
    return {
        image_id: {target: statistics.mean(values) for target, values in targets.items()}
        for image_id, targets in grouped.items()
    }


def _estimate_from_image_map(
    image_map: Mapping[str, Mapping[int, float]],
    chosen_images: Sequence[str],
    *, target_difference: bool,
) -> float | None:
    if target_difference:
        groups: dict[int, list[float]] = {0: [], 1: []}
        for image_id in chosen_images:
            for target, value in image_map.get(image_id, {}).items():
                groups[int(target)].append(float(value))
        if not groups[0] or not groups[1]:
            return None
        return statistics.mean(groups[1]) - statistics.mean(groups[0])
    values = [statistics.mean(list(image_map[image_id].values()))
              for image_id in chosen_images if image_id in image_map]
    return statistics.mean(values) if values else None


def simultaneous_main_inference(
    specs: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Pointwise percentile and max-t intervals, in separate raw/correct families."""

    require(samples >= 100 and 0.5 < confidence_level < 1.0, "invalid bootstrap settings")
    output: list[dict[str, Any]] = []
    for scale in sorted({str(spec["metric_scale"]) for spec in specs}):
        family = [dict(spec) for spec in specs if str(spec["metric_scale"]) == scale]
        maps = {
            spec["estimand"]: _image_target_map(_rows_for_spec(spec, effects, differences))
            for spec in family
        }
        images = sorted({image_id for image_map in maps.values() for image_id in image_map})
        require(len(images) >= 2, f"{scale} family needs at least two images")
        estimates = {
            spec["estimand"]: _estimate_from_image_map(
                maps[spec["estimand"]], images,
                target_difference=bool(spec["target_difference"]),
            ) for spec in family
        }
        require(all(value is not None for value in estimates.values()),
                f"{scale} family cannot estimate all target contrasts")
        rng = random.Random(stable_seed(seed, "simultaneous", scale))
        draws: dict[str, list[float]] = {spec["estimand"]: [] for spec in family}
        attempts = 0
        while len(next(iter(draws.values()))) < samples and attempts < samples * 20:
            attempts += 1
            chosen = [images[rng.randrange(len(images))] for _ in images]
            current = {
                spec["estimand"]: _estimate_from_image_map(
                    maps[spec["estimand"]], chosen,
                    target_difference=bool(spec["target_difference"]),
                ) for spec in family
            }
            if any(value is None for value in current.values()):
                continue
            for name, value in current.items():
                assert value is not None
                draws[name].append(value)
        require(len(next(iter(draws.values()))) == samples,
                f"could not bootstrap the complete {scale} family")
        scales_sd = {
            name: statistics.stdev(values) if len(values) > 1 else 0.0
            for name, values in draws.items()
        }
        variable = [name for name, value in scales_sd.items() if value > 0]
        maxima = []
        for index in range(samples):
            maxima.append(max(
                abs(draws[name][index] - float(estimates[name])) / scales_sd[name]
                for name in variable
            ) if variable else 0.0)
        critical = percentile(maxima, confidence_level)
        alpha = (1.0 - confidence_level) / 2.0
        for spec in family:
            name = str(spec["estimand"])
            estimate = float(estimates[name])
            values = draws[name]
            image_map = maps[name]
            target_images = {
                target: sum(target in row for row in image_map.values()) for target in (0, 1)
            }
            output.append({
                **spec,
                "estimate": estimate,
                "ci_low": percentile(values, alpha),
                "ci_high": percentile(values, 1.0 - alpha),
                "simultaneous_ci_low": estimate - critical * scales_sd[name],
                "simultaneous_ci_high": estimate + critical * scales_sd[name],
                "confidence_level": confidence_level,
                "familywise_confidence_level": confidence_level,
                "bootstrap_samples": samples,
                "bootstrap_unit": "image_id",
                "family_id": f"main_{scale}",
                "family_size": len(family),
                "critical_max_t": critical,
                "n_images": len(image_map),
                "n_target_0_images": target_images[0],
                "n_target_1_images": target_images[1],
            })
    return sorted(output, key=lambda row: str(row["estimand"]))


def blocked_robustness(
    specs: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Equal-group bootstrap plus complete leave-one-attribute/species diagnostics."""

    require(samples >= 100 and 0.5 < confidence_level < 1.0, "invalid bootstrap settings")
    summaries: list[dict[str, Any]] = []
    leave_one_out: list[dict[str, Any]] = []
    for spec in specs:
        rows = _rows_for_spec(spec, effects, differences)
        for group_field in group_fields:
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                require(group_field in row and str(row[group_field]) != "",
                        f"{spec['estimand']} lacks {group_field}")
                grouped[str(row[group_field])].append(row)
            group_values: dict[str, float] = {}
            for group, group_rows in grouped.items():
                image_map = _image_target_map(group_rows)
                estimate = _estimate_from_image_map(
                    image_map, sorted(image_map),
                    target_difference=bool(spec["target_difference"]),
                )
                if estimate is not None:
                    group_values[group] = estimate
            groups = sorted(group_values)
            require(len(groups) >= 2, f"{spec['estimand']} needs two eligible {group_field} groups")
            estimate = statistics.mean(group_values.values())
            rng = random.Random(stable_seed(seed, "blocked", spec["estimand"], group_field))
            draws = [statistics.mean(group_values[groups[rng.randrange(len(groups))]]
                                     for _ in groups) for _ in range(samples)]
            alpha = (1.0 - confidence_level) / 2.0
            loo: dict[str, float] = {}
            for omitted in groups:
                value = statistics.mean(group_values[group] for group in groups if group != omitted)
                loo[omitted] = value
                leave_one_out.append({
                    "estimand": spec["estimand"],
                    "metric_scale": spec["metric_scale"],
                    "stage": spec["stage"],
                    "contrast_type": spec["contrast_type"],
                    "selection_method": spec["selection_method"],
                    "target": spec["target"],
                    "grouping": group_field,
                    "omitted_group": omitted,
                    "leave_one_group_out_estimate": value,
                    "full_equal_group_estimate": estimate,
                    "change_from_full": value - estimate,
                })
            minimum = min(loo, key=loo.get)
            maximum = max(loo, key=loo.get)
            summaries.append({
                "estimand": spec["estimand"],
                "metric_scale": spec["metric_scale"],
                "stage": spec["stage"],
                "contrast_type": spec["contrast_type"],
                "selection_method": spec["selection_method"],
                "target": spec["target"],
                "grouping": group_field,
                "equal_group_estimate": estimate,
                "cluster_bootstrap_ci_low": percentile(draws, alpha),
                "cluster_bootstrap_ci_high": percentile(draws, 1.0 - alpha),
                "confidence_level": confidence_level,
                "bootstrap_samples": samples,
                "n_groups_total": len(grouped),
                "n_groups_eligible": len(groups),
                "leave_one_group_out_min": loo[minimum],
                "leave_one_group_out_min_omitted": minimum,
                "leave_one_group_out_max": loo[maximum],
                "leave_one_group_out_max_omitted": maximum,
                "max_absolute_leave_one_out_change": max(
                    abs(value - estimate) for value in loo.values()
                ),
            })
    return summaries, leave_one_out


def _cluster_accuracy_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(finite(row[value_field], value_field))
    require(bool(by_image), f"accuracy metric {value_field} has no evaluable rows")
    images = sorted(by_image)
    # Accuracy is the fraction of attribute decisions answered correctly. Keep
    # that decision-weighted estimand while resampling whole images to respect
    # dependence among decisions made on the same photograph.
    estimate = statistics.mean(
        value for values in by_image.values() for value in values
    )
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [images[rng.randrange(len(images))] for _ in images]
        values = [value for image in sampled for value in by_image[image]]
        draws.append(statistics.mean(values))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": estimate,
        "ci_low": percentile(draws, alpha),
        "ci_high": percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "bootstrap_unit": "image_id",
        "estimand_weighting": "decision",
        "n_images": len(images),
        "n_evaluable": len(rows),
    }


def absolute_accuracy_table(
    baselines: Sequence[Mapping[str, Any]],
    behavior_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Report absolute teacher-forced and generated accuracy, including parse rates."""

    expected_decisions = {str(row["decision_id"]) for row in baselines}
    output: list[dict[str, Any]] = []

    baseline_values = [{
        **row,
        "value": float(finite(row["correct_answer_margin"], "correct_answer_margin") > 0),
    } for row in baselines]

    def append_metric(
        *, model: str, source: str, control: str, metric: str,
        rows: Sequence[Mapping[str, Any]], target: int | str,
    ) -> None:
        interval = _cluster_accuracy_interval(
            rows, value_field="value", samples=samples,
            seed=stable_seed(seed, "accuracy", model, control, metric, target),
            confidence_level=confidence_level,
        )
        output.append({
            "model": model,
            "source": source,
            "control": control,
            "metric": metric,
            "target": target,
            **interval,
        })

    for target in ("all", 0, 1):
        subset = baseline_values if target == "all" else [
            row for row in baseline_values if int(row["target"]) == target
        ]
        append_metric(
            model="llava", source="phase10_intervention_baseline", control="image",
            metric="teacher_forced_accuracy", rows=subset, target=target,
        )

    for model, source_rows in sorted(behavior_by_model.items()):
        rows = [dict(row) for row in source_rows]
        require(bool(rows), f"behavior table is empty for {model}")
        decisions = {str(row["decision_id"]) for row in rows}
        require(decisions == expected_decisions,
                f"{model} behavior decisions do not match the Phase 10 cohort")
        controls = sorted({str(row["control"]) for row in rows})
        for control in controls:
            control_rows = [row for row in rows if str(row["control"]) == control]
            require(len(control_rows) == len(expected_decisions),
                    f"{model}/{control} does not contain one row per decision")
            for target in ("all", 0, 1):
                subset = control_rows if target == "all" else [
                    row for row in control_rows if int(row["target"]) == target
                ]
                teacher = [{**row, "value": float(int(row["margin_correct"]))} for row in subset]
                append_metric(
                    model=model, source="retained_behavior", control=control,
                    metric="teacher_forced_accuracy", rows=teacher, target=target,
                )
                parseable = [row for row in subset if str(row.get("generation_correct", "")) != ""]
                generation_all = [{
                    **row,
                    "value": float(int(row["generation_correct"]))
                    if str(row.get("generation_correct", "")) != "" else 0.0,
                } for row in subset]
                append_metric(
                    model=model, source="retained_behavior", control=control,
                    metric="generation_accuracy_all", rows=generation_all, target=target,
                )
                parse_rows = [{
                    **row, "value": float(str(row.get("generation_correct", "")) != "")
                } for row in subset]
                append_metric(
                    model=model, source="retained_behavior", control=control,
                    metric="generation_parse_rate", rows=parse_rows, target=target,
                )
                if parseable:
                    parsed_values = [{
                        **row, "value": float(int(row["generation_correct"]))
                    } for row in parseable]
                    append_metric(
                        model=model, source="retained_behavior", control=control,
                        metric="generation_accuracy_parseable", rows=parsed_values,
                        target=target,
                    )
    return output


def run_confirmatory_analysis(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    decision_metadata: Iterable[Mapping[str, Any]],
    behavior_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    bootstrap_samples: int = 10000,
    seed: int = 20260915,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Run the complete retained-data analysis and return paper-ready tables."""

    metadata = decision_metadata_index(decision_metadata)
    effects, differences, baselines = build_effect_tables(
        outcomes, plan=plan, metadata=metadata
    )
    stages = (str(plan["selected_stage"]), str(plan["neighbor_stage"]))
    selectors = tuple(str(value) for value in plan["selector_methods"])
    scales = ("raw_yes_minus_no", "correct_answer_margin")
    specs = main_estimand_specs(stages=stages, selectors=selectors, scales=scales)
    inference = simultaneous_main_inference(
        specs, effects, differences, samples=bootstrap_samples, seed=seed,
        confidence_level=confidence_level,
    )
    robustness, leave_one_out = blocked_robustness(
        specs, effects, differences, group_fields=("attribute_id", "class_id"),
        samples=bootstrap_samples, seed=seed, confidence_level=confidence_level,
    )
    accuracy = absolute_accuracy_table(
        baselines, behavior_by_model, samples=bootstrap_samples, seed=seed,
        confidence_level=confidence_level,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "purpose": PURPOSE,
        "development_only": True,
        "official_test_images_used": 0,
        "model_forwards_performed": False,
        "decision_count": len(baselines),
        "image_count": len({row["image_id"] for row in baselines}),
        "attribute_count": len({row["attribute_id"] for row in baselines}),
        "species_count": len({row["class_id"] for row in baselines}),
        "outcome_count": len(plan["records"]),
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "main_family_definition": (
            "Separate 18-estimand max-t families for raw yes-minus-no and "
            "correct-answer margins across both stages, selectors, targets, and interactions."
        ),
        "effect_direction": (
            "Positive values mean selected-patch mean replacement reduces the stated margin "
            "more than matched-random replacement."
        ),
        "claim_boundary": (
            "Raw-margin results test whether the polarity pattern survives without the target-sign "
            "transformation. Mean-replacement effects show controlled sensitivity, not natural use, "
            "literal information erasure, or distinct positive/negative routing mechanisms."
        ),
        "selector_effect_rows": effects,
        "selector_difference_rows": differences,
        "baseline_rows": baselines,
        "main_inference": inference,
        "blocked_robustness": robustness,
        "leave_one_group_out": leave_one_out,
        "absolute_accuracy": accuracy,
    }
