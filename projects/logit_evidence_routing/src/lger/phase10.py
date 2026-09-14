"""Balanced-target and K-dose intervention extensions for CUB development data."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .paper_robustness import simultaneous_image_bootstrap
from .phase7 import PRIMARY_METRIC, clustered_paired_bootstrap
from .phase9 import (
    GLOBAL_METHOD,
    GLOBAL_TYPE,
    Phase9ValidationError,
    build_selector_intervention_plan,
)


SCHEMA_VERSION = 1
PURPOSE_PLAN = "phase10_development_balanced_target_and_k_dose_plan"
PURPOSE_ANALYSIS = "phase10_development_balanced_target_and_k_dose_analysis"


class Phase10ValidationError(Phase9ValidationError):
    """Raised when the Priority 1 extension is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase10ValidationError(message)


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_priority1_plan(
    token_rows: Iterable[Mapping[str, Any]],
    *,
    selected_stage: str,
    neighbor_stage: str,
    stage_order: Sequence[str],
    selector_methods: Sequence[str],
    balanced_k: int = 32,
    dose_selector: str = "vision_cls_attention",
    dose_k_values: Sequence[int] = (8, 16, 32, 64),
    random_seeds: Sequence[int] = (0, 1, 2),
    seed: int = 909,
    norm_quantiles: int = 4,
) -> dict[str, Any]:
    """Build one deduplicated plan for balanced targets and a selector K curve.

    Matching is performed separately within each target stratum.  This retains
    the original positive-target Phase 9 design at K=32, allowing exact reuse of
    matching retained outcomes when the source metadata and seed are unchanged.
    """

    copied = []
    for source in token_rows:
        row = dict(source)
        raw_token_index = row.get("token_index")
        require(not isinstance(raw_token_index, bool), "token_index must be an integer")
        try:
            row["token_index"] = int(raw_token_index)
        except (TypeError, ValueError) as error:
            raise Phase10ValidationError("token_index must be an integer") from error
        require(row["token_index"] >= 0, "token_index must be nonnegative")
        copied.append(row)
    require(copied, "selector token metadata is empty")
    methods = tuple(str(value) for value in selector_methods)
    ks = tuple(int(value) for value in dose_k_values)
    require(len(methods) == 2 and len(set(methods)) == 2,
            "balanced intervention requires exactly two selectors")
    require(dose_selector in methods, "dose selector must be one of the balanced selectors")
    require(ks == tuple(sorted(set(ks))) and balanced_k in ks and all(value > 0 for value in ks),
            "dose K values must be unique, sorted, positive, and include balanced_k")
    identities: dict[str, tuple[str, int]] = {}
    for row in copied:
        decision_id = str(row["decision_id"])
        identity = (str(row["image_id"]), int(row["target"]))
        prior = identities.setdefault(decision_id, identity)
        require(prior == identity, f"decision identity differs: {decision_id}")
    target_counts = {
        target: sum(value[1] == target for value in identities.values()) for target in (0, 1)
    }
    require(target_counts[0] == target_counts[1] and target_counts[0] > 0,
            "Priority 1 plan requires a nonempty target-balanced decision cohort")

    records: list[dict[str, Any]] = []
    matching: dict[str, Any] = {}
    for target in (0, 1):
        target_rows = [row for row in copied if int(row["target"]) == target]
        for k in ks:
            selected_methods = methods if k == balanced_k else (dose_selector,)
            component_rows = [
                row for row in target_rows
                if str(row["selection_method"]) in selected_methods
            ]
            spatial_mode = "exact_bird_box"
            try:
                component = build_selector_intervention_plan(
                    component_rows,
                    selected_stage=selected_stage,
                    neighbor_stage=neighbor_stage,
                    stage_order=stage_order,
                    k=k,
                    selector_methods=selected_methods,
                    random_seeds=random_seeds,
                    seed=seed,
                    norm_quantiles=norm_quantiles,
                    include_global_control=(k == balanced_k),
                )
            except Phase9ValidationError as error:
                # Large K can make exact inside/outside-box matching impossible
                # even though an otherwise matched random control exists.  The
                # sole predeclared fallback drops the spatial stratum while
                # retaining target stratification and the norm-bin coarsening
                # performed by Phase 9.  Partial spatial metadata remains an
                # error, and all fallback use is recorded in the frozen plan.
                if "lacks exact matched-random support" not in str(error):
                    raise
                spatial_mode = "norm_only_fallback"
                spatial_fields = {
                    "bird_box_membership", "bird_box_stratum", "inside_bird_box"
                }
                fallback_rows = [
                    {key: value for key, value in row.items() if key not in spatial_fields}
                    for row in component_rows
                ]
                component = build_selector_intervention_plan(
                    fallback_rows,
                    selected_stage=selected_stage,
                    neighbor_stage=neighbor_stage,
                    stage_order=stage_order,
                    k=k,
                    selector_methods=selected_methods,
                    random_seeds=random_seeds,
                    seed=seed,
                    norm_quantiles=norm_quantiles,
                    include_global_control=(k == balanced_k),
                )
            matching[f"target_{target}_k_{k}"] = {
                "spatial_mode": spatial_mode,
                "norm_quantiles_by_selector": component[
                    "matching_norm_quantiles_by_selector"
                ],
            }
            for source in component["records"]:
                method = str(source["selection_method"])
                families = ["balanced_target"] if k == balanced_k else []
                if method == dose_selector:
                    families.append("k_dose_response")
                records.append({
                    **source,
                    "selection_k": 0 if method == GLOBAL_METHOD else k,
                    "analysis_families": families,
                    "matching_target_stratum": target,
                    "matching_spatial_mode": spatial_mode,
                })

    keys = {(str(row["decision_id"]), str(row["intervention_id"])) for row in records}
    require(len(keys) == len(records), "combined Priority 1 intervention IDs are not unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "purpose": PURPOSE_PLAN,
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "official_test_images_used": 0,
        "selected_stage": selected_stage,
        "neighbor_stage": neighbor_stage,
        "score_origin_stage": selected_stage,
        "stage_order": list(stage_order),
        "selector_methods": list(methods),
        "balanced_k": balanced_k,
        "dose_selector": dose_selector,
        "dose_k_values": list(ks),
        "matched_random_seeds": list(random_seeds),
        "norm_quantiles": norm_quantiles,
        "matching_stratified_by_target": True,
        "matching_details": matching,
        "matching_spatial_fallback_count": sum(
            value["spatial_mode"] == "norm_only_fallback" for value in matching.values()
        ),
        "replacement": "development_train_stage_mean",
        "fixed_mask_across_stages": True,
        "target_counts": {str(key): value for key, value in target_counts.items()},
        "decision_count": len(identities),
        "image_count": len({value[0] for value in identities.values()}),
        "intervention_count": len(records),
        "records": records,
    }


def _group_interval(
    values: Sequence[tuple[str, int, float]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Image-cluster bootstrap for group-one minus group-zero."""

    by_image: dict[str, dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    for image_id, group, value in values:
        require(group in (0, 1), "interaction group must be binary")
        by_image[str(image_id)][group].append(float(value))
    image_values = {
        image_id: {group: statistics.mean(group_values)
                   for group, group_values in groups.items() if group_values}
        for image_id, groups in by_image.items()
    }

    def contrast(chosen: Sequence[str]) -> float | None:
        groups: dict[int, list[float]] = {0: [], 1: []}
        for image_id in chosen:
            for group, value in image_values[image_id].items():
                groups[group].append(value)
        if not groups[0] or not groups[1]:
            return None
        return statistics.mean(groups[1]) - statistics.mean(groups[0])

    images = sorted(image_values)
    estimate = contrast(images)
    require(estimate is not None, "both target groups are required")
    rng = random.Random(seed)
    draws: list[float] = []
    attempts = 0
    while len(draws) < samples and attempts < samples * 20:
        attempts += 1
        draw = contrast([images[rng.randrange(len(images))] for _ in images])
        if draw is not None:
            draws.append(draw)
    require(len(draws) == samples, "could not bootstrap both target groups")
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": estimate,
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "n_images": len(images),
        "cluster_unit": "image_id",
    }


def _interval(
    values: Sequence[Mapping[str, Any]], *, samples: int, seed: int, confidence_level: float
) -> dict[str, Any]:
    return clustered_paired_bootstrap(
        [str(row["image_id"]) for row in values], [float(row["effect"]) for row in values],
        samples=samples, seed=seed, confidence_level=confidence_level,
    )


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    require(denominator > 0, "dose slope needs distinct K values")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def aggregate_priority1(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    bootstrap_samples: int = 10000,
    seed: int = 20260914,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Aggregate balanced-target interactions and Vision-CLS K dose-response."""

    require(plan.get("status") == "PASS" and plan.get("purpose") == PURPOSE_PLAN,
            "a passing Priority 1 plan is required")
    require(plan.get("development_only") is True and plan.get("official_test_images_used") == 0,
            "Priority 1 aggregation is development-only")
    records = plan.get("records")
    require(isinstance(records, list) and records, "Priority 1 plan records are missing")
    index = {(str(row["decision_id"]), str(row["intervention_id"])): dict(row)
             for row in records}
    require(len(index) == len(records), "Priority 1 plan identities are not unique")

    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    baselines: dict[tuple[str, str], float] = {}
    for source in outcomes:
        key = (str(source.get("decision_id", "")), str(source.get("intervention_id", "")))
        require(key in index and key not in seen, "outcome identity is absent or duplicated")
        seen.add(key)
        planned = index[key]
        before = float(source["correct_answer_teacher_forced_margin_before"])
        after = float(source["correct_answer_teacher_forced_margin_after"])
        require(math.isfinite(before) and math.isfinite(after), "outcome margins must be finite")
        baseline_key = (str(planned["decision_id"]), str(planned["stage"]))
        previous = baselines.setdefault(baseline_key, before)
        require(math.isclose(previous, before, rel_tol=0, abs_tol=1e-9),
                f"baseline changes within {baseline_key}")
        # The immutable plan owns all design fields.  This also permits exact
        # reuse of Phase 9 CSV rows, which predate the ``selection_k`` column.
        normalized.append({**source, **planned, "damage": before - after})
    require(seen == set(index), "outcomes do not cover the complete Priority 1 plan")

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    global_rows: list[dict[str, Any]] = []
    for row in normalized:
        if row["selection_method"] == GLOBAL_METHOD:
            global_rows.append(row)
        else:
            grouped[(str(row["decision_id"]), str(row["stage"]),
                     str(row["selection_method"]), int(row["selection_k"]))].append(row)
    effects: list[dict[str, Any]] = []
    for (decision_id, stage, method, k), rows in sorted(grouped.items()):
        top = [row for row in rows if row["intervention_type"] == "top_evidence"]
        matched = [row for row in rows if row["intervention_type"] == "matched_random"]
        require(len(top) == 1 and matched, "incomplete Priority 1 top/random design")
        effects.append({
            "decision_id": decision_id,
            "image_id": str(top[0]["image_id"]),
            "attribute_id": int(top[0]["attribute_id"]),
            "target": int(top[0]["target"]),
            "stage": stage,
            "selection_method": method,
            "selection_k": k,
            "effect": float(top[0]["damage"]) - statistics.mean(
                float(row["damage"]) for row in matched
            ),
        })

    balanced_k = int(plan["balanced_k"])
    methods = [str(value) for value in plan["selector_methods"]]
    stages = [str(plan["selected_stage"]), str(plan["neighbor_stage"])]
    balanced = [row for row in effects if row["selection_k"] == balanced_k]
    by_design: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in balanced:
        by_design[(row["stage"], row["selection_method"])].append(row)

    selector_effects = []
    target_stratified = []
    target_interactions = []
    for stage in stages:
        for method in methods:
            values = by_design[(stage, method)]
            interval = _interval(
                values, samples=bootstrap_samples,
                seed=_stable_seed(seed, "balanced", stage, method),
                confidence_level=confidence_level,
            )
            selector_effects.append({"stage": stage, "selection_method": method,
                                     "selection_k": balanced_k, **interval})
            for target in (0, 1):
                subset = [row for row in values if row["target"] == target]
                target_stratified.append({
                    "stage": stage, "selection_method": method, "target": target,
                    "selection_k": balanced_k,
                    **_interval(subset, samples=bootstrap_samples,
                                seed=_stable_seed(seed, "target", stage, method, target),
                                confidence_level=confidence_level),
                })
            target_interactions.append({
                "stage": stage, "selection_method": method, "selection_k": balanced_k,
                "contrast_direction": "target_1_minus_target_0_selector_effect",
                **_group_interval(
                    [(row["image_id"], row["target"], row["effect"]) for row in values],
                    samples=bootstrap_samples,
                    seed=_stable_seed(seed, "target_interaction", stage, method),
                    confidence_level=confidence_level,
                ),
            })

    effect_index = {
        (row["decision_id"], row["stage"], row["selection_method"], row["selection_k"]): row
        for row in effects
    }
    decision_ids = sorted({row["decision_id"] for row in balanced})
    selector_by_target = []
    selector_differences = []
    for stage in stages:
        differences = []
        for decision_id in decision_ids:
            left = effect_index[(decision_id, stage, methods[0], balanced_k)]
            right = effect_index[(decision_id, stage, methods[1], balanced_k)]
            differences.append({**left, "effect": left["effect"] - right["effect"]})
        for target in (0, 1):
            subset = [row for row in differences if row["target"] == target]
            selector_differences.append({
                "stage": stage, "target": target,
                "left_selector": methods[0], "right_selector": methods[1],
                **_interval(subset, samples=bootstrap_samples,
                            seed=_stable_seed(seed, "selector_difference", stage, target),
                            confidence_level=confidence_level),
            })
        selector_by_target.append({
            "stage": stage, "left_selector": methods[0], "right_selector": methods[1],
            "contrast_direction": "target_1_minus_target_0_of_selector_difference",
            **_group_interval(
                [(row["image_id"], row["target"], row["effect"]) for row in differences],
                samples=bootstrap_samples,
                seed=_stable_seed(seed, "selector_by_target", stage),
                confidence_level=confidence_level,
            ),
        })

    dose_selector = str(plan["dose_selector"])
    ks = [int(value) for value in plan["dose_k_values"]]
    dose_effect_rows = [row for row in effects if row["selection_method"] == dose_selector]
    dose_effects = []
    simultaneous_input = []
    for stage in stages:
        for k in ks:
            values = [row for row in dose_effect_rows if row["stage"] == stage
                      and row["selection_k"] == k]
            interval = _interval(
                values, samples=bootstrap_samples,
                seed=_stable_seed(seed, "dose", stage, k),
                confidence_level=confidence_level,
            )
            dose_effects.append({"stage": stage, "selection_method": dose_selector,
                                 "selection_k": k, **interval})
            simultaneous_input.extend({
                "estimand": f"dose::{dose_selector}::{stage}::k_{k}",
                "image_id": row["image_id"], "value": row["effect"],
            } for row in values)

    dose_simultaneous = simultaneous_image_bootstrap(
        simultaneous_input, samples=bootstrap_samples,
        seed=_stable_seed(seed, "dose_simultaneous"), confidence_level=confidence_level,
    )
    dose_adjacent = []
    dose_trends = []
    for stage in stages:
        stage_decisions = sorted({row["decision_id"] for row in dose_effect_rows
                                  if row["stage"] == stage})
        for left_k, right_k in zip(ks, ks[1:]):
            values = []
            for decision_id in stage_decisions:
                left = effect_index[(decision_id, stage, dose_selector, left_k)]
                right = effect_index[(decision_id, stage, dose_selector, right_k)]
                values.append({**right, "effect": right["effect"] - left["effect"]})
            dose_adjacent.append({
                "stage": stage, "left_k": left_k, "right_k": right_k,
                "contrast_direction": "larger_k_minus_smaller_k_selector_effect",
                **_interval(values, samples=bootstrap_samples,
                            seed=_stable_seed(seed, "dose_adjacent", stage, left_k, right_k),
                            confidence_level=confidence_level),
            })
        slopes = []
        xs = [math.log2(k) for k in ks]
        for decision_id in stage_decisions:
            rows = [effect_index[(decision_id, stage, dose_selector, k)] for k in ks]
            slopes.append({
                "image_id": rows[0]["image_id"],
                "effect": _linear_slope(xs, [row["effect"] for row in rows]),
            })
        dose_trends.append({
            "stage": stage, "selection_method": dose_selector,
            "contrast_direction": "effect_change_per_doubling_of_k",
            **_interval(slopes, samples=bootstrap_samples,
                        seed=_stable_seed(seed, "dose_slope", stage),
                        confidence_level=confidence_level),
        })

    globals_summary = []
    for stage in stages:
        values = [row for row in global_rows if row["stage"] == stage]
        globals_summary.append({
            "stage": stage,
            **clustered_paired_bootstrap(
                [str(row["image_id"]) for row in values], [float(row["damage"]) for row in values],
                samples=bootstrap_samples, seed=_stable_seed(seed, "global", stage),
                confidence_level=confidence_level,
            ),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "purpose": PURPOSE_ANALYSIS,
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "official_test_images_used": 0,
        "primary_metric": PRIMARY_METRIC,
        "decision_count": len({row["decision_id"] for row in effects}),
        "image_count": len({row["image_id"] for row in effects}),
        "outcome_count": len(normalized),
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "selector_specific_contrasts": selector_effects,
        "target_stratified_selector_effects": target_stratified,
        "selector_target_interactions": target_interactions,
        "selector_differences_by_target": selector_differences,
        "selector_by_target_interactions": selector_by_target,
        "dose_effects": dose_effects,
        "dose_simultaneous_intervals": dose_simultaneous,
        "dose_adjacent_differences": dose_adjacent,
        "dose_log2_trends": dose_trends,
        "global_manipulation_checks": globals_summary,
        "claim_boundary": (
            "Development-only mean-replacement interventions. Target interactions test scope; "
            "K trends test mask-size robustness. Neither establishes literal information erasure, "
            "a natural bottleneck, or official-test generalization."
        ),
    }
