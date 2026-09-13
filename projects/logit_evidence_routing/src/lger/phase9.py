"""Development-only selector-aligned causal follow-up after the frozen Phase 8 protocol.

Phase 7 intervened on probe-ranked patches.  Phase 6, however, selected a
Vision-CLS versus Logit-Lens localization mismatch.  This module closes that
design gap by applying masks from each diagnostic selector at the same two
stages, with the same mask held fixed across stages.  It does not alter or
replace the frozen Phase 8 confirmatory analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .phase7 import (
    PRIMARY_METRIC,
    REPLACEMENT,
    Phase7ValidationError,
    build_intervention_plan,
    clustered_paired_bootstrap,
)


SCHEMA_VERSION = 1
PURPOSE_PLAN = "phase9_development_selector_aligned_mechanism_plan"
PURPOSE_ANALYSIS = "phase9_development_selector_aligned_mechanism_analysis"
GLOBAL_METHOD = "global_control"
GLOBAL_TYPE = "global_all"


class Phase9ValidationError(Phase7ValidationError):
    """Raised when the development-only mechanism design is not auditable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase9ValidationError(message)


def _identifier(value: Any, field: str) -> str:
    _require(isinstance(value, (str, int)) and str(value).strip(), f"{field} is required")
    return str(value)


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Phase9ValidationError(f"{field} must be numeric") from error
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _binary(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise Phase9ValidationError(f"{field} must be binary") from error
    _require(result in (0, 1), f"{field} must be binary")
    return result


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _intervention_id(
    decision_id: str,
    method: str,
    stage: str,
    intervention_type: str,
    replicate: int,
    indices: Sequence[int],
) -> str:
    payload = json.dumps(
        [decision_id, method, stage, intervention_type, replicate, list(indices)],
        separators=(",", ":"),
    )
    return "p9-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_selector_intervention_plan(
    token_rows: Iterable[Mapping[str, Any]],
    *,
    selected_stage: str,
    neighbor_stage: str,
    stage_order: Sequence[str],
    k: int,
    selector_methods: Sequence[str],
    random_seeds: Sequence[int] = (0, 1, 2),
    seed: int = 909,
    norm_quantiles: int = 4,
    include_low_evidence: bool = False,
    include_global_control: bool = True,
) -> dict[str, Any]:
    """Build fixed-mask selector interventions for development decisions.

    Input rows describe scores at ``selected_stage`` only and require
    ``decision_id``, ``image_id``, ``attribute_id``, ``target``,
    ``phase5_failure``, ``selection_method``, ``stage``, ``token_index`` and
    ``evidence_score``.  Each selector must cover the same decisions and token
    universe.  A selected mask and each matched-random mask are copied unchanged
    to the neighboring stage, which makes the stage comparison interpretable.
    """

    _require(isinstance(include_low_evidence, bool), "include_low_evidence must be boolean")
    _require(isinstance(include_global_control, bool), "include_global_control must be boolean")
    methods = tuple(str(item) for item in selector_methods)
    _require(methods and len(methods) == len(set(methods)), "selector_methods must be unique")
    _require(GLOBAL_METHOD not in methods, f"{GLOBAL_METHOD} is reserved")
    seeds = tuple(random_seeds)
    _require(seeds and len(seeds) == len(set(seeds)), "random_seeds must be nonempty and unique")

    copied = [dict(row) for row in token_rows]
    _require(copied, "selector token metadata is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    universes: dict[tuple[str, str], tuple[int, ...]] = {}
    for row in copied:
        decision_id = _identifier(row.get("decision_id"), "decision_id")
        method = _identifier(row.get("selection_method"), "selection_method")
        _require(method in methods, f"unexpected selection_method: {method}")
        _require(_identifier(row.get("stage"), "stage") == selected_stage,
                 "selector scores must originate at selected_stage")
        identity = {
            "decision_id": decision_id,
            "image_id": row.get("image_id"),
            "attribute_id": int(row.get("attribute_id")),
            "target": _binary(row.get("target"), "target"),
            "phase5_failure": _binary(row.get("phase5_failure"), "phase5_failure"),
        }
        previous = identities.setdefault(decision_id, identity)
        _require(previous == identity, f"decision metadata differs across selectors: {decision_id}")
        grouped[method].append(row)

    decision_sets = {
        method: {str(row["decision_id"]) for row in rows}
        for method, rows in grouped.items()
    }
    _require(set(grouped) == set(methods), "selector metadata does not cover every requested method")
    reference_decisions = decision_sets[methods[0]]
    _require(reference_decisions and all(value == reference_decisions for value in decision_sets.values()),
             "selectors must cover the same decisions")

    records: list[dict[str, Any]] = []
    token_universe_by_decision: dict[str, tuple[int, ...]] = {}
    matching_quantiles_by_method: dict[str, int] = {}
    for method in methods:
        source_rows = grouped[method]
        by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            by_decision[str(row["decision_id"])].append(row)
        for decision_id, rows in by_decision.items():
            indices = tuple(sorted(int(row["token_index"]) for row in rows))
            _require(len(indices) == len(set(indices)),
                     f"duplicate token index for {decision_id}/{method}")
            prior = token_universe_by_decision.setdefault(decision_id, indices)
            _require(prior == indices, f"selector token universe differs for {decision_id}")
            universes[(decision_id, method)] = indices

        # Reuse the audited Phase 7 selection/matching implementation.  Scores
        # are duplicated only to satisfy its two-stage schema; selections are
        # taken from selected_stage and then explicitly copied across stages.
        duplicated: list[dict[str, Any]] = []
        for row in source_rows:
            duplicated.append({**row, "stage": selected_stage})
            duplicated.append({**row, "stage": neighbor_stage})
        attempts = []
        for candidate in (norm_quantiles, 2, 1):
            if candidate > norm_quantiles or candidate in attempts:
                continue
            attempts.append(candidate)
        base = None
        last_error = None
        for matching_quantiles in attempts:
            try:
                base = build_intervention_plan(
                    duplicated,
                    selected_stage=selected_stage,
                    neighbor_stage=neighbor_stage,
                    stage_order=stage_order,
                    k=k,
                    seed=_stable_seed(seed, method),
                    random_replicates=len(seeds),
                    random_seeds=seeds,
                    norm_quantiles=matching_quantiles,
                )
                matching_quantiles_by_method[method] = matching_quantiles
                break
            except Phase7ValidationError as error:
                if "insufficient matched-random candidates" not in str(error):
                    raise
                last_error = error
        if base is None:
            assert last_error is not None
            raise Phase9ValidationError(
                f"selector {method} lacks exact matched-random support even after "
                "predeclared quantile coarsening"
            ) from last_error
        source_records = [record for record in base["records"] if record["stage"] == selected_stage]
        if not include_low_evidence:
            source_records = [record for record in source_records
                              if record["intervention_type"] != "low_evidence"]
        for source in source_records:
            identity = identities[str(source["decision_id"])]
            for stage, role in ((selected_stage, "selected"), (neighbor_stage, "neighbor")):
                record = {
                    **source,
                    **identity,
                    "selection_method": method,
                    "score_origin_stage": selected_stage,
                    "stage": stage,
                    "stage_role": role,
                }
                record["intervention_id"] = _intervention_id(
                    record["decision_id"], method, stage, record["intervention_type"],
                    int(record["replicate"]), record["token_indices"],
                )
                records.append(record)

    if include_global_control:
        for decision_id in sorted(reference_decisions):
            identity = identities[decision_id]
            indices = list(token_universe_by_decision[decision_id])
            for stage, role in ((selected_stage, "selected"), (neighbor_stage, "neighbor")):
                records.append({
                    **identity,
                    "intervention_id": _intervention_id(
                        decision_id, GLOBAL_METHOD, stage, GLOBAL_TYPE, 0, indices
                    ),
                    "selection_method": GLOBAL_METHOD,
                    "score_origin_stage": selected_stage,
                    "stage": stage,
                    "stage_role": role,
                    "intervention_type": GLOBAL_TYPE,
                    "replicate": 0,
                    "selection_seed": -1,
                    "token_indices": indices,
                    "token_count": len(indices),
                    "replacement": REPLACEMENT,
                    "matching_fields": [],
                })

    keys = {(row["decision_id"], row["intervention_id"]) for row in records}
    _require(len(keys) == len(records), "Phase 9 intervention IDs are not unique")
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
        "selection_k": k,
        "matched_random_seeds": list(seeds),
        "norm_quantiles": norm_quantiles,
        "matching_norm_quantiles_by_selector": matching_quantiles_by_method,
        "matching_fallback": "global per-selector norm quantiles coarsen from requested to 2 then 1; bird-box stratum remains exact",
        "replacement": REPLACEMENT,
        "fixed_mask_across_stages": True,
        "include_low_evidence": include_low_evidence,
        "include_global_control": include_global_control,
        "decision_count": len(reference_decisions),
        "image_count": len({str(value["image_id"]) for value in identities.values()}),
        "intervention_count": len(records),
        "records": records,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _clustered_group_difference(
    rows: Sequence[tuple[str, int, float]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap a failure-minus-success contrast by image cluster."""

    by_image: dict[str, dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    for image_id, group, value in rows:
        by_image[image_id][group].append(value)
    image_values = {
        image_id: {group: statistics.mean(values) for group, values in groups.items() if values}
        for image_id, groups in by_image.items()
    }

    def contrast(sampled: Sequence[str]) -> float | None:
        values = {0: [], 1: []}
        for image_id in sampled:
            for group, value in image_values[image_id].items():
                values[group].append(value)
        if not values[0] or not values[1]:
            return None
        return statistics.mean(values[1]) - statistics.mean(values[0])

    keys = sorted(image_values)
    estimate = contrast(keys)
    _require(estimate is not None, "failure and success groups must both be present")
    rng = random.Random(seed)
    draws: list[float] = []
    attempts = 0
    while len(draws) < samples and attempts < samples * 20:
        attempts += 1
        value = contrast([keys[rng.randrange(len(keys))] for _ in keys])
        if value is not None:
            draws.append(value)
    _require(len(draws) == samples, "could not bootstrap both outcome groups")
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": estimate,
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "n_images": len(keys),
        "n_rows": len(rows),
        "cluster_unit": "image_id",
        "contrast_direction": "phase5_failure_minus_phase5_success",
    }


def _interval_label(interval: Mapping[str, Any], positive: str, negative: str) -> str:
    low, high = float(interval["ci_low"]), float(interval["ci_high"])
    if low <= 0 <= high:
        return "null_compatible"
    return positive if low > 0 else negative


def aggregate_selector_interventions(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    bootstrap_samples: int = 10000,
    seed: int = 20260913,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Aggregate selector specificity, stage interaction, and failure interaction."""

    _require(plan.get("purpose") == PURPOSE_PLAN and plan.get("status") == "PASS",
             "a passing Phase 9 plan is required")
    _require(plan.get("development_only") is True and plan.get("official_test_images_used") == 0,
             "Phase 9 analysis accepts development-only plans")
    records = plan.get("records")
    _require(isinstance(records, list) and records, "Phase 9 plan records are missing")
    index = {(str(row["decision_id"]), str(row["intervention_id"])): row for row in records}
    _require(len(index) == len(records), "Phase 9 plan contains duplicate identities")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    baselines: dict[tuple[str, str], float] = {}
    for source in outcomes:
        key = (_identifier(source.get("decision_id"), "decision_id"),
               _identifier(source.get("intervention_id"), "intervention_id"))
        _require(key in index and key not in seen, "outcome identity is absent or duplicated")
        seen.add(key)
        planned = index[key]
        for field in ("image_id", "stage", "selection_method", "intervention_type"):
            if field in source:
                _require(str(source[field]) == str(planned[field]),
                         f"outcome contradicts planned {field}")
        before = _finite(source.get("correct_answer_teacher_forced_margin_before"), "margin before")
        after = _finite(source.get("correct_answer_teacher_forced_margin_after"), "margin after")
        base_key = (str(planned["decision_id"]), str(planned["stage"]))
        previous = baselines.setdefault(base_key, before)
        _require(math.isclose(previous, before, rel_tol=0, abs_tol=1e-9),
                 f"baseline margin changes within {base_key}")
        normalized.append({**planned, **source, "damage": before - after})
    _require(seen == set(index), "outcomes do not cover every planned intervention")

    by_design: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        if row["selection_method"] != GLOBAL_METHOD:
            by_design[(str(row["decision_id"]), str(row["stage"]),
                       str(row["selection_method"]))].append(row)

    selector_values: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    decision_effects: dict[tuple[str, str, str], dict[str, Any]] = {}
    for design, rows in sorted(by_design.items()):
        top = [row for row in rows if row["intervention_type"] == "top_evidence"]
        random_rows = [row for row in rows if row["intervention_type"] == "matched_random"]
        _require(len(top) == 1 and random_rows, f"incomplete top/random design: {design}")
        effect = float(top[0]["damage"]) - statistics.mean(float(row["damage"]) for row in random_rows)
        value = {
            "decision_id": design[0],
            "stage": design[1],
            "selection_method": design[2],
            "image_id": str(top[0]["image_id"]),
            "phase5_failure": int(top[0]["phase5_failure"]),
            "effect": effect,
        }
        selector_values[(design[1], design[2])].append(value)
        decision_effects[design] = value

    selector_contrasts: list[dict[str, Any]] = []
    for (stage, method), values in sorted(selector_values.items()):
        interval = clustered_paired_bootstrap(
            [row["image_id"] for row in values], [row["effect"] for row in values],
            samples=bootstrap_samples,
            seed=_stable_seed(seed, "selector", stage, method),
            confidence_level=confidence_level,
        )
        selector_contrasts.append({
            "stage": stage,
            "selection_method": method,
            "metric": PRIMARY_METRIC,
            "contrast_direction": "top_evidence_drop_minus_matched_random_drop",
            **interval,
            "interpretation": _interval_label(
                interval, "selector_specific_causal_effect", "top_less_damaging_than_random"
            ),
        })

    methods = list(plan["selector_methods"])
    stages = (str(plan["selected_stage"]), str(plan["neighbor_stage"]))
    stage_interactions: list[dict[str, Any]] = []
    for method in methods:
        values = []
        for decision_id in sorted({key[0] for key in decision_effects}):
            selected_key = (decision_id, stages[0], method)
            neighbor_key = (decision_id, stages[1], method)
            _require(selected_key in decision_effects and neighbor_key in decision_effects,
                     f"stage pair is incomplete for {decision_id}/{method}")
            left, right = decision_effects[selected_key], decision_effects[neighbor_key]
            _require(left["image_id"] == right["image_id"], "stage pair image differs")
            values.append((left["image_id"], left["effect"] - right["effect"]))
        interval = clustered_paired_bootstrap(
            [item[0] for item in values], [item[1] for item in values],
            samples=bootstrap_samples,
            seed=_stable_seed(seed, "stage", method), confidence_level=confidence_level,
        )
        stage_interactions.append({
            "selection_method": method,
            "metric": PRIMARY_METRIC,
            "contrast_direction": f"{stages[0]}_minus_{stages[1]}_selector_specific_effect",
            **interval,
            "interpretation": _interval_label(
                interval, "greater_selected_stage_susceptibility", "greater_neighbor_stage_susceptibility"
            ),
        })

    selector_interactions: list[dict[str, Any]] = []
    for stage in stages:
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1:]:
                values = []
                for decision_id in sorted({key[0] for key in decision_effects}):
                    left = decision_effects[(decision_id, stage, left_method)]
                    right = decision_effects[(decision_id, stage, right_method)]
                    values.append((left["image_id"], left["effect"] - right["effect"]))
                interval = clustered_paired_bootstrap(
                    [item[0] for item in values], [item[1] for item in values],
                    samples=bootstrap_samples,
                    seed=_stable_seed(seed, "selector_interaction", stage, left_method, right_method),
                    confidence_level=confidence_level,
                )
                selector_interactions.append({
                    "stage": stage,
                    "left_selector": left_method,
                    "right_selector": right_method,
                    "metric": PRIMARY_METRIC,
                    "contrast_direction": f"{left_method}_minus_{right_method}_selector_specific_effect",
                    **interval,
                    "interpretation": _interval_label(
                        interval, "left_selector_more_causal", "right_selector_more_causal"
                    ),
                })

    outcome_interactions: list[dict[str, Any]] = []
    for (stage, method), values in sorted(selector_values.items()):
        interval = _clustered_group_difference(
            [(row["image_id"], row["phase5_failure"], row["effect"]) for row in values],
            samples=bootstrap_samples,
            seed=_stable_seed(seed, "failure", stage, method),
            confidence_level=confidence_level,
        )
        outcome_interactions.append({
            "stage": stage,
            "selection_method": method,
            "metric": PRIMARY_METRIC,
            **interval,
            "interpretation": _interval_label(
                interval, "larger_effect_on_failures", "larger_effect_on_successes"
            ),
        })

    global_summaries: list[dict[str, Any]] = []
    for stage in stages:
        values = [row for row in normalized
                  if row["stage"] == stage and row["intervention_type"] == GLOBAL_TYPE]
        if not values:
            continue
        interval = clustered_paired_bootstrap(
            [row["image_id"] for row in values], [row["damage"] for row in values],
            samples=bootstrap_samples,
            seed=_stable_seed(seed, "global", stage), confidence_level=confidence_level,
        )
        global_summaries.append({
            "stage": stage,
            "metric": PRIMARY_METRIC,
            **interval,
            "interpretation": _interval_label(
                interval, "intervention_path_changes_margin", "global_replacement_improves_margin"
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
        "damage_direction": "before_minus_after; positive means the correct-answer margin fell",
        "bootstrap_unit": "image_id",
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "decision_count": len({row["decision_id"] for row in normalized}),
        "image_count": len({str(row["image_id"]) for row in normalized}),
        "outcome_count": len(normalized),
        "selector_specific_contrasts": selector_contrasts,
        "stage_interactions": stage_interactions,
        "selector_interactions": selector_interactions,
        "failure_success_interactions": outcome_interactions,
        "global_manipulation_checks": global_summaries,
        "claim_boundary": (
            "Development-only mechanism diagnostics. Positive selector-specific effects show "
            "causal importance under this replacement intervention; null intervals do not prove "
            "absence, and stage interactions measure susceptibility rather than literal information loss."
        ),
    }
