"""Model-agnostic planning and aggregation for Phase 7 interventions.

This module deliberately does not load a model. It turns token-score metadata
into a frozen mean-replacement plan and summarizes precomputed outcomes. The
primary effect is a *drop* in the teacher-forced correct-answer margin, so a
positive value means that an intervention damaged the correct answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch


SCHEMA_VERSION = 1
INTERVENTION_TYPES = ("top_evidence", "matched_random", "low_evidence")
REPLACEMENT = "development_train_stage_mean"
PRIMARY_METRIC = "correct_answer_teacher_forced_margin_drop"
SECONDARY_METRICS = (
    "generated_correctness_drop",
    "unrelated_attribute_margin_drop",
)


class Phase7ValidationError(RuntimeError):
    """Raised when an intervention design or outcome table is not auditable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase7ValidationError(message)


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Phase7ValidationError(f"{field} must be numeric") from error
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _nonempty_id(value: Any, field: str) -> str:
    _require(isinstance(value, (str, int)) and str(value).strip() != "", f"{field} is required")
    return str(value)


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _canonical_stratum(value: Any) -> str:
    _require(value is not None, "bird-box stratum cannot be null")
    if isinstance(value, str):
        normalized = value.strip()
        _require(bool(normalized), "bird-box stratum cannot be empty")
        return normalized
    if isinstance(value, bool):
        return "inside" if value else "outside"
    _require(isinstance(value, (int, float)) and math.isfinite(float(value)),
             "bird-box stratum must be a JSON scalar")
    return str(value)


def _metadata_mode(
    rows: Sequence[Mapping[str, Any]],
    primary: str,
    aliases: Sequence[str] = (),
) -> str | None:
    primary_present = [primary in row for row in rows]
    if all(primary_present):
        return primary
    alias_presence = {alias: [alias in row for row in rows] for alias in aliases}
    complete_aliases = [alias for alias, present in alias_presence.items() if all(present)]
    _require(len(complete_aliases) <= 1,
             f"multiple aliases supplied for {primary}; use one canonical metadata field")
    if complete_aliases:
        _require(not any(primary_present),
                 f"{primary} and its alias are mixed; use one canonical metadata field")
        return complete_aliases[0]
    if not any(primary_present) and not any(any(present) for present in alias_presence.values()):
        return None
    raise Phase7ValidationError(
        f"{primary} metadata is partially or inconsistently present; refusing unmatched fallback"
    )


def _norm_quantile_bins(rows: Sequence[Mapping[str, Any]], field: str, bins: int) -> dict[int, int]:
    _require(bins >= 1, "norm_quantiles must be positive")
    ordered = sorted(
        ((_finite_float(row[field], field), int(row["token_index"])) for row in rows),
        key=lambda pair: (pair[0], pair[1]),
    )
    _require(all(value >= 0 for value, _ in ordered), "hidden_state_norm cannot be negative")
    count = len(ordered)
    return {token_index: min(bins - 1, rank * bins // count)
            for rank, (_, token_index) in enumerate(ordered)}


def _selection_id(
    decision_id: str,
    stage: str,
    intervention_type: str,
    replicate: int,
    token_indices: Sequence[int],
) -> str:
    payload = json.dumps(
        [decision_id, stage, intervention_type, replicate, list(token_indices)],
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"p7-{intervention_type}-{suffix}"


def validate_stage_pair(selected_stage: str, neighbor_stage: str, stage_order: Sequence[str]) -> None:
    """Require two distinct adjacent stages in an explicit frozen ordering."""

    order = list(stage_order)
    _require(order and len(order) == len(set(order)), "stage_order must be nonempty and unique")
    _require(selected_stage in order and neighbor_stage in order, "selected and neighbor stages must be in stage_order")
    _require(abs(order.index(selected_stage) - order.index(neighbor_stage)) == 1,
             "neighbor_stage must be adjacent to selected_stage")


def _normalize_token_group(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in rows:
        token_index = source.get("token_index")
        _require(isinstance(token_index, int) and not isinstance(token_index, bool) and token_index >= 0,
                 "token_index must be a nonnegative integer")
        _require(token_index not in seen, "token_index must be unique within decision and stage")
        seen.add(token_index)
        normalized.append({**source, "token_index": token_index,
                           "evidence_score": _finite_float(source.get("evidence_score"), "evidence_score")})
    return normalized


def _matched_random_indices(
    rows: Sequence[Mapping[str, Any]],
    top: Sequence[int],
    low: Sequence[int],
    *,
    norm_quantiles: int,
    seed: int,
) -> tuple[list[int], list[str]]:
    spatial_field = _metadata_mode(
        rows, "bird_box_membership", ("bird_box_stratum", "inside_bird_box")
    )
    norm_field = _metadata_mode(rows, "hidden_state_norm")
    norm_bins = _norm_quantile_bins(rows, norm_field, norm_quantiles) if norm_field else {}
    matching_fields: list[str] = []
    if spatial_field:
        matching_fields.append("bird_box_membership")
    if norm_field:
        matching_fields.append("hidden_state_norm_quantile")

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        parts: list[str] = []
        if spatial_field:
            parts.append(_canonical_stratum(row[spatial_field]))
        if norm_field:
            parts.append(str(norm_bins[int(row["token_index"])]))
        return tuple(parts)

    by_index = {int(row["token_index"]): row for row in rows}
    requested: dict[tuple[str, ...], int] = defaultdict(int)
    for token_index in top:
        requested[key(by_index[token_index])] += 1
    excluded = set(top) | set(low)
    candidates: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row in rows:
        token_index = int(row["token_index"])
        if token_index not in excluded:
            candidates[key(row)].append(token_index)
    chosen: list[int] = []
    rng = random.Random(seed)
    for stratum in sorted(requested):
        available = sorted(candidates.get(stratum, []))
        needed = requested[stratum]
        _require(len(available) >= needed,
                 f"insufficient matched-random candidates for stratum {stratum}: need {needed}, have {len(available)}")
        rng.shuffle(available)
        chosen.extend(available[:needed])
    _require(len(chosen) == len(top) and len(set(chosen)) == len(chosen),
             "matched-random selection is malformed")
    return sorted(chosen), matching_fields


def build_intervention_plan(
    token_rows: Iterable[Mapping[str, Any]],
    *,
    selected_stage: str,
    neighbor_stage: str,
    stage_order: Sequence[str],
    k: int,
    seed: int = 0,
    random_replicates: int = 3,
    random_seeds: Sequence[int] | None = None,
    norm_quantiles: int = 4,
) -> dict[str, Any]:
    """Build a deterministic top/low/matched-random mean-replacement plan.

    Required token fields are ``decision_id``, ``image_id``, ``stage``,
    ``token_index``, and ``evidence_score``. Exact matching additionally uses
    ``bird_box_membership`` (or a documented alias) and ``hidden_state_norm`` when
    each is present for every token in a decision-stage group. Partially present
    metadata is rejected instead of silently ignored.
    """

    _require(isinstance(k, int) and not isinstance(k, bool) and k > 0, "k must be positive")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")
    _require(isinstance(random_replicates, int) and not isinstance(random_replicates, bool)
             and random_replicates > 0,
             "random_replicates must be positive")
    _require(isinstance(norm_quantiles, int) and not isinstance(norm_quantiles, bool)
             and norm_quantiles > 0,
             "norm_quantiles must be a positive integer")
    if random_seeds is None:
        matched_seeds = tuple(range(random_replicates))
    else:
        matched_seeds = tuple(random_seeds)
        _require(bool(matched_seeds) and all(isinstance(item, int) and not isinstance(item, bool)
                                             and item >= 0 for item in matched_seeds),
                 "random_seeds must contain nonnegative integers")
        _require(len(matched_seeds) == len(set(matched_seeds)), "random_seeds must be unique")
        _require(random_replicates == len(matched_seeds),
                 "random_replicates must equal the number of random_seeds")
    validate_stage_pair(selected_stage, neighbor_stage, stage_order)
    stages = (selected_stage, neighbor_stage)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    decision_images: dict[str, Any] = {}
    decision_image_keys: dict[str, str] = {}
    for row in token_rows:
        _require(isinstance(row, Mapping), "token rows must be mappings")
        decision_id = _nonempty_id(row.get("decision_id"), "decision_id")
        image_value = row.get("image_id")
        image_id = _nonempty_id(image_value, "image_id")
        stage = _nonempty_id(row.get("stage"), "stage")
        _require(stage in stages, f"unexpected token metadata stage: {stage}")
        previous = decision_image_keys.setdefault(decision_id, image_id)
        _require(previous == image_id, f"decision {decision_id} maps to multiple images")
        decision_images.setdefault(decision_id, image_value)
        grouped[(decision_id, stage)].append(row)
    _require(bool(grouped), "token metadata is empty")
    decisions = sorted(decision_images)
    for decision_id in decisions:
        _require(all((decision_id, stage) in grouped for stage in stages),
                 f"decision {decision_id} does not cover selected stage and neighbor")

    records: list[dict[str, Any]] = []
    for decision_id in decisions:
        for stage in stages:
            rows = _normalize_token_group(grouped[(decision_id, stage)])
            _require(len(rows) >= 3 * k,
                     f"decision {decision_id} stage {stage} needs at least 3*k tokens")
            descending = sorted(rows, key=lambda row: (-float(row["evidence_score"]), int(row["token_index"])))
            ascending = sorted(rows, key=lambda row: (float(row["evidence_score"]), int(row["token_index"])))
            top = [int(row["token_index"]) for row in descending[:k]]
            top_set = set(top)
            low = [int(row["token_index"]) for row in ascending if int(row["token_index"]) not in top_set][:k]
            _require(len(low) == k and not (set(top) & set(low)), "top and low evidence selections overlap")
            role = "selected" if stage == selected_stage else "neighbor"

            def add(intervention_type: str, indices: Sequence[int], replicate: int, selection_seed: int) -> None:
                intervention_id = _selection_id(decision_id, stage, intervention_type, replicate, indices)
                records.append({
                    "decision_id": decision_id,
                    "image_id": decision_images[decision_id],
                    "intervention_id": intervention_id,
                    "stage": stage,
                    "stage_role": role,
                    "intervention_type": intervention_type,
                    "replicate": replicate,
                    "selection_seed": selection_seed,
                    "token_indices": list(indices),
                    "token_count": len(indices),
                    "replacement": REPLACEMENT,
                })

            add("top_evidence", top, 0, -1)
            add("low_evidence", low, 0, -1)
            matching_fields: list[str] | None = None
            for replicate, selection_seed in enumerate(matched_seeds):
                rng_seed = _stable_seed(seed, decision_id, stage, selection_seed)
                matched, fields = _matched_random_indices(
                    rows, top, low, norm_quantiles=norm_quantiles, seed=rng_seed
                )
                matching_fields = fields
                add("matched_random", matched, replicate, selection_seed)
                records[-1]["matching_fields"] = fields
            for record in records[-(len(matched_seeds) + 2):]:
                record.setdefault("matching_fields", matching_fields or [])

    ids = {(record["decision_id"], record["intervention_id"]) for record in records}
    _require(len(ids) == len(records), "intervention IDs are not unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "phase7_targeted_causal_intervention_plan",
        "selected_stage": selected_stage,
        "neighbor_stage": neighbor_stage,
        "stage_order": list(stage_order),
        "k": k,
        "seed": seed,
        "random_replicates": len(matched_seeds),
        "matched_random_seeds": list(matched_seeds),
        "norm_quantiles": norm_quantiles,
        "replacement": REPLACEMENT,
        "matching_policy": "exact joint bird-box-membership/norm-quantile strata when fully available",
        "matching_fallback": "absent metadata only; partial metadata or insufficient strata fail closed",
        "low_evidence_policy": "bottom-k by evidence score with token-index tie break",
        "decision_count": len(decisions),
        "intervention_count": len(records),
        "records": records,
    }


def compute_train_stage_means(rows: Iterable[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Compute one replacement vector per stage from training-token records only."""

    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        _require(row.get("development_split") == "train",
                 "train-stage means must be computed from training rows only")
        stage = _nonempty_id(row.get("stage"), "stage")
        vector = torch.as_tensor(row.get("hidden_state"))
        _require(vector.ndim == 1 and vector.numel() > 0, "hidden_state must be a nonempty vector")
        _require(vector.dtype.is_floating_point and bool(torch.isfinite(vector).all()),
                 "hidden_state must be finite floating point")
        grouped[stage].append(vector.detach().cpu().to(torch.float64))
    _require(bool(grouped), "no training hidden states were provided")
    result: dict[str, torch.Tensor] = {}
    for stage, vectors in grouped.items():
        shape = vectors[0].shape
        _require(all(vector.shape == shape for vector in vectors),
                 f"hidden-state shape differs within stage {stage}")
        result[stage] = torch.stack(vectors).mean(dim=0).to(torch.float32)
    return result


def replace_tokens_with_stage_mean(
    hidden_states: torch.Tensor,
    token_indices: Sequence[int],
    train_stage_mean: torch.Tensor,
    *,
    token_dim: int = -2,
) -> torch.Tensor:
    """Return a clone with selected token vectors replaced by a training mean."""

    _require(isinstance(hidden_states, torch.Tensor) and hidden_states.ndim >= 2,
             "hidden_states must be a tensor with token and hidden dimensions")
    _require(hidden_states.dtype.is_floating_point and bool(torch.isfinite(hidden_states).all()),
             "hidden_states must be finite floating point")
    dimension = token_dim + hidden_states.ndim if token_dim < 0 else token_dim
    _require(0 <= dimension < hidden_states.ndim - 1,
             "token_dim must precede at least one representation dimension")
    indices = list(token_indices)
    _require(indices and all(isinstance(index, int) and not isinstance(index, bool) for index in indices),
             "token_indices must be nonempty integers")
    _require(len(indices) == len(set(indices)), "token_indices must be unique")
    _require(min(indices) >= 0 and max(indices) < hidden_states.shape[dimension],
             "token index is out of range")
    mean = torch.as_tensor(train_stage_mean, device=hidden_states.device, dtype=hidden_states.dtype)
    expected = hidden_states.shape[dimension + 1:]
    _require(tuple(mean.shape) == tuple(expected),
             f"train_stage_mean shape must be {tuple(expected)}")
    _require(bool(torch.isfinite(mean).all()), "train_stage_mean must be finite")
    output = hidden_states.clone()
    selector: list[Any] = [slice(None)] * output.ndim
    selector[dimension] = torch.tensor(indices, device=output.device, dtype=torch.long)
    target = output[tuple(selector)]
    view_shape = (1,) * (target.ndim - mean.ndim) + tuple(mean.shape)
    output[tuple(selector)] = mean.reshape(view_shape).expand_as(target)
    return output


def apply_planned_intervention(
    hidden_states_by_stage: Mapping[str, torch.Tensor],
    train_stage_means: Mapping[str, torch.Tensor],
    intervention: Mapping[str, Any],
    *,
    token_dim: int = -2,
) -> dict[str, torch.Tensor]:
    """Apply one plan record without mutating the caller's stage mapping."""

    _require(intervention.get("replacement") == REPLACEMENT,
             f"only {REPLACEMENT} replacement is supported")
    stage = _nonempty_id(intervention.get("stage"), "stage")
    _require(stage in hidden_states_by_stage and stage in train_stage_means,
             f"missing hidden states or training mean for {stage}")
    output = dict(hidden_states_by_stage)
    output[stage] = replace_tokens_with_stage_mean(
        hidden_states_by_stage[stage], intervention.get("token_indices", ()),
        train_stage_means[stage], token_dim=token_dim,
    )
    return output


def _bool01(value: Any, field: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int) and value in (0, 1):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "0", "1"):
        return float(value.strip().lower() in ("true", "1"))
    raise Phase7ValidationError(f"{field} must be boolean or 0/1")


def _margin_collection(value: Any, field: str) -> dict[str, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise Phase7ValidationError(f"{field} must contain valid JSON") from error
    if isinstance(value, Mapping):
        _require(bool(value), f"{field} cannot be empty")
        return {str(key): _finite_float(item, field) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        _require(bool(value), f"{field} cannot be empty")
        return {str(index): _finite_float(item, field) for index, item in enumerate(value)}
    return {"0": _finite_float(value, field)}


def _optional_pair(row: Mapping[str, Any], before: str, after: str) -> bool:
    left = before in row and row.get(before) not in (None, "")
    right = after in row and row.get(after) not in (None, "")
    _require(left == right, f"{before} and {after} must be supplied together")
    return left


def _select_optional_pair(
    row: Mapping[str, Any],
    pairs: Sequence[tuple[str, str]],
    label: str,
) -> tuple[str, str] | None:
    present = [pair for pair in pairs if _optional_pair(row, *pair)]
    _require(len(present) <= 1, f"supply at most one {label} field pair")
    return present[0] if present else None


def _join_plan(
    outcomes: list[dict[str, Any]],
    plan: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if plan is None:
        return outcomes
    if isinstance(plan, Mapping):
        _require(plan.get("schema_version") == SCHEMA_VERSION,
                 "unsupported intervention plan schema_version")
        _require(plan.get("purpose") == "phase7_targeted_causal_intervention_plan",
                 "unexpected intervention plan purpose")
        _require(plan.get("replacement") == REPLACEMENT,
                 f"intervention plan must use {REPLACEMENT} replacement")
    records = plan.get("records") if isinstance(plan, Mapping) else plan
    _require(isinstance(records, Sequence), "plan must contain a records sequence")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (_nonempty_id(record.get("decision_id"), "decision_id"),
               _nonempty_id(record.get("intervention_id"), "intervention_id"))
        _require(key not in index, "duplicate decision_id/intervention_id in plan")
        index[key] = record
    seen: set[tuple[str, str]] = set()
    joined: list[dict[str, Any]] = []
    for outcome in outcomes:
        key = (_nonempty_id(outcome.get("decision_id"), "decision_id"),
               _nonempty_id(outcome.get("intervention_id"), "intervention_id"))
        _require(key in index, f"outcome is absent from plan: {key}")
        source = index[key]
        merged = dict(source)
        for field, value in outcome.items():
            if field in merged and field in ("image_id", "stage", "stage_role", "intervention_type"):
                _require(str(merged[field]) == str(value), f"outcome contradicts plan field {field}")
            merged[field] = value
        joined.append(merged)
        seen.add(key)
    _require(seen == set(index), "outcomes do not cover every planned intervention")
    return joined


def normalize_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate outcome identity, paired fields, and unchanged baselines."""

    copied = [dict(row) for row in rows]
    _require(bool(copied), "intervention outcomes are empty")
    joined = _join_plan(copied, plan)
    normalized: list[dict[str, Any]] = []
    ids: set[tuple[str, str]] = set()
    decisions: dict[str, str] = {}
    optional_presence: dict[str, set[bool]] = defaultdict(set)
    for row in joined:
        decision_id = _nonempty_id(row.get("decision_id"), "decision_id")
        intervention_id = _nonempty_id(row.get("intervention_id"), "intervention_id")
        image_id = _nonempty_id(row.get("image_id"), "image_id")
        stage = _nonempty_id(row.get("stage"), "stage")
        stage_role = _nonempty_id(row.get("stage_role"), "stage_role")
        _require(stage_role in ("selected", "neighbor"), "stage_role must be selected or neighbor")
        intervention_type = _nonempty_id(row.get("intervention_type"), "intervention_type")
        _require(intervention_type in INTERVENTION_TYPES, "unknown intervention_type")
        key = (decision_id, intervention_id)
        _require(key not in ids, "duplicate decision_id/intervention_id outcome")
        ids.add(key)
        previous = decisions.setdefault(decision_id, image_id)
        _require(previous == image_id, f"decision {decision_id} maps to multiple images")
        margin_pairs = (
            ("correct_answer_teacher_forced_margin_before",
             "correct_answer_teacher_forced_margin_after"),
            ("teacher_forced_correct_answer_margin_before",
             "teacher_forced_correct_answer_margin_after"),
            ("correct_answer_margin_before", "correct_answer_margin_after"),
        )
        margin_pair = _select_optional_pair(row, margin_pairs, "teacher-forced correct-answer margin")
        _require(margin_pair is not None,
                 "supply exactly one teacher-forced correct-answer margin field pair")
        before_field, after_field = margin_pair
        before = _finite_float(row.get(before_field), before_field)
        after = _finite_float(row.get(after_field), after_field)
        metrics = {PRIMARY_METRIC: before - after}
        generated_pair = _select_optional_pair(row, (
            ("generated_correct_before", "generated_correct_after"),
            ("generation_correct_before", "generation_correct_after"),
            ("generated_binary_accuracy_before", "generated_binary_accuracy_after"),
        ), "generated correctness")
        optional_presence["generated"].add(generated_pair is not None)
        generated_before: float | None = None
        if generated_pair is not None:
            generated_before = _bool01(row[generated_pair[0]], generated_pair[0])
            metrics[SECONDARY_METRICS[0]] = (
                generated_before - _bool01(row[generated_pair[1]], generated_pair[1])
            )
        unrelated_pair = _select_optional_pair(row, (
            ("unrelated_attribute_margins_before", "unrelated_attribute_margins_after"),
            ("unrelated_attribute_margin_before", "unrelated_attribute_margin_after"),
        ), "unrelated-attribute margin")
        optional_presence["unrelated"].add(unrelated_pair is not None)
        unrelated_before: dict[str, float] | None = None
        if unrelated_pair is not None:
            unrelated_before = _margin_collection(
                row[unrelated_pair[0]], unrelated_pair[0]
            )
            unrelated_after = _margin_collection(
                row[unrelated_pair[1]], unrelated_pair[1]
            )
            _require(set(unrelated_before) == set(unrelated_after),
                     "unrelated attribute identities differ before and after")
            metrics[SECONDARY_METRICS[1]] = statistics.mean(
                unrelated_before[name] - unrelated_after[name] for name in sorted(unrelated_before)
            )
        normalized.append({
            **row,
            "decision_id": decision_id,
            "intervention_id": intervention_id,
            "image_id": image_id,
            "stage": stage,
            "stage_role": stage_role,
            "intervention_type": intervention_type,
            "metrics": metrics,
            "_baseline": (
                before,
                generated_before,
                unrelated_before,
            ),
        })
    for name, states in optional_presence.items():
        _require(len(states) == 1, f"{name} secondary outcomes are only partially present")

    baselines: dict[tuple[str, str], tuple[Any, ...]] = {}
    for row in normalized:
        key = (row["decision_id"], row["stage"])
        previous = baselines.setdefault(key, row["_baseline"])
        _require(previous == row["_baseline"],
                 f"baseline outcomes differ across interventions for decision-stage {key}")
    return normalized


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    _require(bool(sorted_values), "cannot take percentile of empty values")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def clustered_paired_bootstrap(
    image_ids: Sequence[Any],
    paired_values: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 314159,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Equal-image-weighted paired estimate and cluster bootstrap interval."""

    _require(len(image_ids) == len(paired_values) and len(image_ids) > 0,
             "bootstrap inputs must have the same nonzero length")
    _require(isinstance(samples, int) and samples > 0, "bootstrap samples must be positive")
    _require(0 < confidence_level < 1, "confidence_level must lie in (0,1)")
    by_image: dict[str, list[float]] = defaultdict(list)
    for image_id, value in zip(image_ids, paired_values):
        by_image[_nonempty_id(image_id, "image_id")].append(_finite_float(value, "paired value"))
    image_means = {image_id: statistics.mean(values) for image_id, values in by_image.items()}
    keys = sorted(image_means)
    estimate = statistics.mean(image_means.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(statistics.mean(image_means[keys[rng.randrange(len(keys))]] for _ in keys))
    draws.sort()
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": estimate,
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "n_images": len(keys),
        "n_paired_rows": len(paired_values),
        "cluster_unit": "image_id",
        "image_weighting": "equal",
    }


def _interpret(interval: Mapping[str, Any], *, contrast: bool) -> str:
    low, high = float(interval["ci_low"]), float(interval["ci_high"])
    if low <= 0 <= high:
        return "null_compatible"
    if int(interval["n_images"]) < 2:
        return "insufficient_image_clusters"
    if low > 0:
        return "top_more_damaging" if contrast else "positive_drop"
    if high < 0:
        return "top_less_damaging" if contrast else "negative_drop"
    raise AssertionError("unreachable confidence-interval interpretation")


def aggregate_intervention_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    bootstrap_samples: int = 10000,
    seed: int = 314159,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Aggregate paired intervention damage and top-versus-control contrasts."""

    normalized = normalize_outcomes(rows, plan=plan)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    designs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[(row["stage"], row["stage_role"], row["intervention_type"])].append(row)
        designs[(row["decision_id"], row["stage"])].append(row)
    for key, values in designs.items():
        counts = {name: sum(row["intervention_type"] == name for row in values)
                  for name in INTERVENTION_TYPES}
        _require(counts["top_evidence"] == 1 and counts["low_evidence"] == 1
                 and counts["matched_random"] >= 1,
                 f"incomplete intervention design for decision-stage {key}: {counts}")
    roles_by_decision: dict[str, dict[str, str]] = defaultdict(dict)
    global_stage_roles: dict[str, str] = {}
    for row in normalized:
        previous = global_stage_roles.setdefault(row["stage"], row["stage_role"])
        _require(previous == row["stage_role"], f"stage role changes across decisions: {row['stage']}")
        roles_by_decision[row["decision_id"]][row["stage_role"]] = row["stage"]
    _require(set(global_stage_roles.values()) == {"selected", "neighbor"}
             and len(global_stage_roles) == 2,
             "outcomes must contain exactly one selected stage and one neighbor stage")
    for decision_id, roles in roles_by_decision.items():
        _require(set(roles) == {"selected", "neighbor"},
                 f"decision {decision_id} does not cover selected and neighbor stages")

    metric_names = sorted({name for row in normalized for name in row["metrics"]})
    summaries: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items()):
        for metric in metric_names:
            applicable = [row for row in values if metric in row["metrics"]]
            if not applicable:
                continue
            interval = clustered_paired_bootstrap(
                [row["image_id"] for row in applicable],
                [row["metrics"][metric] for row in applicable],
                samples=bootstrap_samples,
                seed=_stable_seed(seed, "summary", *group_key, metric),
                confidence_level=confidence_level,
            )
            summaries.append({
                "stage": group_key[0],
                "stage_role": group_key[1],
                "intervention_type": group_key[2],
                "metric": metric,
                **interval,
                "interpretation": _interpret(interval, contrast=False),
            })

    contrast_values: dict[tuple[str, str, str, str], list[tuple[str, float]]] = defaultdict(list)
    for (decision_id, stage), values in sorted(designs.items()):
        role = values[0]["stage_role"]
        image_id = values[0]["image_id"]
        _require(all(row["stage_role"] == role and row["image_id"] == image_id for row in values),
                 f"inconsistent design identity for {(decision_id, stage)}")
        top = next(row for row in values if row["intervention_type"] == "top_evidence")
        for control in ("matched_random", "low_evidence"):
            controls = [row for row in values if row["intervention_type"] == control]
            for metric in metric_names:
                if metric not in top["metrics"]:
                    continue
                _require(all(metric in row["metrics"] for row in controls),
                         f"control metric {metric} is partially missing for {(decision_id, stage)}")
                control_mean = statistics.mean(row["metrics"][metric] for row in controls)
                contrast_values[(stage, role, control, metric)].append(
                    (image_id, top["metrics"][metric] - control_mean)
                )

    contrasts: list[dict[str, Any]] = []
    for group_key, values in sorted(contrast_values.items()):
        interval = clustered_paired_bootstrap(
            [image_id for image_id, _ in values], [value for _, value in values],
            samples=bootstrap_samples,
            seed=_stable_seed(seed, "contrast", *group_key),
            confidence_level=confidence_level,
        )
        contrasts.append({
            "stage": group_key[0],
            "stage_role": group_key[1],
            "reference": group_key[2],
            "metric": group_key[3],
            **interval,
            "interpretation": _interpret(interval, contrast=True),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "phase7_targeted_causal_intervention_aggregation",
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": [name for name in SECONDARY_METRICS if name in metric_names],
        "delta_direction": "before_minus_after; positive means intervention damage",
        "contrast_direction": "top_evidence_drop_minus_control_drop",
        "bootstrap_unit": "image_id",
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "seed": seed,
        "decision_count": len({row["decision_id"] for row in normalized}),
        "image_count": len({row["image_id"] for row in normalized}),
        "outcome_count": len(normalized),
        "null_result_valid": True,
        "intervention_summaries": summaries,
        "paired_contrasts": contrasts,
    }
