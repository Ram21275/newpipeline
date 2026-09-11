"""Phase 6 cross-phase joining, clustered inference, and transition selection.

The module deliberately consumes CSV artifacts rather than model checkpoints.  It
keeps the three measurements distinct: Phase 3R linear accessibility, Phase 4
localization, and Phase 5 answer behaviour.  All cross-phase comparisons are
made on the same ``image_id::attribute_id::prompt_id`` decision key.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


DECISION_ID_SCHEMA = "image_id::attribute_id::prompt_id"
ALLOWED_TRANSITIONS = (
    "visual_to_language_bottleneck",
    "representation_utilization_gap",
    "discriminative_semantic_localization_mismatch",
    "evidence_redistribution",
    "mixed_or_null",
)
PRIMARY_AGGREGATION = "fixed_attribute_macro"
ROBUSTNESS_AGGREGATIONS = ("micro", PRIMARY_AGGREGATION, "two_way_micro")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class Phase6Config:
    """Frozen analysis choices for the joint development analysis."""

    prompt_id: str = "cub_attribute_yes_no_v1"
    split: str = "val"
    phase3_pooling: str = "mean"
    phase3_control: str = "primary"
    vision_stage: str = "vision.late"
    language_stage: str = "llm.final"
    phase4_k: int = 32
    discriminative_selector: str = "vision_cls_attention"
    semantic_selector: str = "logit_concept"
    localization_metric: str = "part_patch_recall"
    phase5_control: str = "image"
    phase5_outcome: str = "margin_correct"
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20260911
    confidence_level: float = 0.95
    minimum_effect: float = 0.05

    def validate(self) -> None:
        require(self.prompt_id.strip() == self.prompt_id and self.prompt_id,
                "prompt_id must be non-empty and trimmed")
        require("::" not in self.prompt_id, "prompt_id cannot contain '::'")
        require(self.split in ("train", "val"), "Phase 6 is development-only")
        require(self.vision_stage != self.language_stage,
                "vision and language stages must differ")
        require(self.discriminative_selector != self.semantic_selector,
                "localization selectors must differ")
        require(self.phase5_outcome in ("margin_correct", "generation_correct"),
                "phase5_outcome must be margin_correct or generation_correct")
        require(self.phase4_k > 0, "phase4_k must be positive")
        require(self.bootstrap_resamples >= 100,
                "at least 100 bootstrap resamples are required")
        require(0.5 < self.confidence_level < 1.0,
                "confidence_level must lie between 0.5 and 1")
        require(0.0 <= self.minimum_effect <= 1.0,
                "minimum_effect must lie in [0, 1]")


def read_csv(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"CSV has no data rows: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]],
               fields: Sequence[str] | None = None) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    if fields is None:
        fields = tuple(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: object, field: str) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid integer {field}={value!r}") from error
    return number


def _as_float(value: object, field: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid numeric {field}={value!r}") from error
    require(math.isfinite(number), f"Non-finite {field}={value!r}")
    return number


def _as_binary(value: object, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes"):
        return 1
    if text in ("0", "false", "no"):
        return 0
    raise RuntimeError(f"Invalid binary {field}={value!r}")


def decision_id(image_id: int, attribute_id: int, prompt_id: str) -> str:
    require(image_id >= 0 and attribute_id > 0, "Invalid decision identity")
    require(bool(prompt_id) and "::" not in prompt_id,
            "Invalid prompt identity")
    return f"{image_id}::{attribute_id}::{prompt_id}"


def _normalize_identity(row: Mapping[str, object], config: Phase6Config,
                        source: str) -> dict[str, object]:
    require("image_id" in row and "attribute_id" in row,
            f"{source} lacks image_id or attribute_id")
    image_id = _as_int(row["image_id"], "image_id")
    attribute_id = _as_int(row["attribute_id"], "attribute_id")
    prompt = str(row.get("prompt_id") or config.prompt_id)
    expected = decision_id(image_id, attribute_id, prompt)
    recorded = str(row.get("decision_id") or expected)
    require(recorded == expected,
            f"{source} decision_id does not match its component columns: {recorded}")
    require(prompt == config.prompt_id,
            f"{source} prompt_id differs from the frozen Phase 6 prompt")
    split = str(row.get("split", ""))
    require(split in ("train", "val"), f"{source} has invalid split {split!r}")
    result = dict(row)
    result.update(decision_id=recorded, image_id=image_id,
                  attribute_id=attribute_id, prompt_id=prompt, split=split)
    return result


def _zero_official_test_guard(rows: Sequence[Mapping[str, object]], source: str) -> None:
    """Reject any test evidence before analysis filters can hide it."""

    for row in rows:
        split_values = (row.get("split"), row.get("official_split"),
                        row.get("dataset_split"))
        require(all(str(value).strip().lower() != "test" for value in split_values
                    if value not in (None, "")),
                f"Official-test row encountered in {source}")
        for field in ("official_test_images_used", "official_test_images"):
            if row.get(field) not in (None, ""):
                require(_as_int(row[field], field) == 0,
                        f"Nonzero official-test use in {source}")


def _metadata_signature(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row.get(field, "") for field in
                 ("image_id", "attribute_id", "prompt_id", "split",
                  "attribute_name", "attribute_group", "target"))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    require(bool(values), "Cannot average an empty sequence")
    return statistics.fmean(values)


def _aggregate_phase3(rows: Sequence[Mapping[str, object]], config: Phase6Config
                      ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    normalized = [_normalize_identity(row, config, "Phase 3R") for row in rows]
    selected = [row for row in normalized
                if row["split"] == config.split
                and str(row.get("pooling")) == config.phase3_pooling
                and str(row.get("control")) == config.phase3_control
                and str(row.get("stage")) in (config.vision_stage, config.language_stage)
                and _as_binary(row.get("observed", 1), "observed")]
    require(bool(selected), "No Phase 3R rows match the frozen analysis choices")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["decision_id"]), str(row["stage"]))].append(row)
    by_decision: dict[str, dict[str, object]] = defaultdict(dict)
    for (key, stage), group in grouped.items():
        signatures = {_metadata_signature(row) for row in group}
        require(len(signatures) == 1, f"Phase 3R metadata differs within {key}/{stage}")
        seeds = [str(row.get("seed", "")) for row in group]
        require(len(seeds) == len(set(seeds)),
                f"Duplicate Phase 3R seed for {key}/{stage}")
        correct = [_as_binary(row["probe_correct"], "probe_correct") for row in group]
        margins = [_as_float(row["probe_margin"], "probe_margin") for row in group
                   if row.get("probe_margin") not in (None, "")]
        base = by_decision[key]
        if not base:
            base.update({field: group[0].get(field, "") for field in
                         ("decision_id", "image_id", "attribute_id", "prompt_id",
                          "split", "attribute_name", "attribute_group", "target")})
        prefix = "vision" if stage == config.vision_stage else "language"
        require(f"phase3_{prefix}_correct" not in base,
                f"Phase 3R cardinality is not one row per decision/stage: {key}")
        base[f"phase3_{prefix}_correct"] = _mean(correct)
        base[f"phase3_{prefix}_margin"] = _mean(margins) if margins else ""
        base[f"phase3_{prefix}_seeds"] = len(group)
    complete = {key: value for key, value in by_decision.items()
                if "phase3_vision_correct" in value and "phase3_language_correct" in value}
    require(bool(complete), "No Phase 3R decision has both frozen stages")
    prompts = sorted({str(row.get("representation_prompt")) for row in selected
                      if row.get("representation_prompt") not in (None, "")})
    alignments = sorted({str(row.get("prompt_context_alignment")) for row in selected
                         if row.get("prompt_context_alignment") not in (None, "")})
    require(len(prompts) <= 1 and len(alignments) <= 1,
            "Phase 3R prompt-context provenance changes across rows")
    return complete, {
        "input_rows": len(rows), "selected_rows": len(selected),
        "candidate_decisions": len(by_decision), "complete_decisions": len(complete),
        "incomplete_decisions": len(by_decision) - len(complete),
        "representation_prompt": prompts[0] if prompts else "unspecified",
        "prompt_context_alignment": alignments[0] if alignments else "unspecified",
        "language_states_question_conditioned": (
            alignments == ["question_conditioned_vqa_prompt"]
            if alignments else None
        ),
    }


def _json_integer_sequence(value: object, field: str) -> list[int]:
    try:
        decoded = json.loads(str(value)) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON sequence in {field}") from error
    require(isinstance(decoded, list) and decoded, f"{field} must be a non-empty list")
    values = [_as_int(item, field) for item in decoded]
    require(len(values) == len(set(values)), f"{field} contains duplicate indices")
    return values


def _aggregate_phase3_patch(
    rows: Sequence[Mapping[str, object]], config: Phase6Config
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Aggregate direct patch-level probe concentration and persistence."""

    normalized = [_normalize_identity(row, config, "Phase 3R patch evidence") for row in rows]
    selected = [
        row
        for row in normalized
        if row["split"] == config.split
        and str(row.get("pooling")) == config.phase3_pooling
        and str(row.get("control")) == config.phase3_control
        and str(row.get("stage")) in (config.vision_stage, config.language_stage)
    ]
    require(bool(selected), "No Phase 3R patch rows match the frozen analysis choices")
    topk_field = f"top{config.phase4_k}_indices"
    grouped: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in selected:
        key = (str(row["decision_id"]), str(row["stage"]), _as_int(row["seed"], "seed"))
        require(key not in grouped, f"Duplicate Phase 3R patch row: {key}")
        require(topk_field in row, f"Phase 3R patch evidence lacks {topk_field}")
        grouped[key] = row
    decision_stage: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for (key, stage, _), row in grouped.items():
        decision_stage[(key, stage)].append(row)
    result: dict[str, dict[str, object]] = defaultdict(dict)
    for (key, stage), group in decision_stage.items():
        prefix = "vision" if stage == config.vision_stage else "language"
        result[key][f"phase3_{prefix}_evidence_entropy"] = _mean(
            _as_float(row["evidence_entropy"], "evidence_entropy") for row in group
        )
        result[key][f"phase3_{prefix}_evidence_effective_tokens"] = _mean(
            _as_float(row["evidence_effective_tokens"], "evidence_effective_tokens")
            for row in group
        )
        result[key][f"phase3_{prefix}_topk_by_seed"] = {
            _as_int(row["seed"], "seed"): set(_json_integer_sequence(row[topk_field], topk_field))
            for row in group
        }
    complete: dict[str, dict[str, object]] = {}
    for key, values in result.items():
        vision = values.get("phase3_vision_topk_by_seed")
        language = values.get("phase3_language_topk_by_seed")
        if not isinstance(vision, dict) or not isinstance(language, dict):
            continue
        require(set(vision) == set(language), f"Patch-evidence seeds differ for {key}")
        overlaps = [
            len(vision[seed] & language[seed]) / len(vision[seed] | language[seed])
            for seed in sorted(vision)
        ]
        cleaned = {
            field: value for field, value in values.items() if not field.endswith("_topk_by_seed")
        }
        cleaned["phase3_topk_jaccard_vision_language"] = _mean(overlaps)
        cleaned["phase3_spatial_redistribution"] = 1.0 - _mean(overlaps)
        cleaned["phase3_zero"] = 0.0
        complete[key] = cleaned
    require(bool(complete), "No decision has both vision and language patch evidence")
    return complete, {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "complete_decisions": len(complete),
        "topk": config.phase4_k,
    }


def _aggregate_phase4(rows: Sequence[Mapping[str, object]], config: Phase6Config
                      ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    normalized = [_normalize_identity(row, config, "Phase 4") for row in rows]
    selectors = (config.discriminative_selector, config.semantic_selector)
    selected = [row for row in normalized
                if row["split"] == config.split
                and _as_int(row.get("K"), "K") == config.phase4_k
                and str(row.get("selector", row.get("method", ""))) in selectors]
    require(bool(selected), "No Phase 4 rows match the frozen analysis choices")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        selector = str(row.get("selector", row.get("method", "")))
        grouped[(str(row["decision_id"]), selector)].append(row)
    by_decision: dict[str, dict[str, object]] = defaultdict(dict)
    for (key, selector), group in grouped.items():
        signatures = {_metadata_signature(row) for row in group}
        require(len(signatures) == 1, f"Phase 4 metadata differs within {key}/{selector}")
        seeds = [str(row.get("selection_seed", "")) for row in group]
        require(len(seeds) == len(set(seeds)),
                f"Duplicate Phase 4 selection seed for {key}/{selector}")
        require(all(config.localization_metric in row for row in group),
                f"Phase 4 lacks metric {config.localization_metric}")
        scores = [_as_float(row[config.localization_metric], config.localization_metric)
                  for row in group]
        base = by_decision[key]
        if not base:
            base.update({field: group[0].get(field, "") for field in
                         ("decision_id", "image_id", "attribute_id", "prompt_id",
                          "split", "attribute_name", "attribute_group", "target")})
        prefix = ("discriminative" if selector == config.discriminative_selector
                  else "semantic")
        require(f"phase4_{prefix}_localization" not in base,
                f"Phase 4 cardinality is not one row per decision/selector: {key}")
        base[f"phase4_{prefix}_localization"] = _mean(scores)
        base[f"phase4_{prefix}_seeds"] = len(group)
    complete = {key: value for key, value in by_decision.items()
                if "phase4_discriminative_localization" in value
                and "phase4_semantic_localization" in value}
    require(bool(complete), "No Phase 4 decision has both frozen selectors")
    return complete, {
        "input_rows": len(rows), "selected_rows": len(selected),
        "candidate_decisions": len(by_decision), "complete_decisions": len(complete),
        "incomplete_decisions": len(by_decision) - len(complete),
    }


def _aggregate_phase5(rows: Sequence[Mapping[str, object]], config: Phase6Config
                      ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    normalized = [_normalize_identity(row, config, "Phase 5") for row in rows]
    selected = [row for row in normalized
                if row["split"] == config.split
                and str(row.get("control")) == config.phase5_control]
    require(bool(selected), "No Phase 5 rows match the frozen analysis choices")
    result: dict[str, dict[str, object]] = {}
    for row in selected:
        key = str(row["decision_id"])
        require(key not in result,
                f"Phase 5 cardinality is not one row per decision/control: {key}")
        target = _as_binary(row["target"], "target")
        require(config.phase5_outcome in row,
                f"Phase 5 lacks frozen outcome {config.phase5_outcome}")
        correct = _as_binary(row[config.phase5_outcome], config.phase5_outcome)
        value = {field: row.get(field, "") for field in
                 ("decision_id", "image_id", "attribute_id", "prompt_id", "split",
                  "attribute_name", "attribute_group")}
        value.update(target=target, phase5_answer_correct=correct,
                     phase5_failure=1 - correct)
        if row.get("signed_answer_margin") not in (None, ""):
            value["phase5_signed_answer_margin"] = _as_float(
                row["signed_answer_margin"], "signed_answer_margin")
        result[key] = value
    return result, {
        "input_rows": len(rows), "selected_rows": len(selected),
        "complete_decisions": len(result),
    }


def join_phase_rows(phase3_rows: Sequence[Mapping[str, object]],
                    phase4_rows: Sequence[Mapping[str, object]],
                    phase5_rows: Sequence[Mapping[str, object]],
                    config: Phase6Config,
                    phase3_patch_rows: Sequence[Mapping[str, object]] | None = None,
                    ) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate, aggregate, and inner-join all phases with an explicit audit."""

    config.validate()
    _zero_official_test_guard(phase3_rows, "Phase 3R")
    _zero_official_test_guard(phase4_rows, "Phase 4")
    _zero_official_test_guard(phase5_rows, "Phase 5")
    p3, audit3 = _aggregate_phase3(phase3_rows, config)
    audit_patch: dict[str, object] | None = None
    if phase3_patch_rows is not None:
        _zero_official_test_guard(phase3_patch_rows, "Phase 3R patch evidence")
        patch, audit_patch = _aggregate_phase3_patch(phase3_patch_rows, config)
        require(set(p3) == set(patch), "Phase 3R score and patch-evidence decisions differ")
        for key in p3:
            p3[key].update(patch[key])
    p4, audit4 = _aggregate_phase4(phase4_rows, config)
    p5, audit5 = _aggregate_phase5(phase5_rows, config)
    sets = {"phase3r": set(p3), "phase4": set(p4), "phase5": set(p5)}
    attribute_sets = {
        "phase3r": {int(row["attribute_id"]) for row in p3.values()},
        "phase4": {int(row["attribute_id"]) for row in p4.values()},
        "phase5": {int(row["attribute_id"]) for row in p5.values()},
    }
    require(len({tuple(sorted(values)) for values in attribute_sets.values()}) == 1,
            "Cross-phase fixed attribute sets differ")
    common = sets["phase3r"] & sets["phase4"] & sets["phase5"]
    require(bool(common), "No decision_id is shared by Phase 3R, Phase 4, and Phase 5")
    joined: list[dict[str, object]] = []
    for key in sorted(common, key=lambda value: tuple(
            int(part) if index < 2 else part
            for index, part in enumerate(value.split("::", 2)))):
        rows = (p3[key], p4[key], p5[key])
        identities = {(int(row["image_id"]), int(row["attribute_id"]),
                       str(row["prompt_id"]), str(row["split"])) for row in rows}
        require(len(identities) == 1, f"Cross-phase identity differs for {key}")
        targets = {str(row.get("target")) for row in rows
                   if row.get("target") not in (None, "")}
        require(len(targets) <= 1, f"Cross-phase target differs for {key}")
        names = {str(row.get("attribute_name")) for row in rows
                 if row.get("attribute_name") not in (None, "")}
        require(len(names) <= 1, f"Cross-phase attribute name differs for {key}")
        groups = {str(row.get("attribute_group")) for row in rows
                  if row.get("attribute_group") not in (None, "")}
        require(len(groups) <= 1, f"Cross-phase attribute group differs for {key}")
        merged: dict[str, object] = {}
        for row in rows:
            merged.update({field: value for field, value in row.items()
                           if value not in (None, "")})
        joined.append(merged)
    failures = sum(int(row["phase5_failure"]) for row in joined)
    require(0 < failures < len(joined),
            "Joint cohort must contain both Phase 5 failures and successes")
    audit = {
        "decision_id_schema": DECISION_ID_SCHEMA,
        "phase3r": audit3, "phase4": audit4, "phase5": audit5,
        "phase3r_patch": audit_patch,
        "joined_decisions": len(joined),
        "joined_images": len({row["image_id"] for row in joined}),
        "joined_attributes": len({row["attribute_id"] for row in joined}),
        "fixed_attribute_ids": sorted(attribute_sets["phase5"]),
        "failures": failures, "successes": len(joined) - failures,
        "unmatched": {name: len(keys - common) for name, keys in sets.items()},
        "official_test_images_used": 0,
    }
    return joined, audit


def _quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "Cannot take a quantile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _weighted_mean(pairs: Iterable[tuple[float, float]]) -> float | None:
    numerator = denominator = 0.0
    for value, weight in pairs:
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator > 0 else None


def _paired_estimate(rows: Sequence[Mapping[str, object]], left: str, right: str,
                     aggregation: str, image_weights: Mapping[int, int] | None = None,
                     attribute_weights: Mapping[int, int] | None = None) -> float | None:
    values = []
    for row in rows:
        if row.get(left) in (None, "") or row.get(right) in (None, ""):
            continue
        image_id, attribute_id = int(row["image_id"]), int(row["attribute_id"])
        weight = (image_weights.get(image_id, 0) if image_weights is not None else 1)
        if attribute_weights is not None:
            weight *= attribute_weights.get(attribute_id, 0)
        if weight:
            values.append((attribute_id, _as_float(row[left], left)
                           - _as_float(row[right], right), float(weight)))
    if aggregation in ("micro", "two_way_micro"):
        return _weighted_mean((value, weight) for _, value, weight in values)
    require(aggregation == PRIMARY_AGGREGATION, f"Unknown aggregation {aggregation}")
    by_attribute: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for attribute_id, value, weight in values:
        by_attribute[attribute_id].append((value, weight))
    estimates = [_weighted_mean(pairs) for pairs in by_attribute.values()]
    defined = [value for value in estimates if value is not None]
    return statistics.fmean(defined) if defined else None


def matched_failure_success_rows(rows: Sequence[Mapping[str, object]],
                                 metrics: Sequence[str]) -> list[dict[str, object]]:
    """Return exact attribute/target-stratum failure-minus-success contrasts."""

    output: list[dict[str, object]] = []
    for metric in metrics:
        strata: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        images: dict[tuple[int, int], dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set))
        for row in rows:
            if row.get(metric) in (None, ""):
                continue
            key = (int(row["attribute_id"]), int(row["target"]))
            failure = int(row["phase5_failure"])
            strata[key][failure].append(_as_float(row[metric], metric))
            images[key][failure].add(int(row["image_id"]))
        for (attribute_id, target), groups in sorted(strata.items()):
            if not groups[0] or not groups[1]:
                continue
            output.append({
                "metric": metric, "attribute_id": attribute_id, "target": target,
                "failure_decisions": len(groups[1]), "success_decisions": len(groups[0]),
                "failure_images": len(images[(attribute_id, target)][1]),
                "success_images": len(images[(attribute_id, target)][0]),
                "failure_mean": _mean(groups[1]), "success_mean": _mean(groups[0]),
                "failure_minus_success": _mean(groups[1]) - _mean(groups[0]),
                "matched_support": min(len(groups[1]), len(groups[0])),
            })
    require(bool(output), "No attribute/target stratum contains both failures and successes")
    return output


def _failure_estimate(rows: Sequence[Mapping[str, object]], metric: str,
                      aggregation: str, image_weights: Mapping[int, int] | None = None,
                      attribute_weights: Mapping[int, int] | None = None) -> float | None:
    strata: dict[tuple[int, int], dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        if row.get(metric) in (None, ""):
            continue
        image_id, attribute_id = int(row["image_id"]), int(row["attribute_id"])
        weight = image_weights.get(image_id, 0) if image_weights is not None else 1
        if attribute_weights is not None:
            weight *= attribute_weights.get(attribute_id, 0)
        if weight:
            key = (attribute_id, int(row["target"]))
            strata[key][int(row["phase5_failure"])].append(
                (_as_float(row[metric], metric), float(weight)))
    effects: list[tuple[int, float, float]] = []
    for (attribute_id, _), outcomes in strata.items():
        failure = _weighted_mean(outcomes[1])
        success = _weighted_mean(outcomes[0])
        if failure is None or success is None:
            continue
        matched_weight = min(sum(weight for _, weight in outcomes[1]),
                             sum(weight for _, weight in outcomes[0]))
        effects.append((attribute_id, failure - success, matched_weight))
    if aggregation in ("micro", "two_way_micro"):
        return _weighted_mean((effect, weight) for _, effect, weight in effects)
    require(aggregation == PRIMARY_AGGREGATION, f"Unknown aggregation {aggregation}")
    by_attribute: dict[int, list[float]] = defaultdict(list)
    for attribute_id, effect, _ in effects:
        by_attribute[attribute_id].append(effect)
    return (statistics.fmean(statistics.fmean(values) for values in by_attribute.values())
            if by_attribute else None)


def _bootstrap_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:8], "big")


def _cluster_counts(values: Sequence[int], rng: random.Random) -> Counter[int]:
    return Counter(rng.choice(values) for _ in values)


def bootstrap_contrast(rows: Sequence[Mapping[str, object]], *, name: str,
                       kind: str, left: str | None = None, right: str | None = None,
                       metric: str | None = None, config: Phase6Config,
                       allow_insufficient: bool = False,
                       ) -> list[dict[str, object]]:
    """Image-clustered inference for paired or matched failure contrasts."""

    require(kind in ("paired", "failure_success"), f"Unknown contrast kind {kind}")
    images = sorted({int(row["image_id"]) for row in rows})
    attributes = sorted({int(row["attribute_id"]) for row in rows})
    require(len(images) >= 2 and attributes, "Bootstrap requires at least two image clusters")
    if kind == "paired":
        require(bool(left and right), "Paired contrast needs left and right columns")
        estimator: Callable[..., float | None] = lambda aggregation, iw=None, aw=None: (
            _paired_estimate(rows, left, right, aggregation, iw, aw))
    else:
        require(bool(metric), "Failure/success contrast needs a metric")
        estimator = lambda aggregation, iw=None, aw=None: (
            _failure_estimate(rows, metric, aggregation, iw, aw))
    alpha = (1.0 - config.confidence_level) / 2.0
    output = []
    for aggregation in ROBUSTNESS_AGGREGATIONS:
        estimate_aggregation = "micro" if aggregation == "two_way_micro" else aggregation
        estimate = estimator(estimate_aggregation)
        require(estimate is not None, f"Contrast {name}/{aggregation} has no estimate")
        rng = random.Random(_bootstrap_seed(config.bootstrap_seed, name, aggregation))
        samples: list[float] = []
        for _ in range(config.bootstrap_resamples):
            image_weights = _cluster_counts(images, rng)
            attribute_weights = (_cluster_counts(attributes, rng)
                                 if aggregation == "two_way_micro" else None)
            value = estimator(estimate_aggregation, image_weights, attribute_weights)
            if value is not None:
                samples.append(value)
        required_samples = max(20, config.bootstrap_resamples // 2)
        if len(samples) < required_samples and not allow_insufficient:
            raise RuntimeError(f"Too few valid bootstrap replicates for {name}/{aggregation}")
        output.append({
            "contrast": name, "kind": kind, "aggregation": aggregation,
            "estimate": estimate,
            "ci_low": _quantile(samples, alpha) if len(samples) >= required_samples else "",
            "ci_high": (_quantile(samples, 1.0 - alpha)
                        if len(samples) >= required_samples else ""),
            "inference_status": ("complete" if len(samples) >= required_samples
                                 else "insufficient_bootstrap_support"),
            "confidence_level": config.confidence_level,
            "bootstrap_resamples_requested": config.bootstrap_resamples,
            "bootstrap_resamples_valid": len(samples),
            "bootstrap_seed": config.bootstrap_seed,
            "cluster_unit": "image_id",
            "attribute_resampling": aggregation == "two_way_micro",
            "decisions": len(rows), "images": len(images),
            "attributes": len(attributes),
        })
    return output


def apply_transition_rubric(inference: Sequence[Mapping[str, object]],
                            config: Phase6Config,
                            *,
                            language_prompt_aligned: bool = True) -> dict[str, object]:
    """Apply a conservative, predeclared rubric without ranking weak signals."""

    primary = {str(row["contrast"]): row for row in inference
               if row["aggregation"] == PRIMARY_AGGREGATION and row["kind"] == "paired"}
    required = {
        "visual_to_language_bottleneck": "vision_minus_language_probe",
        "representation_utilization_gap": "language_probe_minus_answer",
        "discriminative_semantic_localization_mismatch":
            "discriminative_minus_semantic_localization",
    }
    require(set(required.values()) <= primary.keys(), "Missing primary rubric contrast")
    evidence: dict[str, dict[str, object]] = {}
    for label, contrast in required.items():
        row = primary[contrast]
        estimate = float(row["estimate"])
        low, high = float(row["ci_low"]), float(row["ci_high"])
        if label == "discriminative_semantic_localization_mismatch":
            supported = abs(estimate) >= config.minimum_effect and (low > 0 or high < 0)
            criterion = "absolute effect >= minimum and 95% CI excludes zero"
        else:
            supported = estimate >= config.minimum_effect and low > 0
            criterion = "positive effect >= minimum and 95% CI lower bound > 0"
        evidence[label] = {
            "supported": supported, "contrast": contrast, "estimate": estimate,
            "ci_low": low, "ci_high": high, "criterion": criterion,
        }
    if not language_prompt_aligned:
        for label in ("visual_to_language_bottleneck", "representation_utilization_gap"):
            evidence[label]["diagnostic_supported_before_context_gate"] = evidence[label]["supported"]
            evidence[label]["supported"] = False
            evidence[label]["context_gate"] = (
                "blocked because Phase 3R LLM states use the neutral Phase 2 prompt, "
                "not the Phase 5 VQA question"
            )
    redistribution = primary.get("spatial_evidence_redistribution")
    entropy_shift = primary.get("language_minus_vision_evidence_entropy")
    if redistribution is None or entropy_shift is None:
        evidence["evidence_redistribution"] = {
            "supported": False,
            "criterion": "requires direct aligned patch-map persistence and entropy contrasts",
            "reason": "Phase 3R patch evidence was not supplied",
        }
    else:
        moved = (
            float(redistribution["estimate"]) >= 0.5
            and float(redistribution["ci_low"]) > 0.5
        )
        entropy = float(entropy_shift["estimate"])
        entropy_low = float(entropy_shift["ci_low"])
        entropy_high = float(entropy_shift["ci_high"])
        concentration_changed = (
            abs(entropy) >= config.minimum_effect
            and (entropy_low > 0 or entropy_high < 0)
        )
        evidence["evidence_redistribution"] = {
            "supported": moved and concentration_changed,
            "criterion": (
                "Top-K redistribution >= 0.5 with CI lower bound > 0.5 and "
                "absolute entropy shift >= minimum with CI excluding zero"
            ),
            "redistribution_estimate": float(redistribution["estimate"]),
            "redistribution_ci_low": float(redistribution["ci_low"]),
            "entropy_shift": entropy,
            "entropy_ci_low": entropy_low,
            "entropy_ci_high": entropy_high,
        }
    supported = [label for label in ALLOWED_TRANSITIONS[:-1]
                 if evidence[label]["supported"]]
    selected = supported[0] if len(supported) == 1 else "mixed_or_null"
    require(selected in ALLOWED_TRANSITIONS, "Rubric emitted an unknown transition")
    return {
        "selected_transition": selected,
        "supported_positive_labels": supported,
        "minimum_effect": config.minimum_effect,
        "primary_aggregation": PRIMARY_AGGREGATION,
        "multiple_positive_policy": "mixed_or_null",
        "no_positive_policy": "mixed_or_null",
        "evidence": evidence,
        "interpretation_boundary": (
            "The selected label prioritizes a Phase 7 intervention target; it is not causal proof."
        ),
        "language_prompt_aligned": language_prompt_aligned,
    }


def _format_findings(audit: Mapping[str, object], inference: Sequence[Mapping[str, object]],
                     rubric: Mapping[str, object], config: Phase6Config) -> str:
    primary = [row for row in inference
               if row["kind"] == "paired" and row["aggregation"] == PRIMARY_AGGREGATION]
    lines = [
        "# Intermediate findings", "",
        "## Scope and gate", "",
        (f"This development-only analysis joined {audit['joined_decisions']} decisions from "
         f"{audit['joined_images']} images and {audit['joined_attributes']} fixed attributes. "
         "Official test images used: 0."), "",
        "## Paired cross-phase evidence", "",
        "| Contrast | Fixed-attribute macro effect | 95% clustered bootstrap CI |",
        "|---|---:|---:|",
    ]
    for row in primary:
        lines.append(f"| {row['contrast']} | {float(row['estimate']):.4f} | "
                     f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |")
    lines.extend([
        "", "Bootstrap clusters are images; micro, fixed-attribute macro, and two-way "
        "image/attribute resampling results are retained in `inference_summary.csv`.", "",
        "## Failure-versus-success comparison", "",
        (f"The joined cohort contains {audit['failures']} Phase 5 failures and "
         f"{audit['successes']} successes. Comparisons are matched within attribute and "
         "target strata; unsupported strata are not imputed."), "",
        "## Predeclared transition decision", "",
        f"**{rubric['selected_transition']}**", "",
    ])
    positives = rubric["supported_positive_labels"]
    if positives:
        lines.append("Rubric-positive diagnostics: " + ", ".join(positives) + ".")
    else:
        lines.append("No positive diagnostic satisfied both the minimum-effect and CI rule.")
    if rubric["selected_transition"] == "mixed_or_null" and len(positives) > 1:
        lines.append("Multiple diagnostics passed, so the rubric does not force a single transition.")
    lines.extend([
        "", "## Alternatives and limitations", "",
        "- Linear-probe correctness measures accessibility, not use.",
        "- Localization is diagnostic and neither attention nor dense similarity is causal evidence.",
        "- Failure/success contrasts can reveal association, not intervention effects.",
        "- Evidence redistribution is assessed only when aligned Phase 3R patch maps are supplied.",
    ])
    if rubric.get("language_prompt_aligned") is False:
        lines.append(
            "- Vision/projector states are image-conditioned, but cached Phase 3R LLM states use "
            "the neutral Phase 2 prompt rather than the Phase 5 attribute question; language-side "
            "bottleneck/utilization labels are therefore blocked by the context gate."
        )
    lines.extend([
        "- The development validation split has informed analysis choices and is not final held-out evidence.",
        "", "## Minimal next causal test", "",
        ("Run a matched top-evidence removal/replacement intervention at the transition named above, "
         "plus a neighboring stage and matched random-token controls. If the rubric selected "
         "`mixed_or_null`, do not choose an intervention until the ambiguity is resolved."), "",
    ])
    return "\n".join(lines)


def run_phase6_analysis(phase3_csv: Path | str, phase4_csv: Path | str,
                        phase5_csv: Path | str, output_dir: Path | str,
                        config: Phase6Config,
                        phase3_patch_csv: Path | str | None = None) -> dict[str, object]:
    """Run the complete local Phase 6 analysis and write auditable artifacts."""

    config.validate()
    paths = {"phase3r": Path(phase3_csv), "phase4": Path(phase4_csv),
             "phase5": Path(phase5_csv)}
    if phase3_patch_csv is not None:
        paths["phase3r_patch"] = Path(phase3_patch_csv)
    inputs = {name: read_csv(path) for name, path in paths.items()}
    joined, audit = join_phase_rows(
        inputs["phase3r"], inputs["phase4"], inputs["phase5"], config,
        inputs.get("phase3r_patch"),
    )
    paired_specs = [
        ("vision_minus_language_probe", "phase3_vision_correct", "phase3_language_correct"),
        ("language_probe_minus_answer", "phase3_language_correct", "phase5_answer_correct"),
        ("discriminative_minus_semantic_localization",
         "phase4_discriminative_localization", "phase4_semantic_localization"),
    ]
    if phase3_patch_csv is not None:
        paired_specs.extend(
            [
                (
                    "language_minus_vision_evidence_entropy",
                    "phase3_language_evidence_entropy",
                    "phase3_vision_evidence_entropy",
                ),
                (
                    "spatial_evidence_redistribution",
                    "phase3_spatial_redistribution",
                    "phase3_zero",
                ),
            ]
        )
    inference: list[dict[str, object]] = []
    for name, left, right in paired_specs:
        inference.extend(bootstrap_contrast(joined, name=name, kind="paired",
                                            left=left, right=right, config=config))
    failure_metrics = ["phase3_vision_correct", "phase3_language_correct",
                       "phase4_discriminative_localization",
                       "phase4_semantic_localization"]
    if phase3_patch_csv is not None:
        failure_metrics.extend(
            [
                "phase3_vision_evidence_entropy",
                "phase3_language_evidence_entropy",
                "phase3_topk_jaccard_vision_language",
            ]
        )
    if all("phase5_signed_answer_margin" in row for row in joined):
        failure_metrics.append("phase5_signed_answer_margin")
    matched = matched_failure_success_rows(joined, failure_metrics)
    for metric in failure_metrics:
        inference.extend(bootstrap_contrast(joined,
                                            name=f"failure_minus_success__{metric}",
                                            kind="failure_success", metric=metric,
                                            config=config, allow_insufficient=True))
    prompt_alignment = audit["phase3r"].get("language_states_question_conditioned")
    rubric = apply_transition_rubric(
        inference,
        config,
        language_prompt_aligned=(prompt_alignment is not False),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "joint_decisions.csv", joined)
    _write_csv(output / "matched_failure_success.csv", matched)
    _write_csv(output / "inference_summary.csv", inference)
    _write_json(output / "transition_decision.json", rubric)
    findings = _format_findings(audit, inference, rubric, config)
    temporary = output / "INTERMEDIATE_FINDINGS.md.tmp"
    temporary.write_text(findings, encoding="utf-8")
    temporary.replace(output / "INTERMEDIATE_FINDINGS.md")
    artifact_names = ["joint_decisions.csv", "matched_failure_success.csv",
                      "inference_summary.csv", "transition_decision.json",
                      "INTERMEDIATE_FINDINGS.md"]
    report = {
        "schema_version": 1, "status": "PASS",
        "purpose": "phase6_joint_trajectory_and_transition_selection",
        "decision_id_schema": DECISION_ID_SCHEMA,
        "config": asdict(config), "cardinality_audit": audit,
        "selected_transition": rubric["selected_transition"],
        "allowed_transitions": list(ALLOWED_TRANSITIONS),
        "official_test_images_used": 0,
        "input_artifacts": {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                            for name, path in paths.items()},
        "artifacts": artifact_names,
        "artifact_manifest": {
            name: {"bytes": (output / name).stat().st_size,
                   "sha256": _sha256(output / name)}
            for name in artifact_names
        },
    }
    _write_json(output / "phase6_run_report.json", report)
    return report
