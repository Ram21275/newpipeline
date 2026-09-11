"""Pure Phase 5 cohort, measurement, aggregation, and validation mechanics.

The model-facing adapter may run on Kaggle and serialize decision measurements.
This module deliberately depends only on the Python standard library so records
can be audited and unit-tested without a model, checkpoint, or GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import string
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
APPROVED_CERTAINTY_NAMES = frozenset({"probably", "definitely"})
CONTROLS = ("image", "prompt_only", "image_shuffled")
COHORTS = ("grounded_positive", "grounded_negative")
SUMMARY_FIELDS = (
    "mean_answer_margin",
    "mean_signed_answer_margin",
    "margin_accuracy",
    "generation_accuracy",
    "parse_rate",
)


class Phase5ValidationError(RuntimeError):
    """A fail-closed Phase 5 schema or scientific-protocol violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase5ValidationError(message)


def decision_id(image_id: int, attribute_id: int, prompt_id: str) -> str:
    """Return the frozen cross-artifact identity for one scientific decision."""

    image_id = _integer(image_id, "image_id")
    attribute_id = _integer(attribute_id, "attribute_id")
    require(image_id >= 0 and attribute_id > 0, "image/attribute IDs are out of range")
    require(isinstance(prompt_id, str) and prompt_id.strip() == prompt_id and prompt_id,
            "prompt_id must be a non-empty stripped string")
    require("::" not in prompt_id, "prompt_id cannot contain the decision separator")
    return f"{image_id}::{attribute_id}::{prompt_id}"


def render_binary_prompt(template: str, attribute_phrase: str) -> str:
    """Render a deterministic attribute question from one allowed placeholder."""

    require(isinstance(template, str) and template.strip() == template and template,
            "prompt template must be a non-empty stripped string")
    fields = [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]
    require(fields == ["attribute_phrase"],
            "prompt template must contain exactly one {attribute_phrase} placeholder")
    require(isinstance(attribute_phrase, str), "attribute_phrase must be text")
    phrase = " ".join(attribute_phrase.split())
    require(phrase and phrase == attribute_phrase,
            "attribute_phrase must be non-empty with canonical whitespace")
    prompt = template.format(attribute_phrase=phrase)
    require(prompt.strip() == prompt and "\n" not in prompt and "\r" not in prompt,
            "rendered prompt must be single-line canonical text")
    return prompt


def strict_parse_binary(
    generated_text: str,
    positive_answer: str = "yes",
    negative_answer: str = "no",
) -> bool | None:
    """Parse only a bare deterministic binary answer; reject punctuation/explanation."""

    if not isinstance(generated_text, str):
        return None
    require(_canonical_answer(positive_answer) != _canonical_answer(negative_answer),
            "binary answer candidates must differ")
    normalized = generated_text.strip().casefold()
    if normalized == _canonical_answer(positive_answer):
        return True
    if normalized == _canonical_answer(negative_answer):
        return False
    return None


def sequence_log_likelihood(token_log_probabilities: Sequence[float]) -> float:
    """Sum every teacher-forced continuation token, including multi-token answers."""

    require(not isinstance(token_log_probabilities, (str, bytes)),
            "token log probabilities must be a sequence")
    values = [float(value) for value in token_log_probabilities]
    require(bool(values), "answer candidates must contain at least one token")
    require(all(math.isfinite(value) and value <= 1e-7 for value in values),
            "token log probabilities must be finite and non-positive")
    return math.fsum(values)


def teacher_forced_margin(
    positive_token_log_probabilities: Sequence[float],
    negative_token_log_probabilities: Sequence[float],
) -> float:
    """Return log P(positive sequence) - log P(negative sequence)."""

    return (sequence_log_likelihood(positive_token_log_probabilities)
            - sequence_log_likelihood(negative_token_log_probabilities))


def make_grounded_decision(
    *,
    image_id: int,
    attribute_id: int,
    attribute_name: str,
    attribute_phrase: str,
    prompt_id: str,
    prompt_template: str,
    primary_target: bool | None,
    certainty_name: str | None,
    relevant_visible_in_crop_parts: int,
    split: str = "val",
    official_split: str = "train",
) -> dict[str, Any] | None:
    """Build an eligible decision, or mask an uncertain/ungrounded annotation.

    Both positive and negative decisions require a relevant visible in-crop part.
    A cached non-null target with any unapproved certainty is conservatively
    excluded rather than trusted.
    """

    require(primary_target in (True, False, None),
            "primary_target must be true, false, or null")
    certainty = str(certainty_name).strip().casefold() if certainty_name is not None else ""
    if certainty not in APPROVED_CERTAINTY_NAMES or primary_target is None:
        return None
    visible = _integer(relevant_visible_in_crop_parts, "relevant_visible_in_crop_parts")
    if visible < 1:
        return None
    require(isinstance(attribute_name, str) and attribute_name.strip() == attribute_name
            and attribute_name, "attribute_name must be canonical text")
    iid = _integer(image_id, "image_id")
    aid = _integer(attribute_id, "attribute_id")
    cohort = "grounded_positive" if primary_target else "grounded_negative"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_id(iid, aid, prompt_id),
        "image_id": iid,
        "attribute_id": aid,
        "attribute_name": attribute_name,
        "attribute_phrase": attribute_phrase,
        "prompt_id": prompt_id,
        "prompt_text": render_binary_prompt(prompt_template, attribute_phrase),
        "target": bool(primary_target),
        "cohort": cohort,
        "certainty_name": certainty,
        "relevant_visible_in_crop_parts": visible,
        "split": split,
        "official_split": official_split,
    }


def balance_grounded_decisions(
    decisions: Iterable[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Deterministically balance each attribute to its smaller grounded cohort."""

    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {cohort: [] for cohort in COHORTS}
    )
    for source in decisions:
        row = dict(source)
        cohort = row.get("cohort")
        require(cohort in COHORTS, "only grounded cohorts can be balanced")
        grouped[_integer(row.get("attribute_id"), "attribute_id")][cohort].append(row)
    require(bool(grouped), "no grounded decisions to balance")
    selected: list[dict[str, Any]] = []
    for aid, cohorts in sorted(grouped.items()):
        positives = cohorts["grounded_positive"]
        negatives = cohorts["grounded_negative"]
        require(bool(positives) and bool(negatives),
                f"attribute {aid} lacks one grounded label cohort")
        count = min(len(positives), len(negatives))
        for cohort, rows in (("positive", positives), ("negative", negatives)):
            ranked = sorted(
                rows,
                key=lambda row: _stable_key(
                    seed, aid, f"{cohort}:{row.get('decision_id', '')}"
                ),
            )
            selected.extend(ranked[:count])
    return sorted(selected, key=lambda row: str(row["decision_id"]))


def shuffled_image_map(
    decisions: Iterable[Mapping[str, Any]], seed: int
) -> dict[str, int]:
    """Create an attribute-stratified deterministic cyclic derangement."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[_integer(row.get("attribute_id"), "attribute_id")].append(row)
    result: dict[str, int] = {}
    for aid, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: _stable_key(seed, aid, str(row.get("decision_id", ""))),
        )
        ids = [_integer(row.get("image_id"), "image_id") for row in ordered]
        require(len(ids) >= 2 and len(ids) == len(set(ids)),
                f"attribute {aid} requires at least two distinct images for shuffling")
        offset = 1 + (int(seed) % (len(ids) - 1))
        rotated = ids[offset:] + ids[:offset]
        require(all(left != right for left, right in zip(ids, rotated)),
                f"attribute {aid} shuffle is not a derangement")
        for row, evaluated in zip(ordered, rotated):
            result[str(row["decision_id"])] = evaluated
    return result


def build_control_requests(
    decisions: Iterable[Mapping[str, Any]], shuffle_seed: int
) -> list[dict[str, Any]]:
    """Expand each scientific decision into matched image/control requests."""

    decisions = [dict(row) for row in decisions]
    shuffled = shuffled_image_map(decisions, shuffle_seed)
    result: list[dict[str, Any]] = []
    for row in sorted(decisions, key=lambda value: str(value["decision_id"])):
        for control in CONTROLS:
            expanded = dict(row)
            expanded["control"] = control
            expanded["evaluated_image_id"] = (
                row["image_id"] if control == "image" else
                None if control == "prompt_only" else
                shuffled[str(row["decision_id"])]
            )
            result.append(expanded)
    return result


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the frozen policy used to interpret precomputed rows."""

    cfg = dict(config)
    require(_integer(cfg.get("schema_version"), "schema_version") == SCHEMA_VERSION,
            "unsupported Phase 5 schema version")
    require(cfg.get("prompt_id") and isinstance(cfg["prompt_id"], str),
            "prompt_id is required")
    render_binary_prompt(str(cfg.get("prompt_template", "")), "canonical phrase")
    positive = str(cfg.get("positive_answer", ""))
    negative = str(cfg.get("negative_answer", ""))
    require(_canonical_answer(positive) == "yes" and _canonical_answer(negative) == "no",
            "Phase 5 binary candidates must be yes/no")
    require(tuple(cfg.get("controls", ())) == CONTROLS, "control policy differs")
    require(tuple(cfg.get("cohorts", ())) == COHORTS, "grounded cohort policy differs")
    require(cfg.get("require_visible_relevant_part_for_both_labels") is True,
            "both cohorts must require a visible relevant part")
    require(cfg.get("development_evaluation_split") == "val",
            "Phase 5 must evaluate the frozen development validation split")
    require(_integer(cfg.get("official_test_images_used"), "official_test_images_used") == 0,
            "official CUB test access is forbidden in Phase 5")
    require(cfg.get("primary_metric") == "teacher_forced_sequence_log_likelihood_margin",
            "unsupported primary metric")
    return cfg


def validate_phase5_records(
    records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fail closed, then return flat metrics, summaries, paired deltas, and gate."""

    cfg = validate_config(config)
    normalized = [_normalize_measurement(row, cfg) for row in records]
    require(bool(normalized), "no Phase 5 measurement records")
    seen: set[tuple[str, str]] = set()
    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        key = (row["decision_id"], row["control"])
        require(key not in seen, f"duplicate decision/control row: {key}")
        seen.add(key)
        by_decision[row["decision_id"]].append(row)
    require(all({row["control"] for row in rows} == set(CONTROLS)
                and len(rows) == len(CONTROLS) for rows in by_decision.values()),
            "every decision must have exactly one row for every control")

    identity_fields = (
        "image_id", "attribute_id", "attribute_name", "attribute_phrase",
        "prompt_id", "prompt_text", "target", "cohort", "certainty_name",
        "relevant_visible_in_crop_parts", "split", "official_split",
    )
    for did, rows in by_decision.items():
        base = tuple(rows[0][field] for field in identity_fields)
        require(all(tuple(row[field] for field in identity_fields) == base for row in rows[1:]),
                f"decision identity changes across controls: {did}")

    decision_rows = [next(row for row in rows if row["control"] == "image")
                     for rows in by_decision.values()]
    _validate_balanced_cohorts(decision_rows)
    _validate_shuffle(decision_rows, by_decision)

    summaries = summarize_utilization(normalized)
    deltas = paired_control_deltas(normalized)
    candidate_lengths = Counter()
    for row in normalized:
        candidate_lengths[f"positive_{len(json.loads(row['positive_token_ids']))}"] += 1
        candidate_lengths[f"negative_{len(json.loads(row['negative_token_ids']))}"] += 1
    cohort_counts = Counter(row["cohort"] for row in decision_rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_digest": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "measurement_rows": len(normalized),
        "decisions": len(by_decision),
        "attributes": len({row["attribute_id"] for row in decision_rows}),
        "controls": list(CONTROLS),
        "cohort_decisions": dict(sorted(cohort_counts.items())),
        "candidate_length_rows": dict(sorted(candidate_lengths.items())),
        "summary_rows": len(summaries),
        "paired_delta_rows": len(deltas),
        "approved_certainty_names": sorted(APPROVED_CERTAINTY_NAMES),
        "official_test_images_used": 0,
        "official_test_split_untouched": True,
        "exact_control_coverage": True,
        "grounded_cohorts_balanced_per_attribute": True,
        "image_shuffle_is_attribute_stratified_derangement": True,
        "all_generated_answers_strictly_parseable": all(
            bool(row["parse_success"]) for row in normalized
        ),
        "unparseable_generation_rows": sum(
            not bool(row["parse_success"]) for row in normalized
        ),
        "teacher_forced_sequence_likelihoods_validated": True,
        "interpretation": "Answer behavior measures utilization; it is not causal evidence.",
    }
    return sorted(normalized, key=lambda row: (row["decision_id"], CONTROLS.index(row["control"]))), summaries, deltas, report


def summarize_utilization(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each grounded cohort separately, per attribute and macro."""

    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["cohort"]), str(row["control"]),
                 _integer(row["attribute_id"], "attribute_id"),
                 str(row["attribute_name"]))].append(row)
    attribute_rows: list[dict[str, Any]] = []
    for (cohort, control, aid, name), rows in sorted(grouped.items()):
        attribute_rows.append(_summary_row(cohort, control, aid, name, rows, "attribute"))

    result = list(attribute_rows)
    macro_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attribute_rows:
        macro_groups[(row["cohort"], row["control"])].append(row)
    for (cohort, control), rows in sorted(macro_groups.items()):
        macro = {
            "scope": "macro_attribute",
            "cohort": cohort,
            "control": control,
            "attribute_id": "ALL_SELECTED_ATTRIBUTES",
            "attribute_name": "ALL_SELECTED_ATTRIBUTES",
            "n_decisions": sum(int(row["n_decisions"]) for row in rows),
            "evaluable_attributes": len(rows),
        }
        for field in SUMMARY_FIELDS:
            macro[field] = statistics.mean(float(row[field]) for row in rows)
        result.append(macro)
    return sorted(result, key=lambda row: (
        row["scope"], row["cohort"], row["control"], str(row["attribute_id"])
    ))


def paired_control_deltas(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare the real image against each matched control within decision."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[str(row["decision_id"])][str(row["control"])] = row
    result: list[dict[str, Any]] = []
    metrics = ("signed_answer_margin", "margin_correct", "generation_correct")
    for did, controls in sorted(grouped.items()):
        require(set(controls) == set(CONTROLS), f"incomplete controls for {did}")
        image = controls["image"]
        for reference in ("prompt_only", "image_shuffled"):
            control = controls[reference]
            row = {
                "decision_id": did,
                "image_id": image["image_id"],
                "attribute_id": image["attribute_id"],
                "attribute_name": image["attribute_name"],
                "cohort": image["cohort"],
                "reference": reference,
            }
            for metric in metrics:
                row[f"image_{metric}"] = image[metric]
                row[f"reference_{metric}"] = control[metric]
                row[f"delta_{metric}"] = float(image[metric]) - float(control[metric])
            result.append(row)
    return result


def _normalize_measurement(source: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    require(_integer(row.get("schema_version", SCHEMA_VERSION), "schema_version") == SCHEMA_VERSION,
            "measurement schema version differs")
    iid = _integer(row.get("image_id"), "image_id")
    aid = _integer(row.get("attribute_id"), "attribute_id")
    prompt_id = str(row.get("prompt_id", ""))
    expected_id = decision_id(iid, aid, prompt_id)
    require(row.get("decision_id") == expected_id,
            f"decision_id must equal {expected_id}")
    require(prompt_id == cfg["prompt_id"], "measurement prompt_id differs from policy")
    attribute_name = _canonical_text(row.get("attribute_name"), "attribute_name")
    attribute_phrase = _canonical_text(row.get("attribute_phrase"), "attribute_phrase")
    prompt_text = _canonical_text(row.get("prompt_text"), "prompt_text")
    require(prompt_text == render_binary_prompt(cfg["prompt_template"], attribute_phrase),
            f"prompt text differs for {expected_id}")
    target = _boolean(row.get("target"), "target")
    cohort = str(row.get("cohort", ""))
    require(cohort == ("grounded_positive" if target else "grounded_negative"),
            f"target/cohort mismatch for {expected_id}")
    certainty = str(row.get("certainty_name", "")).strip().casefold()
    require(certainty in APPROVED_CERTAINTY_NAMES,
            f"unapproved certainty for {expected_id}")
    visible = _integer(row.get("relevant_visible_in_crop_parts"),
                       "relevant_visible_in_crop_parts")
    require(visible >= 1, f"ungrounded decision {expected_id}")
    require(row.get("split") == cfg["development_evaluation_split"],
            f"non-validation decision {expected_id}")
    require(row.get("official_split") == "train", f"official test decision {expected_id}")

    control = str(row.get("control", ""))
    require(control in CONTROLS, f"unknown control for {expected_id}")
    evaluated = _optional_integer(row.get("evaluated_image_id"), "evaluated_image_id")
    if control == "image":
        require(evaluated == iid, f"image condition identity differs for {expected_id}")
    elif control == "prompt_only":
        require(evaluated is None, f"prompt-only condition contains an image for {expected_id}")
    else:
        require(evaluated is not None and evaluated != iid,
                f"image-shuffled condition is not a derangement for {expected_id}")

    positive_answer = str(row.get("positive_answer", cfg["positive_answer"]))
    negative_answer = str(row.get("negative_answer", cfg["negative_answer"]))
    require(_canonical_answer(positive_answer) == _canonical_answer(cfg["positive_answer"])
            and _canonical_answer(negative_answer) == _canonical_answer(cfg["negative_answer"]),
            f"binary answer candidates differ for {expected_id}")
    positive_ids = _integer_sequence(row.get("positive_token_ids"), "positive_token_ids")
    negative_ids = _integer_sequence(row.get("negative_token_ids"), "negative_token_ids")
    require(positive_ids != negative_ids, "positive/negative token sequences are identical")
    positive_ll = _candidate_log_likelihood(row, "positive", positive_ids)
    negative_ll = _candidate_log_likelihood(row, "negative", negative_ids)
    margin = positive_ll - negative_ll
    if row.get("answer_margin") not in (None, ""):
        require(math.isclose(_finite_float(row["answer_margin"], "answer_margin"), margin,
                             rel_tol=0.0, abs_tol=1e-7),
                f"answer margin is inconsistent for {expected_id}")
    generated = row.get("generated_text")
    require(isinstance(generated, str), f"generated text is missing for {expected_id}")
    parsed = strict_parse_binary(generated, positive_answer, negative_answer)
    margin_prediction = margin > 0.0 if margin != 0.0 else None
    require(margin_prediction is not None, f"teacher-forced margin ties for {expected_id}")
    signed = margin if target else -margin
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_id": expected_id,
        "image_id": iid,
        "evaluated_image_id": evaluated,
        "attribute_id": aid,
        "attribute_name": attribute_name,
        "attribute_phrase": attribute_phrase,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "target": target,
        "cohort": cohort,
        "certainty_name": certainty,
        "relevant_visible_in_crop_parts": visible,
        "split": row["split"],
        "official_split": row["official_split"],
        "control": control,
        "positive_answer": _canonical_answer(positive_answer),
        "negative_answer": _canonical_answer(negative_answer),
        "positive_token_ids": json.dumps(positive_ids, separators=(",", ":")),
        "negative_token_ids": json.dumps(negative_ids, separators=(",", ":")),
        "positive_log_likelihood": positive_ll,
        "negative_log_likelihood": negative_ll,
        "answer_margin": margin,
        "signed_answer_margin": signed,
        "margin_prediction": margin_prediction,
        "margin_correct": margin_prediction == target,
        "generated_text": generated,
        "parsed_answer": parsed,
        "generation_correct": parsed == target if parsed is not None else False,
        "parse_success": parsed is not None,
    }


def _candidate_log_likelihood(
    row: Mapping[str, Any], prefix: str, token_ids: Sequence[int]
) -> float:
    probabilities_key = f"{prefix}_token_logprobs"
    likelihood_key = f"{prefix}_log_likelihood"
    raw_probabilities = row.get(probabilities_key)
    has_probabilities = raw_probabilities not in (None, "")
    raw_likelihood = row.get(likelihood_key)
    has_likelihood = raw_likelihood not in (None, "")
    require(has_probabilities or has_likelihood,
            f"{prefix} candidate has no teacher-forced likelihood")
    computed: float | None = None
    if has_probabilities:
        probabilities = _float_sequence(raw_probabilities, probabilities_key)
        require(len(probabilities) == len(token_ids),
                f"{prefix} token IDs/log probabilities differ in length")
        computed = sequence_log_likelihood(probabilities)
    if has_likelihood:
        provided = _finite_float(raw_likelihood, likelihood_key)
        require(provided <= 1e-7, f"{likelihood_key} must be non-positive")
        if computed is not None:
            require(math.isclose(provided, computed, rel_tol=0.0, abs_tol=1e-7),
                    f"{likelihood_key} differs from token sum")
        computed = provided
    assert computed is not None
    return computed


def _validate_balanced_cohorts(rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[_integer(row["attribute_id"], "attribute_id")][str(row["cohort"])] += 1
    for aid, counts in grouped.items():
        require(set(counts) == set(COHORTS) and counts[COHORTS[0]] == counts[COHORTS[1]],
                f"attribute {aid} grounded cohorts are absent or unbalanced")


def _validate_shuffle(
    decisions: Sequence[Mapping[str, Any]],
    by_decision: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[_integer(row["attribute_id"], "attribute_id")].append(row)
    for aid, rows in grouped.items():
        original = [int(row["image_id"]) for row in rows]
        shuffled = []
        for row in rows:
            control = next(item for item in by_decision[str(row["decision_id"])]
                           if item["control"] == "image_shuffled")
            shuffled.append(int(control["evaluated_image_id"]))
        require(len(original) == len(set(original)), f"attribute {aid} repeats source images")
        require(sorted(original) == sorted(shuffled) and len(shuffled) == len(set(shuffled)),
                f"attribute {aid} shuffled images are not a permutation")
        require(all(left != right for left, right in zip(original, shuffled)),
                f"attribute {aid} shuffle contains a fixed point")


def _summary_row(
    cohort: str,
    control: str,
    attribute_id: int,
    attribute_name: str,
    rows: Sequence[Mapping[str, Any]],
    scope: str,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "cohort": cohort,
        "control": control,
        "attribute_id": attribute_id,
        "attribute_name": attribute_name,
        "n_decisions": len(rows),
        "evaluable_attributes": 1,
        "mean_answer_margin": statistics.mean(float(row["answer_margin"]) for row in rows),
        "mean_signed_answer_margin": statistics.mean(float(row["signed_answer_margin"]) for row in rows),
        "margin_accuracy": statistics.mean(float(bool(row["margin_correct"])) for row in rows),
        "generation_accuracy": statistics.mean(float(bool(row["generation_correct"])) for row in rows),
        "parse_rate": statistics.mean(float(bool(row["parse_success"])) for row in rows),
    }


def _canonical_answer(value: Any) -> str:
    require(isinstance(value, str) and value.strip() == value and value,
            "answer candidates must be non-empty canonical text")
    return value.casefold()


def _canonical_text(value: Any, name: str) -> str:
    require(isinstance(value, str) and value and value.strip() == value
            and "\n" not in value and "\r" not in value,
            f"{name} must be non-empty single-line canonical text")
    return value


def _integer(value: Any, name: str) -> int:
    require(not isinstance(value, bool), f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Phase5ValidationError(f"{name} must be an integer") from exc
    require(str(result) == str(value).strip() or isinstance(value, int),
            f"{name} must use canonical integer syntax")
    return result


def _optional_integer(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    return _integer(value, name)


def _boolean(value: Any, name: str) -> bool:
    if value is True or value is False:
        return value
    if value in (1, "1", "true", "True"):
        return True
    if value in (0, "0", "false", "False"):
        return False
    raise Phase5ValidationError(f"{name} must be boolean")


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase5ValidationError(f"{name} must be numeric") from exc
    require(math.isfinite(result), f"{name} must be finite")
    return result


def _decoded_sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise Phase5ValidationError(f"{name} must be a JSON array") from exc
    require(isinstance(value, (list, tuple)) and bool(value), f"{name} must be a non-empty array")
    return list(value)


def _integer_sequence(value: Any, name: str) -> list[int]:
    values = [_integer(item, name) for item in _decoded_sequence(value, name)]
    require(all(item >= 0 for item in values), f"{name} cannot contain negative IDs")
    return values


def _float_sequence(value: Any, name: str) -> list[float]:
    return [_finite_float(item, name) for item in _decoded_sequence(value, name)]


def _stable_key(seed: int, attribute_id: int, identity: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{attribute_id}:{identity}".encode("utf-8")).hexdigest()
