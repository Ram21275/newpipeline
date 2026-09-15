"""Protocol validation and prompt rendering for answer-token robustness."""

from __future__ import annotations

from collections import defaultdict
import math
import random
import statistics
from typing import Any, Mapping, Sequence


class AnswerRobustnessError(RuntimeError):
    """Raised when counterbalancing or semantic answer mappings are invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnswerRobustnessError(message)


def validate_conditions(conditions: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Require two prompt orders for each of three distinct answer vocabularies."""

    require(len(conditions) >= 6, "answer robustness requires at least six conditions")
    output: list[dict[str, str]] = []
    ids: set[str] = set()
    by_vocabulary: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in conditions:
        row = {key: str(source.get(key, "")).strip() for key in (
            "condition_id", "vocabulary", "answer_order", "prompt_template",
            "positive_answer", "negative_answer",
        )}
        require(all(row.values()), "answer condition fields must be non-empty")
        require(row["condition_id"] not in ids, "answer condition IDs must be unique")
        ids.add(row["condition_id"])
        require(row["answer_order"] in ("positive_first", "negative_first"),
                "answer_order must be positive_first or negative_first")
        require(row["prompt_template"].count("{attribute_phrase}") == 1,
                "prompt template must contain one {attribute_phrase} placeholder")
        require("\n" not in row["prompt_template"] and "\r" not in row["prompt_template"],
                "prompt templates must be single-line")
        require(row["positive_answer"].casefold() != row["negative_answer"].casefold(),
                "semantic positive and negative answers must differ")
        by_vocabulary[row["vocabulary"]].append(row)
        output.append(row)
    required_vocabularies = {"yes_no", "true_false", "present_absent"}
    require(required_vocabularies <= set(by_vocabulary),
            "yes/no, true/false, and present/absent vocabularies are required")
    for vocabulary, rows in by_vocabulary.items():
        require({row["answer_order"] for row in rows} == {"positive_first", "negative_first"},
                f"{vocabulary} must counterbalance answer order")
        pairs = {(row["positive_answer"].casefold(), row["negative_answer"].casefold())
                 for row in rows}
        require(len(pairs) == 1, f"{vocabulary} changes its semantic answer mapping")
    return output


def render_condition_prompt(condition: Mapping[str, str], attribute_phrase: str) -> str:
    phrase = str(attribute_phrase).strip()
    require(bool(phrase) and "\n" not in phrase and "\r" not in phrase,
            "attribute_phrase must be non-empty single-line text")
    prompt = str(condition["prompt_template"]).format(attribute_phrase=phrase)
    require(prompt.strip() == prompt and "\n" not in prompt and "\r" not in prompt,
            "rendered answer-robustness prompt must be canonical single-line text")
    return prompt


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnswerRobustnessError(f"{name} must be numeric") from error
    require(math.isfinite(result), f"{name} must be finite")
    return result


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value: str,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    require(samples >= 100 and 0.5 < confidence_level < 1.0, "invalid bootstrap settings")
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(_finite(row[value], value))
    require(bool(by_image), "contrast contains no images")
    images = sorted(by_image)
    estimate = statistics.mean(v for values in by_image.values() for v in values)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [images[rng.randrange(len(images))] for _ in images]
        draws.append(statistics.mean(v for image in chosen for v in by_image[image]))
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


def analyze_answer_robustness(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10000,
    seed: int = 20260915,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Separate vocabulary, answer-order, and visual-condition sensitivity."""

    require(bool(rows), "answer-token result table is empty")
    required = {
        "decision_id", "image_id", "target", "condition_id", "vocabulary",
        "answer_order", "control", "semantic_positive_minus_negative_margin",
        "margin_correct", "generation_correct",
    }
    require(all(required <= set(row) for row in rows), "answer-token table lacks required fields")
    keys = [(str(row["decision_id"]), str(row["condition_id"]), str(row["control"]))
            for row in rows]
    require(len(keys) == len(set(keys)), "answer-token rows are duplicated")
    decisions = sorted({str(row["decision_id"]) for row in rows})
    conditions = sorted({str(row["condition_id"]) for row in rows})
    controls = sorted({str(row["control"]) for row in rows})
    expected = {(decision, condition, control) for decision in decisions
                for condition in conditions for control in controls}
    require(set(keys) == expected, "answer-token result coverage is incomplete")
    require({str(row["vocabulary"]) for row in rows}
            == {"yes_no", "true_false", "present_absent"},
            "answer-token table must contain all three vocabularies")

    normalized = [{
        **row,
        "raw_margin": _finite(row["semantic_positive_minus_negative_margin"], "raw margin"),
        "accuracy": float(int(row["margin_correct"])),
        "generation_parse": float(str(row.get("generation_correct", "")) != ""),
        "generation_accuracy_all": (
            float(int(row["generation_correct"]))
            if str(row.get("generation_correct", "")) != "" else 0.0
        ),
    } for row in rows]
    summary: list[dict[str, Any]] = []
    for control in controls:
        for vocabulary in ("yes_no", "true_false", "present_absent"):
            for order in ("positive_first", "negative_first"):
                subset = [row for row in normalized if row["control"] == control
                          and row["vocabulary"] == vocabulary and row["answer_order"] == order]
                require(bool(subset), "missing vocabulary/order/control cell")
                summary.append({
                    "control": control,
                    "vocabulary": vocabulary,
                    "answer_order": order,
                    "decisions": len(subset),
                    "mean_raw_semantic_margin": statistics.mean(row["raw_margin"] for row in subset),
                    "teacher_forced_accuracy": statistics.mean(row["accuracy"] for row in subset),
                    "generation_parse_rate": statistics.mean(row["generation_parse"] for row in subset),
                    "generation_accuracy_all": statistics.mean(
                        row["generation_accuracy_all"] for row in subset
                    ),
                })

    indexed = {
        (str(row["decision_id"]), str(row["control"]), str(row["vocabulary"]),
         str(row["answer_order"])): row for row in normalized
    }
    order_rows: list[dict[str, Any]] = []
    for decision in decisions:
        for control in controls:
            for vocabulary in ("yes_no", "true_false", "present_absent"):
                positive = indexed[(decision, control, vocabulary, "positive_first")]
                negative = indexed[(decision, control, vocabulary, "negative_first")]
                order_rows.append({
                    "decision_id": decision,
                    "image_id": positive["image_id"],
                    "target": int(positive["target"]),
                    "control": control,
                    "vocabulary": vocabulary,
                    "positive_first_minus_negative_first": (
                        positive["raw_margin"] - negative["raw_margin"]
                    ),
                })
    order_inference = []
    for control in controls:
        for vocabulary in ("yes_no", "true_false", "present_absent"):
            subset = [row for row in order_rows if row["control"] == control
                      and row["vocabulary"] == vocabulary]
            order_inference.append({
                "contrast": "positive_first_minus_negative_first",
                "control": control,
                "vocabulary": vocabulary,
                **_cluster_interval(
                    subset, value="positive_first_minus_negative_first", samples=samples,
                    seed=seed + len(order_inference), confidence_level=confidence_level,
                ),
            })

    averaged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for decision in decisions:
        for control in controls:
            for vocabulary in ("yes_no", "true_false", "present_absent"):
                pair = [indexed[(decision, control, vocabulary, order)]
                        for order in ("positive_first", "negative_first")]
                averaged[(decision, control, vocabulary)] = {
                    "decision_id": decision,
                    "image_id": pair[0]["image_id"],
                    "target": int(pair[0]["target"]),
                    "raw_margin": statistics.mean(row["raw_margin"] for row in pair),
                }
    vocabulary_rows: list[dict[str, Any]] = []
    for decision in decisions:
        for control in controls:
            reference = averaged[(decision, control, "yes_no")]
            for vocabulary in ("true_false", "present_absent"):
                candidate = averaged[(decision, control, vocabulary)]
                vocabulary_rows.append({
                    "decision_id": decision,
                    "image_id": reference["image_id"],
                    "target": reference["target"],
                    "control": control,
                    "vocabulary": vocabulary,
                    "vocabulary_minus_yes_no": candidate["raw_margin"] - reference["raw_margin"],
                })
    vocabulary_inference = []
    for control in controls:
        for vocabulary in ("true_false", "present_absent"):
            subset = [row for row in vocabulary_rows if row["control"] == control
                      and row["vocabulary"] == vocabulary]
            vocabulary_inference.append({
                "contrast": "vocabulary_minus_yes_no_after_averaging_answer_order",
                "control": control,
                "vocabulary": vocabulary,
                **_cluster_interval(
                    subset, value="vocabulary_minus_yes_no", samples=samples,
                    seed=seed + 100 + len(vocabulary_inference),
                    confidence_level=confidence_level,
                ),
            })

    visual_rows: list[dict[str, Any]] = []
    by_key = {(str(row["decision_id"]), str(row["condition_id"]), str(row["control"])): row
              for row in normalized}
    for decision in decisions:
        for condition in conditions:
            image_row = by_key[(decision, condition, "image")]
            for control in ("prompt_only", "image_shuffled"):
                comparator = by_key[(decision, condition, control)]
                visual_rows.append({
                    "decision_id": decision,
                    "image_id": image_row["image_id"],
                    "target": int(image_row["target"]),
                    "condition_id": condition,
                    "vocabulary": image_row["vocabulary"],
                    "answer_order": image_row["answer_order"],
                    "contrast": f"image_minus_{control}",
                    "raw_visual_effect": image_row["raw_margin"] - comparator["raw_margin"],
                })

    stability = []
    for decision in decisions:
        for control in controls:
            values = [row["raw_margin"] for row in normalized
                      if row["decision_id"] == decision and row["control"] == control]
            nonzero_signs = {value > 0 for value in values if value != 0}
            first = next(row for row in normalized
                         if row["decision_id"] == decision and row["control"] == control)
            stability.append({
                "decision_id": decision,
                "image_id": first["image_id"],
                "target": int(first["target"]),
                "control": control,
                "conditions": len(values),
                "raw_margin_mean": statistics.mean(values),
                "raw_margin_population_sd": statistics.pstdev(values),
                "unanimous_nonzero_semantic_sign": int(len(nonzero_signs) == 1),
            })
    return {
        "schema_version": 1,
        "status": "PASS",
        "decisions": len(decisions),
        "conditions": len(conditions),
        "controls": controls,
        "summary": summary,
        "order_inference": order_inference,
        "vocabulary_inference": vocabulary_inference,
        "visual_effect_rows": visual_rows,
        "decision_stability": stability,
        "bootstrap_samples": samples,
        "confidence_level": confidence_level,
        "official_test_images_used": 0,
        "claim_boundary": (
            "Agreement across vocabularies and answer order supports semantic affirmative "
            "evidence; disagreement identifies token/prompt sensitivity, not an internal mechanism."
        ),
    }
