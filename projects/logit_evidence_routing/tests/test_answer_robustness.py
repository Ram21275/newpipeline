"""Tests for answer-vocabulary and prompt-order counterbalancing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lger.answer_robustness import (
    AnswerRobustnessError,
    analyze_answer_robustness,
    render_condition_prompt,
    validate_conditions,
)


PROJECT = Path(__file__).resolve().parents[1]


class AnswerRobustnessTests(unittest.TestCase):
    def config(self):
        return json.loads(
            (PROJECT / "configs" / "answer_token_robustness.json").read_text()
        )

    def test_config_has_three_vocabularies_and_counterbalanced_order(self):
        rows = validate_conditions(self.config()["conditions"])
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["vocabulary"] for row in rows},
            {"yes_no", "true_false", "present_absent"},
        )
        for row in rows:
            prompt = render_condition_prompt(row, "a red crown")
            self.assertIn("a red crown", prompt)

    def test_missing_order_pair_fails(self):
        rows = self.config()["conditions"][:-1]
        with self.assertRaises(AnswerRobustnessError):
            validate_conditions(rows)

    def test_analysis_separates_order_vocabulary_and_visual_effects(self):
        rows = []
        conditions = validate_conditions(self.config()["conditions"])
        for decision, target in (("d0", 0), ("d1", 1)):
            for condition in conditions:
                base = {"yes_no": 0.0, "true_false": 0.2, "present_absent": -0.1}[
                    condition["vocabulary"]
                ]
                order = 0.05 if condition["answer_order"] == "positive_first" else 0.0
                for control, visual in (("image", 0.4), ("prompt_only", 0.0),
                                        ("image_shuffled", 0.1)):
                    margin = base + order + visual
                    rows.append({
                        "decision_id": decision,
                        "image_id": "i" + decision[-1],
                        "target": target,
                        "condition_id": condition["condition_id"],
                        "vocabulary": condition["vocabulary"],
                        "answer_order": condition["answer_order"],
                        "control": control,
                        "semantic_positive_minus_negative_margin": margin,
                        "margin_correct": int((margin > 0) == bool(target)),
                        "generation_correct": int((margin > 0) == bool(target)),
                    })
        result = analyze_answer_robustness(rows, samples=100, seed=3)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["order_inference"]), 9)
        self.assertEqual(len(result["vocabulary_inference"]), 6)
        self.assertEqual(len(result["visual_effect_rows"]), 24)


if __name__ == "__main__":
    unittest.main()
