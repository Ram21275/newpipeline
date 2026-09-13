"""Synthetic tests for the development-only Phase 9 mechanism design."""

from __future__ import annotations

import unittest

from lger.phase9 import (
    GLOBAL_METHOD,
    GLOBAL_TYPE,
    Phase9ValidationError,
    aggregate_selector_interventions,
    build_selector_intervention_plan,
)


STAGES = ("vision.early", "vision.middle", "vision.late", "projector.output", "llm.early")
METHODS = ("vision_cls_attention", "logit_concept")


def token_rows(decisions=("d1", "d2", "d3", "d4")):
    rows = []
    for decision_number, decision_id in enumerate(decisions, 1):
        for method_number, method in enumerate(METHODS):
            for token_index in range(12):
                rows.append({
                    "decision_id": decision_id,
                    "image_id": f"i{decision_number}",
                    "attribute_id": decision_number,
                    "target": 1,
                    "phase5_failure": int(decision_number > 2),
                    "selection_method": method,
                    "stage": "vision.late",
                    "token_index": token_index,
                    "evidence_score": 100 - token_index + method_number * 0.01,
                    "bird_box_membership": "inside" if token_index % 2 == 0 else "outside",
                    "hidden_state_norm": 1.0 + token_index,
                })
    return rows


def make_plan():
    return build_selector_intervention_plan(
        token_rows(),
        selected_stage="vision.late",
        neighbor_stage="projector.output",
        stage_order=STAGES,
        k=2,
        selector_methods=METHODS,
        random_seeds=(0, 1),
        norm_quantiles=2,
    )


def outcomes(plan):
    rows = []
    for record in plan["records"]:
        if record["intervention_type"] == GLOBAL_TYPE:
            drop = 4.0
        elif record["intervention_type"] == "matched_random":
            drop = 1.0
        else:
            method_bonus = 2.0 if record["selection_method"] == METHODS[0] else 1.0
            stage_bonus = 1.0 if record["stage"] == "vision.late" else 0.0
            failure_bonus = 1.0 if record["phase5_failure"] else 0.0
            drop = 1.0 + method_bonus + stage_bonus + failure_bonus
        rows.append({
            "decision_id": record["decision_id"],
            "intervention_id": record["intervention_id"],
            "correct_answer_teacher_forced_margin_before": 5.0,
            "correct_answer_teacher_forced_margin_after": 5.0 - drop,
        })
    return rows


class Phase9PlanTests(unittest.TestCase):
    def test_plan_is_development_only_and_holds_masks_fixed_across_stages(self):
        first, second = make_plan(), make_plan()
        self.assertEqual(first, second)
        self.assertTrue(first["development_only"])
        self.assertTrue(first["phase8_protocol_unchanged"])
        self.assertEqual(first["official_test_images_used"], 0)
        self.assertEqual(first["decision_count"], 4)
        self.assertEqual(first["intervention_count"], 56)
        for decision_id in ("d1", "d2", "d3", "d4"):
            for method in METHODS:
                selected = [row for row in first["records"] if
                            row["decision_id"] == decision_id
                            and row["selection_method"] == method
                            and row["stage"] == "vision.late"]
                neighbor = [row for row in first["records"] if
                            row["decision_id"] == decision_id
                            and row["selection_method"] == method
                            and row["stage"] == "projector.output"]
                self.assertEqual(
                    [(row["intervention_type"], row["replicate"], row["token_indices"])
                     for row in selected],
                    [(row["intervention_type"], row["replicate"], row["token_indices"])
                     for row in neighbor],
                )
        global_rows = [row for row in first["records"] if row["selection_method"] == GLOBAL_METHOD]
        self.assertEqual(len(global_rows), 8)
        self.assertTrue(all(row["token_count"] == 12 for row in global_rows))

    def test_incomplete_selector_coverage_fails_closed(self):
        rows = [row for row in token_rows() if not (
            row["decision_id"] == "d1" and row["selection_method"] == METHODS[1]
        )]
        with self.assertRaisesRegex(Phase9ValidationError, "same decisions"):
            build_selector_intervention_plan(
                rows,
                selected_stage="vision.late",
                neighbor_stage="projector.output",
                stage_order=STAGES,
                k=2,
                selector_methods=METHODS,
                random_seeds=(0,),
            )


class Phase9AggregationTests(unittest.TestCase):
    def test_selector_stage_failure_and_global_effects(self):
        plan = make_plan()
        result = aggregate_selector_interventions(
            outcomes(plan), plan=plan, bootstrap_samples=200, seed=8
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["official_test_images_used"], 0)
        selected = {
            (row["stage"], row["selection_method"]): row
            for row in result["selector_specific_contrasts"]
        }
        self.assertAlmostEqual(selected[("vision.late", METHODS[0])]["estimate"], 3.5)
        self.assertAlmostEqual(selected[("projector.output", METHODS[0])]["estimate"], 2.5)
        self.assertAlmostEqual(selected[("vision.late", METHODS[1])]["estimate"], 2.5)
        stages = {row["selection_method"]: row for row in result["stage_interactions"]}
        self.assertAlmostEqual(stages[METHODS[0]]["estimate"], 1.0)
        selector = next(row for row in result["selector_interactions"]
                        if row["stage"] == "vision.late")
        self.assertAlmostEqual(selector["estimate"], 1.0)
        failures = {(row["stage"], row["selection_method"]): row
                    for row in result["failure_success_interactions"]}
        self.assertAlmostEqual(failures[("vision.late", METHODS[0])]["estimate"], 1.0)
        self.assertTrue(all(row["estimate"] == 4.0
                            for row in result["global_manipulation_checks"]))

    def test_missing_outcome_fails_closed(self):
        plan = make_plan()
        with self.assertRaisesRegex(Phase9ValidationError, "do not cover"):
            aggregate_selector_interventions(
                outcomes(plan)[:-1], plan=plan, bootstrap_samples=20
            )


if __name__ == "__main__":
    unittest.main()
