"""Synthetic tests for raw/correct Phase 10 confirmatory inference."""

from __future__ import annotations

import unittest

from lger.phase10 import build_priority1_plan
from lger.phase10_confirmatory import _cluster_accuracy_interval, run_confirmatory_analysis


METHODS = ("vision_cls_attention", "logit_concept")
STAGES = ("vision.early", "vision.middle", "vision.late", "projector.output")


def token_rows():
    rows = []
    for index in range(8):
        target = index % 2
        attribute = (index // 2) % 2
        class_id = (index // 4) % 2
        for method_index, method in enumerate(METHODS):
            for token in range(12):
                rows.append({
                    "decision_id": f"d{index}",
                    "image_id": f"i{index}",
                    "class_id": class_id,
                    "attribute_id": attribute,
                    "target": target,
                    "phase5_failure": target,
                    "selection_method": method,
                    "stage": "vision.late",
                    "token_index": token,
                    "evidence_score": 100 - token + method_index * 0.01,
                    "bird_box_membership": "inside" if token % 2 else "outside",
                    "hidden_state_norm": 1 + token,
                })
    return rows


def source_plan():
    return build_priority1_plan(
        token_rows(), selected_stage="vision.late", neighbor_stage="projector.output",
        stage_order=STAGES, selector_methods=METHODS, balanced_k=2,
        dose_selector=METHODS[0], dose_k_values=(2, 4), random_seeds=(0, 1),
        norm_quantiles=2,
    )


def outcomes(plan):
    rows = []
    for row in plan["records"]:
        target = int(row["target"])
        sign = 1.0 if target else -1.0
        raw_damage = 0.0
        if row["intervention_type"] == "top_evidence":
            raw_damage = (1.0 + 0.2 * target) \
                if row["selection_method"] == METHODS[0] else (0.2 + 0.1 * target)
        elif row["selection_method"] == "global_control":
            raw_damage = 2.0
        correct_damage = sign * raw_damage
        rows.append({
            "decision_id": row["decision_id"],
            "intervention_id": row["intervention_id"],
            "correct_answer_teacher_forced_margin_before": 1.0,
            "correct_answer_teacher_forced_margin_after": 1.0 - correct_damage,
        })
    return rows


def behavior_rows():
    rows = []
    for index in range(8):
        target = index % 2
        for control in ("image", "prompt_only"):
            correct = int(control == "image" or target == 0)
            rows.append({
                "decision_id": f"d{index}", "image_id": f"i{index}",
                "target": target, "control": control,
                "margin_correct": correct,
                "generation_correct": correct if not (index == 0 and control == "image") else "",
            })
    return rows


class Phase10ConfirmatoryTests(unittest.TestCase):
    def test_accuracy_is_decision_weighted_with_image_cluster_uncertainty(self):
        rows = [
            {"image_id": "a", "value": 1.0},
            {"image_id": "a", "value": 1.0},
            {"image_id": "a", "value": 1.0},
            {"image_id": "b", "value": 0.0},
        ]
        result = _cluster_accuracy_interval(
            rows, value_field="value", samples=100, seed=7, confidence_level=0.95
        )
        self.assertEqual(result["estimate"], 0.75)
        self.assertEqual(result["estimand_weighting"], "decision")
        self.assertEqual(result["n_images"], 2)

    def test_raw_scale_removes_sign_induced_target_reversal(self):
        plan = source_plan()
        result = run_confirmatory_analysis(
            outcomes(plan), plan=plan, decision_metadata=token_rows(),
            behavior_by_model={"toy": behavior_rows()}, bootstrap_samples=100, seed=9,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["decision_count"], 8)
        self.assertEqual(len(result["main_inference"]), 36)
        lookup = {row["estimand"]: row for row in result["main_inference"]}
        self.assertAlmostEqual(
            lookup[
                "selector_difference::raw_yes_minus_no::vision.late::target_0"
            ]["estimate"],
            0.8,
        )
        self.assertAlmostEqual(
            lookup[
                "selector_difference::raw_yes_minus_no::vision.late::target_1"
            ]["estimate"],
            0.9,
        )
        self.assertAlmostEqual(
            lookup[
                "selector_difference_target_difference::raw_yes_minus_no::vision.late"
            ]["estimate"],
            0.1,
        )
        self.assertAlmostEqual(
            lookup[
                "selector_difference_target_difference::correct_answer_margin::vision.late"
            ]["estimate"],
            1.7,
        )
        self.assertTrue(all(row["family_size"] == 18 for row in result["main_inference"]))

    def test_blocked_leave_one_out_and_absolute_accuracy_are_complete(self):
        plan = source_plan()
        result = run_confirmatory_analysis(
            outcomes(plan), plan=plan, decision_metadata=token_rows(),
            behavior_by_model={"toy": behavior_rows()}, bootstrap_samples=100, seed=11,
        )
        self.assertEqual(len(result["blocked_robustness"]), 72)
        self.assertTrue(result["leave_one_group_out"])
        self.assertEqual(
            {row["grouping"] for row in result["blocked_robustness"]},
            {"attribute_id", "class_id"},
        )
        accuracy = result["absolute_accuracy"]
        baseline = next(row for row in accuracy
                        if row["source"] == "phase10_intervention_baseline"
                        and row["target"] == "all")
        self.assertEqual(baseline["estimate"], 1.0)
        generated = next(row for row in accuracy
                         if row["model"] == "toy" and row["control"] == "image"
                         and row["metric"] == "generation_accuracy_all"
                         and row["target"] == "all")
        self.assertAlmostEqual(generated["estimate"], 0.875)


if __name__ == "__main__":
    unittest.main()
