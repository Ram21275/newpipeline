"""Synthetic tests for the balanced-target and K-dose intervention extension."""

from __future__ import annotations

import math
import unittest

from lger.phase10 import aggregate_priority1, build_priority1_plan
from lger.phase9 import build_selector_intervention_plan


METHODS = ("vision_cls_attention", "logit_concept")
STAGES = ("vision.early", "vision.middle", "vision.late", "projector.output")


def token_rows():
    rows = []
    for index in range(1, 5):
        target = index % 2
        for method_index, method in enumerate(METHODS):
            for token in range(16):
                rows.append({
                    "decision_id": f"d{index}", "image_id": f"i{index}",
                    "attribute_id": index % 2, "target": target,
                    "phase5_failure": target, "selection_method": method,
                    "stage": "vision.late", "token_index": token,
                    "evidence_score": 100 - token + method_index * 0.01,
                    "bird_box_membership": "inside" if token % 2 else "outside",
                    "hidden_state_norm": 1 + token,
                })
    return rows


def sparse_box_token_rows():
    rows = []
    for index in range(1, 5):
        target = index % 2
        for method_index, method in enumerate(METHODS):
            for token in range(16):
                rows.append({
                    "decision_id": f"d{index}", "image_id": f"i{index}",
                    "attribute_id": index % 2, "target": target,
                    "phase5_failure": target, "selection_method": method,
                    "stage": "vision.late", "token_index": token,
                    "evidence_score": 100 - token + method_index * 0.01,
                    "bird_box_membership": "inside" if token < 4 else "outside",
                    "hidden_state_norm": 1 + token,
                })
    return rows


def plan():
    return build_priority1_plan(
        token_rows(), selected_stage="vision.late", neighbor_stage="projector.output",
        stage_order=STAGES, selector_methods=METHODS, balanced_k=2,
        dose_selector=METHODS[0], dose_k_values=(2, 4), random_seeds=(0, 1),
        norm_quantiles=2,
    )


def outcomes(source_plan):
    result = []
    for row in source_plan["records"]:
        if row["selection_method"] == "global_control":
            drop = 5.0
        elif row["intervention_type"] == "matched_random":
            drop = 1.0
        else:
            effect = (2.0 if row["selection_method"] == METHODS[0] else 1.0)
            effect += 0.5 * int(row["target"])
            effect += 0.25 * math.log2(max(1, int(row["selection_k"])))
            drop = 1.0 + effect
        result.append({
            "decision_id": row["decision_id"], "intervention_id": row["intervention_id"],
            "correct_answer_teacher_forced_margin_before": 10.0,
            "correct_answer_teacher_forced_margin_after": 10.0 - drop,
        })
    return result


class Phase10Tests(unittest.TestCase):
    def test_plan_is_balanced_deduplicated_and_development_only(self):
        first = plan()
        self.assertEqual(first, plan())
        self.assertEqual(first["target_counts"], {"0": 2, "1": 2})
        self.assertEqual(first["intervention_count"], 80)
        self.assertEqual(len({row["intervention_id"] for row in first["records"]}), 80)
        self.assertTrue(first["phase8_protocol_unchanged"])
        self.assertEqual(first["official_test_images_used"], 0)

    def test_positive_balanced_component_keeps_original_phase9_identities(self):
        original = build_selector_intervention_plan(
            [row for row in token_rows() if row["target"] == 1],
            selected_stage="vision.late", neighbor_stage="projector.output",
            stage_order=STAGES, k=2, selector_methods=METHODS,
            random_seeds=(0, 1), norm_quantiles=2,
        )
        extended = plan()
        positive_k2 = {
            row["intervention_id"] for row in extended["records"]
            if row["target"] == 1 and (row["selection_k"] == 2
                                       or row["selection_method"] == "global_control")
        }
        self.assertEqual(
            positive_k2,
            {row["intervention_id"] for row in original["records"]},
        )

    def test_large_k_predeclared_spatial_fallback_is_audited(self):
        result = build_priority1_plan(
            sparse_box_token_rows(), selected_stage="vision.late",
            neighbor_stage="projector.output", stage_order=STAGES,
            selector_methods=METHODS, balanced_k=2, dose_selector=METHODS[0],
            dose_k_values=(2, 4), random_seeds=(0, 1), norm_quantiles=2,
        )
        self.assertEqual(result["matching_spatial_fallback_count"], 2)
        self.assertEqual(
            {row["matching_spatial_mode"] for row in result["records"]
             if row["selection_k"] == 4},
            {"norm_only_fallback"},
        )

    def test_aggregate_reports_target_interaction_and_dose_slope(self):
        source_plan = plan()
        result = aggregate_priority1(
            outcomes(source_plan), plan=source_plan, bootstrap_samples=200, seed=4
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["selector_specific_contrasts"]), 4)
        self.assertEqual(len(result["selector_target_interactions"]), 4)
        self.assertEqual(len(result["selector_by_target_interactions"]), 2)
        self.assertEqual(len(result["dose_effects"]), 4)
        self.assertEqual(len(result["dose_log2_trends"]), 2)
        for row in result["selector_target_interactions"]:
            self.assertAlmostEqual(row["estimate"], 0.5)
        for row in result["selector_by_target_interactions"]:
            self.assertAlmostEqual(row["estimate"], 0.0)
        for row in result["dose_log2_trends"]:
            self.assertAlmostEqual(row["estimate"], 0.25)


if __name__ == "__main__":
    unittest.main()
