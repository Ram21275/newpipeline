"""Synthetic tests for submission-oriented retained-artifact analyses."""

from __future__ import annotations

import unittest

from lger.paper_robustness import (
    group_robustness,
    image_influence,
    phase9_family_rows,
    replication_family_rows,
    simultaneous_image_bootstrap,
)
from lger.phase9 import build_selector_intervention_plan


METHODS = ("vision_cls_attention", "logit_concept")
STAGES = ("vision.early", "vision.middle", "vision.late", "projector.output")


def manifest():
    return [{
        "decision_id": f"d{i}", "image_id": f"i{i}", "class_id": str((i - 1) % 2),
        "attribute_id": str((i - 1) % 2), "target": str(i % 2),
        "split": "val", "official_split": "train",
    } for i in range(1, 5)]


def token_rows():
    rows = []
    for i in range(1, 5):
        for method_index, method in enumerate(METHODS):
            for token in range(12):
                rows.append({
                    "decision_id": f"d{i}", "image_id": f"i{i}",
                    "attribute_id": (i - 1) % 2, "target": i % 2,
                    "phase5_failure": i % 2, "selection_method": method,
                    "stage": "vision.late", "token_index": token,
                    "evidence_score": 100 - token + method_index * 0.01,
                    "bird_box_membership": "inside" if token % 2 else "outside",
                    "hidden_state_norm": 1 + token,
                })
    return rows


def plan_and_outcomes():
    plan = build_selector_intervention_plan(
        token_rows(), selected_stage="vision.late", neighbor_stage="projector.output",
        stage_order=STAGES, k=2, selector_methods=METHODS, random_seeds=(0, 1),
        norm_quantiles=2,
    )
    outcomes = []
    for row in plan["records"]:
        if row["selection_method"] == "global_control":
            drop = 4.0
        elif row["intervention_type"] == "matched_random":
            drop = 1.0
        else:
            drop = 2.0 + (row["selection_method"] == METHODS[0]) \
                + (row["stage"] == "vision.late")
        outcomes.append({
            "decision_id": row["decision_id"], "intervention_id": row["intervention_id"],
            "correct_answer_teacher_forced_margin_before": 5.0,
            "correct_answer_teacher_forced_margin_after": 5.0 - drop,
        })
    return plan, outcomes


class PaperRobustnessTests(unittest.TestCase):
    def test_phase9_family_simultaneous_groups_and_influence(self):
        plan, outcomes = plan_and_outcomes()
        rows = phase9_family_rows(outcomes, plan, manifest())
        self.assertEqual(len({row["estimand"] for row in rows}), 6)
        simultaneous = simultaneous_image_bootstrap(
            rows, samples=200, seed=2, confidence_level=0.95
        )
        self.assertEqual(len(simultaneous), 6)
        self.assertTrue(all(row["family_size"] == 6 for row in simultaneous))
        groups = group_robustness(
            rows, samples=200, seed=3, confidence_level=0.95
        )
        self.assertEqual(len(groups), 12)
        per_image, influence = image_influence(rows)
        self.assertEqual(len(per_image), 24)
        self.assertEqual(len(influence), 6)

    def test_replication_family_has_all_model_control_metric_pairs(self):
        rows = []
        for item in manifest():
            for control, margin in (
                ("image", 1.0), ("prompt_only", 0.0),
                ("image_shuffled", -0.5), ("opposite_label_image", -1.0),
            ):
                rows.append({
                    "decision_id": item["decision_id"], "control": control,
                    "correct_answer_margin": margin, "margin_correct": int(margin > 0),
                })
        result = replication_family_rows({"llava": rows, "qwen": rows}, manifest())
        self.assertEqual(len({row["estimand"] for row in result}), 12)
        self.assertEqual(len(result), 48)


if __name__ == "__main__":
    unittest.main()
