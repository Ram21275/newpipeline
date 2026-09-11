import unittest

from lger.phase7_bridge import (
    CAUSAL_STAGE_ORDER,
    resolve_intervention_stages,
    select_matched_causal_cohort,
)


class Phase7BridgeTest(unittest.TestCase):
    def test_transition_pairs_are_adjacent(self):
        for transition in (
            "visual_to_language_bottleneck",
            "representation_utilization_gap",
            "discriminative_semantic_localization_mismatch",
            "evidence_redistribution",
        ):
            left, right = resolve_intervention_stages(transition)
            self.assertEqual(abs(CAUSAL_STAGE_ORDER.index(left) - CAUSAL_STAGE_ORDER.index(right)), 1)

    def test_mixed_result_blocks_intervention(self):
        with self.assertRaisesRegex(RuntimeError, "mixed_or_null"):
            resolve_intervention_stages("mixed_or_null")

    def test_cohort_is_balanced_and_capped(self):
        rows = []
        for attribute_id in (1, 2):
            for failure in (0, 1):
                for image_id in range(5):
                    rows.append({
                        "decision_id": f"{attribute_id}-{failure}-{image_id}",
                        "image_id": image_id,
                        "attribute_id": attribute_id,
                        "target": 1,
                        "phase5_failure": failure,
                    })
        chosen = select_matched_causal_cohort(
            rows, max_per_outcome_per_attribute=2, seed=11
        )
        self.assertEqual(len(chosen), 8)
        for attribute_id in (1, 2):
            counts = [
                sum(int(row["attribute_id"]) == attribute_id and int(row["phase5_failure"]) == value
                    for row in chosen)
                for value in (0, 1)
            ]
            self.assertEqual(counts, [2, 2])
        again = select_matched_causal_cohort(
            rows, max_per_outcome_per_attribute=2, seed=11
        )
        self.assertEqual(chosen, again)


if __name__ == "__main__":
    unittest.main()
