import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase6 import (ALLOWED_TRANSITIONS, Phase6Config, apply_transition_rubric,
                         bootstrap_contrast, decision_id, join_phase_rows,
                         run_phase6_analysis)


class Phase6Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Phase6Config(bootstrap_resamples=200, bootstrap_seed=19)
        self.phase3, self.phase4, self.phase5 = self.fixtures()

    def fixtures(self):
        phase3, phase4, phase5 = [], [], []
        # Two attributes and eight independent image clusters each.  Every
        # attribute/target stratum has both a failure and a success.
        for image_id in range(16):
            attribute_id = 2 if image_id < 8 else 5
            target = image_id % 2
            key = decision_id(image_id, attribute_id, self.config.prompt_id)
            base = dict(decision_id=key, image_id=image_id, split="val",
                        attribute_id=attribute_id,
                        attribute_name=f"attribute-{attribute_id}",
                        attribute_group="bill", prompt_id=self.config.prompt_id,
                        target=target)
            for stage in (self.config.vision_stage, self.config.language_stage):
                for seed in (0, 1):
                    phase3.append(dict(base, stage=stage, pooling="mean",
                                       control="primary", seed=seed, observed=1,
                                       probe_correct=1, probe_margin=.8))
            for selector, value in ((self.config.discriminative_selector, .5),
                                    (self.config.semantic_selector, .5)):
                phase4.append(dict(base, K=32, selector=selector,
                                   selection_seed=-1,
                                   part_patch_recall=value))
            # Alternating pairs ensure failures/successes exist for both targets.
            correct = 0 if image_id % 4 in (0, 3) else 1
            phase5.append(dict(base, control="image", margin_correct=correct,
                               generation_correct=correct,
                               signed_answer_margin=1.0 if correct else -1.0))
        return phase3, phase4, phase5

    @staticmethod
    def write(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_join_aggregates_seeds_and_audits_cardinality(self):
        joined, audit = join_phase_rows(self.phase3, self.phase4, self.phase5,
                                        self.config)
        self.assertEqual(len(joined), 16)
        self.assertEqual(audit["joined_images"], 16)
        self.assertEqual(audit["joined_attributes"], 2)
        self.assertEqual(audit["failures"], 8)
        self.assertEqual(audit["unmatched"], {"phase3r": 0, "phase4": 0, "phase5": 0})
        self.assertEqual(joined[0]["phase3_vision_seeds"], 2)

    def test_phase4_legacy_rows_receive_the_frozen_decision_key(self):
        phase4 = []
        for source in self.phase4:
            row = dict(source)
            for field in ("decision_id", "prompt_id", "target"):
                row.pop(field)
            phase4.append(row)
        joined, audit = join_phase_rows(self.phase3, phase4, self.phase5,
                                        self.config)
        self.assertEqual(len(joined), 16)
        self.assertEqual(audit["official_test_images_used"], 0)

    def test_official_test_guard_runs_before_split_filter(self):
        poisoned = list(self.phase4)
        poisoned.append(dict(poisoned[0], split="test", official_split="test"))
        with self.assertRaisesRegex(RuntimeError, "Official-test row"):
            join_phase_rows(self.phase3, poisoned, self.phase5, self.config)

    def test_malformed_key_and_duplicate_cardinality_are_rejected(self):
        malformed = [dict(row) for row in self.phase3]
        malformed[0]["decision_id"] = "999::2::wrong"
        with self.assertRaisesRegex(RuntimeError, "decision_id does not match"):
            join_phase_rows(malformed, self.phase4, self.phase5, self.config)
        duplicate = self.phase5 + [dict(self.phase5[0])]
        with self.assertRaisesRegex(RuntimeError, "one row per decision/control"):
            join_phase_rows(self.phase3, self.phase4, duplicate, self.config)

    def test_partial_inner_join_is_explicitly_reported(self):
        joined, audit = join_phase_rows(self.phase3, self.phase4[:-1], self.phase5,
                                        self.config)
        self.assertEqual(len(joined), 15)
        self.assertEqual(audit["unmatched"]["phase3r"], 1)
        self.assertEqual(audit["phase4"]["incomplete_decisions"], 1)

    def test_fixed_attribute_set_cannot_silently_shrink(self):
        phase4 = [row for row in self.phase4 if row["attribute_id"] == 2]
        with self.assertRaisesRegex(RuntimeError, "fixed attribute sets differ"):
            join_phase_rows(self.phase3, phase4, self.phase5, self.config)

    def test_clustered_bootstrap_is_deterministic(self):
        joined, _ = join_phase_rows(self.phase3, self.phase4, self.phase5,
                                    self.config)
        first = bootstrap_contrast(joined, name="gap", kind="paired",
                                   left="phase3_language_correct",
                                   right="phase5_answer_correct", config=self.config)
        second = bootstrap_contrast(joined, name="gap", kind="paired",
                                    left="phase3_language_correct",
                                    right="phase5_answer_correct", config=self.config)
        self.assertEqual(first, second)
        self.assertEqual({row["aggregation"] for row in first},
                         {"micro", "fixed_attribute_macro", "two_way_micro"})
        self.assertAlmostEqual(first[0]["estimate"], .5)

    def test_direct_patch_evidence_enables_redistribution_rubric(self):
        patch = []
        for base in self.phase5:
            for stage, entropy, indices in (
                (self.config.vision_stage, 4.0, list(range(32))),
                (self.config.language_stage, 5.0, list(range(32, 64))),
            ):
                for seed in (0, 1):
                    patch.append(dict(
                        base,
                        stage=stage,
                        pooling="mean",
                        control="primary",
                        seed=seed,
                        evidence_entropy=entropy,
                        evidence_effective_tokens=20.0,
                        top32_indices=json.dumps(indices),
                    ))
        joined, audit = join_phase_rows(
            self.phase3, self.phase4, self.phase5, self.config, patch
        )
        self.assertEqual(audit["phase3r_patch"]["complete_decisions"], 16)
        inference = []
        inference.extend(bootstrap_contrast(
            joined,
            name="language_minus_vision_evidence_entropy",
            kind="paired",
            left="phase3_language_evidence_entropy",
            right="phase3_vision_evidence_entropy",
            config=self.config,
        ))
        inference.extend(bootstrap_contrast(
            joined,
            name="spatial_evidence_redistribution",
            kind="paired",
            left="phase3_spatial_redistribution",
            right="phase3_zero",
            config=self.config,
        ))
        # Supply neutral rows for the three other predeclared diagnostics.
        for contrast in (
            "vision_minus_language_probe",
            "language_probe_minus_answer",
            "discriminative_minus_semantic_localization",
        ):
            inference.append(dict(
                contrast=contrast,
                kind="paired",
                aggregation="fixed_attribute_macro",
                estimate=0.0,
                ci_low=-0.1,
                ci_high=0.1,
            ))
        rubric = apply_transition_rubric(inference, self.config)
        self.assertTrue(rubric["evidence"]["evidence_redistribution"]["supported"])
        self.assertEqual(rubric["selected_transition"], "evidence_redistribution")

    def test_rubric_selects_single_gap_but_never_forces_multiple_signals(self):
        rows = []
        for contrast, estimate, low, high in (
                ("vision_minus_language_probe", 0.0, -0.1, 0.1),
                ("language_probe_minus_answer", 0.2, 0.1, 0.3),
                ("discriminative_minus_semantic_localization", 0.0, -0.1, 0.1)):
            rows.append(dict(contrast=contrast, kind="paired",
                             aggregation="fixed_attribute_macro",
                             estimate=estimate, ci_low=low, ci_high=high))
        rubric = apply_transition_rubric(rows, self.config)
        self.assertEqual(rubric["selected_transition"],
                         "representation_utilization_gap")
        rows[0].update(estimate=.2, ci_low=.1, ci_high=.3)
        rubric = apply_transition_rubric(rows, self.config)
        self.assertEqual(rubric["selected_transition"], "mixed_or_null")
        self.assertEqual(len(rubric["supported_positive_labels"]), 2)
        self.assertEqual(set(ALLOWED_TRANSITIONS), {
            "visual_to_language_bottleneck", "representation_utilization_gap",
            "discriminative_semantic_localization_mismatch",
            "evidence_redistribution", "mixed_or_null"})

    def test_neutral_prompt_context_blocks_language_side_transition_claim(self):
        rows = []
        for contrast, estimate, low, high in (
                ("vision_minus_language_probe", 0.0, -0.1, 0.1),
                ("language_probe_minus_answer", 0.2, 0.1, 0.3),
                ("discriminative_minus_semantic_localization", 0.0, -0.1, 0.1)):
            rows.append(dict(contrast=contrast, kind="paired",
                             aggregation="fixed_attribute_macro",
                             estimate=estimate, ci_low=low, ci_high=high))
        rubric = apply_transition_rubric(
            rows, self.config, language_prompt_aligned=False
        )
        self.assertEqual(rubric["selected_transition"], "mixed_or_null")
        gap = rubric["evidence"]["representation_utilization_gap"]
        self.assertTrue(gap["diagnostic_supported_before_context_gate"])
        self.assertFalse(gap["supported"])
        self.assertIn("neutral Phase 2 prompt", gap["context_gate"])

    def test_full_run_writes_self_contained_artifacts(self):
        p3, p4, p5 = (self.root / name for name in ("p3.csv", "p4.csv", "p5.csv"))
        self.write(p3, self.phase3)
        self.write(p4, self.phase4)
        self.write(p5, self.phase5)
        output = self.root / "output"
        report = run_phase6_analysis(p3, p4, p5, output, self.config)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["selected_transition"],
                         "representation_utilization_gap")
        for name in report["artifacts"] + ["phase6_run_report.json"]:
            self.assertTrue((output / name).is_file(), name)
        decision = json.loads((output / "transition_decision.json").read_text())
        self.assertEqual(decision["selected_transition"],
                         "representation_utilization_gap")
        findings = (output / "INTERMEDIATE_FINDINGS.md").read_text()
        self.assertIn("Official test images used: 0", findings)
        self.assertIn("matched within attribute and target strata", findings)


if __name__ == "__main__":
    unittest.main()
