"""Synthetic-only Phase 5 identity, likelihood, controls, and gate tests."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lger.phase5 import (
    Phase5ValidationError,
    balance_grounded_decisions,
    build_control_requests,
    decision_id,
    make_grounded_decision,
    render_binary_prompt,
    sequence_log_likelihood,
    strict_parse_binary,
    teacher_forced_margin,
    validate_phase5_records,
)


PROJECT = Path(__file__).resolve().parents[1]


def config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "prompt_id": "cub_attribute_yes_no_v1",
        "prompt_template": "Does the bird have {attribute_phrase}? Answer only yes or no.",
        "positive_answer": "yes",
        "negative_answer": "no",
        "primary_metric": "teacher_forced_sequence_log_likelihood_margin",
        "controls": ["image", "prompt_only", "image_shuffled"],
        "development_evaluation_split": "val",
        "cohorts": ["grounded_positive", "grounded_negative"],
        "require_visible_relevant_part_for_both_labels": True,
        "official_test_images_used": 0,
    }


def decisions() -> list[dict[str, object]]:
    rows = []
    for image_id, target in ((10, True), (11, False), (20, True), (21, False)):
        attribute_id = 7 if image_id < 20 else 8
        phrase = "an all-purpose bill" if attribute_id == 7 else "a cone-shaped bill"
        row = make_grounded_decision(
            image_id=image_id,
            attribute_id=attribute_id,
            attribute_name=f"bill::{attribute_id}",
            attribute_phrase=phrase,
            prompt_id="cub_attribute_yes_no_v1",
            prompt_template=config()["prompt_template"],
            primary_target=target,
            certainty_name="probably" if target else "definitely",
            relevant_visible_in_crop_parts=1,
        )
        assert row is not None
        rows.append(row)
    return rows


def measurements() -> list[dict[str, object]]:
    rows = build_control_requests(decisions(), shuffle_seed=2718)
    for row in rows:
        target = bool(row["target"])
        control = str(row["control"])
        strength = {"image": 3.0, "prompt_only": 0.5, "image_shuffled": 1.0}[control]
        margin = strength if target else -strength
        if target:
            positive_logprobs = [-0.2, -0.3]
            negative_logprobs = [-0.5 - strength]
        else:
            positive_logprobs = [(-0.5 - strength) / 2] * 2
            negative_logprobs = [-0.5]
        row.update(
            positive_answer="yes",
            negative_answer="no",
            positive_token_ids=[101, 102],
            negative_token_ids=[201],
            positive_token_logprobs=positive_logprobs,
            negative_token_logprobs=negative_logprobs,
            answer_margin=margin,
            generated_text="yes" if target else "no",
        )
    return rows


class Phase5MathTests(unittest.TestCase):
    def test_decision_identity_and_deterministic_prompt(self) -> None:
        self.assertEqual(decision_id(10, 7, "p1"), "10::7::p1")
        self.assertEqual(
            render_binary_prompt("Does it show {attribute_phrase}?", "a red crown"),
            "Does it show a red crown?",
        )
        with self.assertRaises(Phase5ValidationError):
            render_binary_prompt("{attribute_phrase} {attribute_phrase}", "a red crown")

    def test_teacher_forced_margin_sums_all_candidate_tokens(self) -> None:
        self.assertAlmostEqual(sequence_log_likelihood([-0.2, -0.3]), -0.5)
        self.assertAlmostEqual(teacher_forced_margin([-0.2, -0.3], [-3.5]), 3.0)
        with self.assertRaises(Phase5ValidationError):
            sequence_log_likelihood([])
        with self.assertRaises(Phase5ValidationError):
            sequence_log_likelihood([-0.2, float("nan")])

    def test_strict_parser_rejects_explanations_and_punctuation(self) -> None:
        self.assertIs(strict_parse_binary(" yes "), True)
        self.assertIs(strict_parse_binary("NO"), False)
        self.assertIsNone(strict_parse_binary("yes."))
        self.assertIsNone(strict_parse_binary("yes, it does"))

    def test_grounding_masks_uncertain_and_invisible_both_labels(self) -> None:
        base = dict(
            image_id=1, attribute_id=7, attribute_name="bill::7",
            attribute_phrase="an all-purpose bill", prompt_id="p1",
            prompt_template="Does it show {attribute_phrase}?",
            relevant_visible_in_crop_parts=1,
        )
        self.assertIsNone(make_grounded_decision(
            **base, primary_target=True, certainty_name="guess"))
        self.assertIsNone(make_grounded_decision(
            **dict(base, relevant_visible_in_crop_parts=0),
            primary_target=False, certainty_name="definitely"))
        negative = make_grounded_decision(
            **base, primary_target=False, certainty_name="probably")
        self.assertEqual(negative["cohort"], "grounded_negative")

    def test_balancing_and_shuffle_are_deterministic(self) -> None:
        source = decisions()
        extra = copy.deepcopy(source[1])
        extra["image_id"] = 12
        extra["decision_id"] = decision_id(12, 7, "cub_attribute_yes_no_v1")
        balanced = balance_grounded_decisions(source + [extra], seed=1729)
        self.assertEqual(len(balanced), 4)
        first = build_control_requests(balanced, shuffle_seed=2718)
        second = build_control_requests(balanced, shuffle_seed=2718)
        self.assertEqual(first, second)
        shuffled = [row for row in first if row["control"] == "image_shuffled"]
        self.assertTrue(all(row["image_id"] != row["evaluated_image_id"] for row in shuffled))


class Phase5ValidationTests(unittest.TestCase):
    def test_complete_precomputed_records_pass_and_keep_cohorts_separate(self) -> None:
        metrics, summaries, deltas, report = validate_phase5_records(measurements(), config())
        self.assertEqual(len(metrics), 12)
        self.assertEqual(len(deltas), 8)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["cohort_decisions"], {
            "grounded_negative": 2, "grounded_positive": 2,
        })
        macros = [row for row in summaries if row["scope"] == "macro_attribute"]
        self.assertEqual(len(macros), 6)
        self.assertEqual({row["cohort"] for row in macros}, {
            "grounded_positive", "grounded_negative",
        })
        image = next(row for row in macros
                     if row["cohort"] == "grounded_positive" and row["control"] == "image")
        self.assertEqual(image["mean_signed_answer_margin"], 3.0)
        self.assertEqual(image["margin_accuracy"], 1.0)

    def test_gate_rejects_identity_certainty_parser_and_margin_tampering(self) -> None:
        mutations = [
            ("decision_id", "wrong"),
            ("certainty_name", "guess"),
            ("answer_margin", 999.0),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                rows = measurements()
                rows[0][field] = value
                with self.assertRaises(Phase5ValidationError):
                    validate_phase5_records(rows, config())

    def test_unparseable_generation_is_measured_not_a_gate_failure(self) -> None:
        rows = measurements()
        rows[0]["generated_text"] = "yes."
        metrics, _, _, report = validate_phase5_records(rows, config())
        changed = next(
            row for row in metrics
            if row["decision_id"] == rows[0]["decision_id"]
            and row["control"] == rows[0]["control"]
        )
        self.assertFalse(changed["parse_success"])
        self.assertFalse(changed["generation_correct"])
        self.assertEqual(report["status"], "PASS")

    def test_gate_rejects_missing_control_and_non_derangement(self) -> None:
        rows = measurements()
        with self.assertRaisesRegex(Phase5ValidationError, "every control"):
            validate_phase5_records(rows[1:], config())
        rows = measurements()
        shuffled = next(row for row in rows if row["control"] == "image_shuffled")
        shuffled["evaluated_image_id"] = shuffled["image_id"]
        with self.assertRaisesRegex(Phase5ValidationError, "derangement"):
            validate_phase5_records(rows, config())

    def test_gate_rejects_unbalanced_cohorts_and_official_test(self) -> None:
        rows = [row for row in measurements() if row["image_id"] != 11]
        with self.assertRaisesRegex(Phase5ValidationError, "unbalanced"):
            validate_phase5_records(rows, config())
        rows = measurements()
        rows[0]["official_split"] = "test"
        with self.assertRaisesRegex(Phase5ValidationError, "official test"):
            validate_phase5_records(rows, config())


class Phase5CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config()), encoding="utf-8")

    def run_cli(self, rows: list[dict[str, object]], output: str) -> subprocess.CompletedProcess[str]:
        path = self.root / f"{output}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(PROJECT / "scripts/run_phase5_utilization.py"),
             "--records", str(path), "--config", str(self.config),
             "--output-dir", str(self.root / output)],
            cwd=PROJECT, text=True, capture_output=True, check=False,
        )

    def test_cli_writes_csv_json_and_pass_gate(self) -> None:
        completed = self.run_cli(measurements(), "pass")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = self.root / "pass"
        gate = json.loads((output / "phase5_gate.json").read_text())
        report = json.loads((output / "phase5_run_report.json").read_text())
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(report["decisions"], 4)
        with (output / "decision_metrics.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)

    def test_cli_failure_cannot_leave_a_pass_gate(self) -> None:
        rows = measurements()
        rows[0]["certainty_name"] = "guess"
        completed = self.run_cli(rows, "fail")
        self.assertNotEqual(completed.returncode, 0)
        gate = json.loads((self.root / "fail/phase5_gate.json").read_text())
        self.assertEqual(gate["status"], "FAIL")
        self.assertFalse((self.root / "fail/phase5_run_report.json").exists())


if __name__ == "__main__":
    unittest.main()
