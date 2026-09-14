"""Tests for cross-model VQA manifest controls without loading a model."""

from __future__ import annotations

import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DONORS = load("build_opposite_label_controls", "build_opposite_label_controls.py")
REPLICATION = load("extract_replication_vqa", "extract_replication_vqa.py")
ANALYSIS = load("analyze_replication_vqa", "analyze_replication_vqa.py")


def decisions():
    rows = []
    for attribute in (1, 2):
        for target in (0, 1):
            for index in range(3):
                image_id = attribute * 100 + target * 10 + index
                rows.append({
                    "decision_id": f"d-{image_id}-{attribute}",
                    "image_id": image_id,
                    "attribute_id": attribute,
                    "attribute_name": f"a{attribute}",
                    "target": target,
                    "class_id": index % 2,
                    "relative_path": f"{image_id}.jpg",
                    "shuffled_image_id": image_id + 1000,
                    "shuffled_relative_path": f"s-{image_id}.jpg",
                    "prompt_text": "Answer yes or no.",
                })
    return rows


class OppositeLabelTests(unittest.TestCase):
    def test_donors_are_deterministic_opposite_and_prefer_same_class(self):
        first = DONORS.add_opposite_label_donors(decisions(), seed=7)
        second = DONORS.add_opposite_label_donors(decisions(), seed=7)
        self.assertEqual(first, second)
        self.assertTrue(all(row["opposite_label_target"] == 1 - row["target"] for row in first))
        self.assertTrue(all(row["opposite_label_image_id"] != row["image_id"] for row in first))
        self.assertTrue(all(row["opposite_label_same_class"] == 1 for row in first))

    def test_missing_opposite_label_fails(self):
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            DONORS.add_opposite_label_donors(
                [row for row in decisions() if row["target"] == 1], seed=1
            )


class ReplicationCohortTests(unittest.TestCase):
    def test_smoke_has_both_labels_and_full_keeps_all(self):
        rows = DONORS.add_opposite_label_donors(decisions(), seed=7)
        smoke = REPLICATION.choose_decisions(rows, "smoke", 4)
        self.assertEqual({int(row["target"]) for row in smoke}, {0, 1})
        self.assertEqual(REPLICATION.choose_decisions(rows, "full", 4), sorted(
            rows, key=lambda row: (int(row["attribute_id"]), int(row["target"]), row["decision_id"])
        ))

    def test_pilot_round_robins_attribute_label_strata(self):
        rows = DONORS.add_opposite_label_donors(decisions(), seed=7)
        pilot = REPLICATION.choose_decisions(rows, "pilot", 8)
        self.assertEqual(len(pilot), 8)
        self.assertEqual({(int(row["attribute_id"]), int(row["target"])) for row in pilot},
                         {(1, 0), (1, 1), (2, 0), (2, 1)})

    def test_result_row_preserves_boolean_no_and_scores_both_answers(self):
        base = {
            "decision_id": "d1", "image_id": "1", "attribute_id": "2",
            "attribute_name": "has_bill_shape::dagger", "target": "0",
            "shuffled_image_id": "2", "opposite_label_image_id": "3",
        }
        result = SimpleNamespace(
            positive_log_likelihood=-2.0, negative_log_likelihood=-1.0,
            answer_margin=-1.0, generated_text="No", attention_entropy=None,
            attention_effective_tokens=None, positive_token_ids=[1], negative_token_ids=[2],
        )
        negative = REPLICATION.result_row(base, "image", result, "llava")
        self.assertIs(negative["parsed_answer"], False)
        self.assertEqual(negative["generation_correct"], 1)

        positive_decision = {**base, "target": "1"}
        positive_result = SimpleNamespace(**{**result.__dict__, "generated_text": "Yes"})
        positive = REPLICATION.result_row(positive_decision, "image", positive_result, "llava")
        self.assertIs(positive["parsed_answer"], True)
        self.assertEqual(positive["generation_correct"], 1)


class ReplicationAnalysisTests(unittest.TestCase):
    def write_results(self, root: Path) -> Path:
        path = root / "replication_vqa_decisions.csv"
        rows = []
        margins = {
            "image": (0.9, 0.7),
            "prompt_only": (0.1, 0.0),
            "image_shuffled": (-0.1, 0.2),
            "opposite_label_image": (-0.7, -0.4),
        }
        for index, target in enumerate((1, 0)):
            for control in ANALYSIS.CONTROLS:
                margin = margins[control][index]
                rows.append({
                    "decision_id": f"d{index}",
                    "image_id": f"i{index}",
                    "attribute_id": "1",
                    "target": target,
                    "control": control,
                    "correct_answer_margin": margin,
                    "margin_tie": int(margin == 0),
                    "margin_correct": int(margin > 0),
                    "generation_correct": int(margin > 0),
                })
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_analysis_keeps_ties_incorrect_and_pairs_all_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            decisions_by_id = ANALYSIS.normalize("test", self.write_results(Path(directory)))
        self.assertEqual(decisions_by_id["d1"]["prompt_only"]["tie"], 1)
        self.assertEqual(decisions_by_id["d1"]["prompt_only"]["correct"], 0)
        result = ANALYSIS.analyze_model(
            "test", decisions_by_id, samples=100, seed=4, confidence=0.95
        )
        image = next(row for row in result["summaries"] if row["control"] == "image")
        self.assertEqual(image["margin_accuracy"], 1.0)
        self.assertEqual(image["generation_parse_rate"], 1.0)
        self.assertEqual(image["generation_accuracy"], 1.0)
        contrast = next(row for row in result["contrasts"]
                        if row["contrast"] == "image_minus_prompt_only"
                        and row["metric"] == "correct_answer_margin")
        self.assertAlmostEqual(contrast["estimate"], 0.75)

    def test_exact_tie_cannot_be_marked_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_results(Path(directory))
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["margin_tie"] = "1"
            rows[0]["margin_correct"] = "1"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "tie"):
                ANALYSIS.normalize("test", path)


if __name__ == "__main__":
    unittest.main()
