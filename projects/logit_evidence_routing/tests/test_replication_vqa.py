"""Tests for cross-model VQA manifest controls without loading a model."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DONORS = load("build_opposite_label_controls", "build_opposite_label_controls.py")
REPLICATION = load("extract_replication_vqa", "extract_replication_vqa.py")


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


if __name__ == "__main__":
    unittest.main()
