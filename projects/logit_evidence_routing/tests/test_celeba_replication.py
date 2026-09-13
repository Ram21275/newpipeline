"""Tests for identity-disjoint CelebA development sampling."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_celeba_replication", PROJECT / "scripts" / "prepare_celeba_replication.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CelebASamplingTests(unittest.TestCase):
    def test_balanced_sample_is_deterministic_and_identity_unique(self):
        candidates = []
        for index in range(40):
            candidates.append({
                "filename": f"{index:06d}.jpg",
                "identity_id": index,
                "attributes": {"Eyeglasses": index % 2, "Smiling": (index // 2) % 2},
            })
        first = MODULE.balanced_identity_sample(
            candidates, count=20, attributes=["Eyeglasses", "Smiling"], seed=3, split="val"
        )
        second = MODULE.balanced_identity_sample(
            candidates, count=20, attributes=["Eyeglasses", "Smiling"], seed=3, split="val"
        )
        self.assertEqual(first, second)
        self.assertEqual(len({row["identity_id"] for row in first}), 20)
        for attribute in ("Eyeglasses", "Smiling"):
            positives = sum(row["attributes"][attribute] for row in first)
            self.assertTrue(7 <= positives <= 13)

    def test_insufficient_identities_fail(self):
        with self.assertRaisesRegex(RuntimeError, "distinct identities"):
            MODULE.balanced_identity_sample(
                [{"filename": "a.jpg", "identity_id": 1,
                  "attributes": {"Eyeglasses": 1}}],
                count=2, attributes=["Eyeglasses"], seed=1, split="train",
            )


if __name__ == "__main__":
    unittest.main()
