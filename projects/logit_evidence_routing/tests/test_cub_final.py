from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cub_final.controls import norm_matched_random
from cub_final.core import ShardWriter, canonical_hash, environment_inventory, read_jsonl
from cub_final.dola import restricted_binary_dola
from cub_final.models import LLAVA_CHECKPOINT, LLAVA_REVISION, QWEN3_CHECKPOINT, QWEN3_REVISION
from cub_final.protocol import lock_protocol, validate_smoke_report
from cub_final.scoring import fixed_binary_parser, polarity_effects
from cub_final.shared import _finished_condition
from cub_final.statistics import paired_image_cluster_bootstrap


class CubFinalTests(unittest.TestCase):
    @staticmethod
    def _smoke(architecture: str) -> dict[str, object]:
        checkpoint, revision = {
            "llava": (LLAVA_CHECKPOINT, LLAVA_REVISION),
            "qwen3": (QWEN3_CHECKPOINT, QWEN3_REVISION),
        }[architecture]
        return {
            "status": "PASS",
            "official_test_images_used": 0,
            "checkpoint": checkpoint,
            "required_revision": revision,
            "requested_quantization": "none",
            "architecture_audit": {
                "architecture": architecture,
                "resolved_revision": revision,
                "runtime": {"quantization": "none"},
            },
            "training_decision": {"image_id": 2, "official_split": "train"},
            "measurement": {
                "architecture": architecture,
                "resolved_revision": revision,
                "final_logit_agreement": {"max_abs_error": 0.0},
            },
        }

    def test_polarity_audit_distinguishes_raw_and_correctness_scales(self):
        positive = polarity_effects(baseline_margin=1.0, intervention_margin=0.5, target=1)
        negative = polarity_effects(baseline_margin=1.0, intervention_margin=0.5, target=0)
        self.assertEqual(positive["delta_raw"], negative["delta_raw"])
        self.assertEqual(positive["delta_correct"], -negative["delta_correct"])

    def test_environment_inventory_includes_huggingface_hub(self):
        self.assertIn("huggingface_hub", environment_inventory())

    def test_fixed_parser_keeps_invalid_answers(self):
        self.assertEqual(fixed_binary_parser("Yes, it is.", positive="yes", negative="no"), "positive")
        self.assertEqual(fixed_binary_parser("perhaps", positive="yes", negative="no"), "invalid")

    def test_control_policy_never_silently_falls_back(self):
        result = norm_matched_random([0, 1], [1.0, 2.0, 1.1, 2.1], seed=7)
        self.assertTrue(result.feasible)
        strict = norm_matched_random(
            [0], [1.0, 1.1], seed=7, spatial_bins=["corner", "center"], strict_spatial=True
        )
        self.assertFalse(strict.feasible)
        self.assertEqual(strict.policy, "spatial_and_norm")

    def test_atomic_shards_resume_only_same_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ShardWriter(root, config_hash="a")
            writer.write("q1::full", {"value": 1})
            self.assertTrue(writer.completed("q1::full"))
            self.assertFalse(ShardWriter(root, config_hash="b").completed("q1::full"))
            output = root / "joined.jsonl"
            self.assertEqual(writer.consolidate(output), 1)
            self.assertEqual(read_jsonl(output)[0]["key"], "q1::full")

    def test_shared_resume_retries_failures_but_preserves_oracle_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ShardWriter(Path(directory), config_hash="config")
            writer.write("q1::full", {"value": 1})
            writer.write("q1::predicted_crop", {"reason": "temporary"}, status="failed")
            writer.write("q1::oracle_part_crop", {"reason": "no visible part"}, status="excluded")
            self.assertTrue(_finished_condition(writer, "q1::full", "full"))
            self.assertFalse(
                _finished_condition(writer, "q1::predicted_crop", "predicted_crop")
            )
            self.assertTrue(
                _finished_condition(writer, "q1::oracle_part_crop", "oracle_part_crop")
            )

    def test_cluster_bootstrap_preserves_image_unit(self):
        rows = [
            {"image_id": "a", "family": "x", "effect": 1.0},
            {"image_id": "a", "family": "x", "effect": 3.0},
            {"image_id": "b", "family": "x", "effect": -2.0},
            {"image_id": "c", "family": "x", "effect": 0.0},
        ]
        result = paired_image_cluster_bootstrap(
            rows, effect_field="effect", family_field="family", resamples=200, seed=3
        )
        self.assertEqual(result["cluster_unit"], "image_id")
        self.assertEqual(result["families"]["x"]["image_count"], 3)
        self.assertAlmostEqual(result["families"]["x"]["estimate"], 0.0)

    def test_restricted_dola_reports_scope(self):
        result = restricted_binary_dola(
            [[0.0, 0.0], [2.0, -1.0], [1.0, 0.0]],
            candidate_layers=[0, 1],
            plausibility_alpha=0.1,
        )
        self.assertIn("not_full_generation", result["scope"])
        self.assertIn(result["selected_premature_layer"], [0, 1])

    def test_protocol_lock_requires_two_training_smokes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests, smoke = root / "manifests", root / "smoke"
            manifests.mkdir()
            smoke.mkdir()
            (manifests / "manifest_summary.json").write_text(
                json.dumps({"manifest_hash": canonical_hash({"x": 1})})
            )
            for architecture in ("llava", "qwen3"):
                (smoke / f"{architecture}_smoke.json").write_text(
                    json.dumps(self._smoke(architecture))
                )
            output = root / "FINAL_PROTOCOL.yaml"
            value = lock_protocol(
                manifest_dir=manifests, smoke_dir=smoke, destination=output
            )
            self.assertEqual(value["status"], "LOCKED")
            self.assertTrue(output.is_file())

    def test_protocol_rejects_a_smoke_without_exact_revision_evidence(self):
        value = self._smoke("qwen3")
        value["measurement"]["resolved_revision"] = "unknown"  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "locked revision"):
            validate_smoke_report(value, architecture="qwen3")

    def test_protocol_rejects_official_test_smoke(self):
        value = self._smoke("llava")
        value["training_decision"]["official_split"] = "test"  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "official training split"):
            validate_smoke_report(value, architecture="llava")


if __name__ == "__main__":
    unittest.main()
