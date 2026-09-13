"""Tests for the compact Kaggle workflow without loading a model."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_kaggle_followup", PROJECT / "scripts" / "run_kaggle_followup.py"
)
COMPACT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPACT)


class CompactWorkflowTests(unittest.TestCase):
    def test_finds_cub_and_celeba_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cub = root / "dataset" / "CUB_200_2011"
            (cub / "images").mkdir(parents=True)
            (cub / "images.txt").write_text("1 bird.jpg\n", encoding="utf-8")
            celeba = root / "faces"
            (celeba / "Img" / "img_celeba").mkdir(parents=True)
            self.assertEqual(COMPACT.find_cub_root(root), cub)
            self.assertEqual(COMPACT.find_celeba_root(root), celeba)

    def test_report_gate_requires_pass_and_zero_official_test_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({
                "status": "PASS", "mode": "smoke", "official_test_images_used": 0,
            }), encoding="utf-8")
            self.assertEqual(COMPACT.validate_report(path, mode="smoke")["status"], "PASS")
            path.write_text(json.dumps({
                "status": "PASS", "official_test_images_used": 1,
            }), encoding="utf-8")
            with self.assertRaisesRegex(COMPACT.WorkflowError, "official-test"):
                COMPACT.validate_report(path)

    def test_compact_cli_has_four_resumable_stages(self):
        command = COMPACT.parser()
        self.assertEqual(command.parse_args(["prepare", "--skip-tests"]).command, "prepare")
        self.assertEqual(command.parse_args(["llava"]).command, "llava")
        qwen = command.parse_args(["qwen", "--clear-llava-checkpoint"])
        self.assertTrue(qwen.clear_llava_checkpoint)
        self.assertEqual(command.parse_args(["status"]).command, "status")

    def test_archive_hash_is_verified_before_checkpoint_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working, inputs = root / "working", root / "input"
            working.mkdir()
            inputs.mkdir()
            (working / "artifact.txt").write_text("retained result\n", encoding="utf-8")
            workflow = COMPACT.Workflow(working, inputs, "no")
            archive = workflow.create_archive("result.tar.gz", ["artifact.txt"])
            self.assertTrue((working / "result.sha256").is_file())
            self.assertEqual(workflow.verify_archive(archive.name), COMPACT.sha256(archive))
            archive.write_bytes(archive.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(COMPACT.WorkflowError, "SHA-256 differs"):
                workflow.verify_archive(archive.name)


if __name__ == "__main__":
    unittest.main()
