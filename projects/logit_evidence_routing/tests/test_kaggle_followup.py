"""Tests for the compact Kaggle workflow without loading a model."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
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

EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_phase9_full_selector_metadata",
    PROJECT / "scripts" / "export_phase9_full_selector_metadata.py",
)
EXPORTER = importlib.util.module_from_spec(EXPORT_SPEC)
assert EXPORT_SPEC.loader is not None
EXPORT_SPEC.loader.exec_module(EXPORTER)


class CompactWorkflowTests(unittest.TestCase):
    def test_resolves_nested_corrected_phase1b_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            corrected = Path(directory) / "phase1b_corrected"
            cache = corrected / "cache"
            (cache / "records").mkdir(parents=True)
            (cache / "extraction_config.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(COMPACT.resolve_localizer_cache(corrected), cache)
            self.assertEqual(EXPORTER.resolve_localizer_cache(corrected), cache)

    def test_finds_cub_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cub = root / "dataset" / "CUB_200_2011"
            (cub / "images").mkdir(parents=True)
            (cub / "images.txt").write_text("1 bird.jpg\n", encoding="utf-8")
            self.assertEqual(COMPACT.find_cub_root(root), cub)

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

    def test_cub_manifest_audit_checks_all_control_identities_and_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cub_replication_manifest.csv"
            rows = []
            for target in (0, 1):
                rows.append({
                    "decision_id": f"d{target}",
                    "image_id": f"i{target}",
                    "attribute_id": "1",
                    "target": str(target),
                    "prompt_text": "Answer yes or no.",
                    "relative_path": f"i{target}.jpg",
                    "shuffled_image_id": f"s{target}",
                    "shuffled_relative_path": f"s{target}.jpg",
                    "opposite_label_image_id": f"i{1 - target}",
                    "opposite_label_relative_path": f"i{1 - target}.jpg",
                    "opposite_label_target": str(1 - target),
                    "opposite_label_seed": "31415",
                    "official_split": "train",
                    "official_test_image": "0",
                })
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(COMPACT.Workflow.validate_replication_manifest(path)), 2)
            rows[0]["official_test_image"] = "1"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(COMPACT.WorkflowError, "official-test"):
                COMPACT.Workflow.validate_replication_manifest(path)

    def test_interval_audit_requires_conservative_null_interpretation(self):
        path = Path("analysis.json")
        report = {"contrasts": [{
            "estimate": 0.1, "ci_low": -0.2, "ci_high": 0.3,
            "interpretation": "null_compatible",
        }]}
        COMPACT.validate_intervals(path, report, ("contrasts",))
        report["contrasts"][0]["interpretation"] = "positive"
        with self.assertRaisesRegex(COMPACT.WorkflowError, "null-compatible"):
            COMPACT.validate_intervals(path, report, ("contrasts",))

    def test_compact_cli_has_cub_only_resumable_stages(self):
        command = COMPACT.parser()
        prepare = command.parse_args([
            "--notebook-safe", "--celeba", "no", "prepare", "--skip-tests",
        ])
        self.assertEqual(prepare.command, "prepare")
        self.assertTrue(prepare.notebook_safe)
        self.assertEqual(command.parse_args(["--celeba", "no", "preflight"]).command, "preflight")
        self.assertEqual(command.parse_args(["llava"]).command, "llava")
        qwen = command.parse_args(["qwen", "--clear-llava-checkpoint"])
        self.assertTrue(qwen.clear_llava_checkpoint)
        self.assertEqual(command.parse_args(["status"]).command, "status")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            command.parse_args(["--celeba", "auto", "preflight"])

    def test_notebook_safe_failure_returns_zero_and_saves_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working, inputs = root / "working", root / "input"
            inputs.mkdir()
            result = subprocess.run([
                sys.executable, str(PROJECT / "scripts" / "run_kaggle_followup.py"),
                "--working-root", str(working), "--input-root", str(inputs),
                "--notebook-safe", "prepare", "--skip-tests",
            ], check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            failure = working / "lger_compact_last_failure.json"
            self.assertTrue(failure.is_file())
            report = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAILED")
            self.assertIn("CUB_200_2011", report["error"])
            self.assertIn("STOP HERE", result.stderr)

    def test_failed_stage_retains_subprocess_output_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = COMPACT.Workflow(root / "working", root / "input")
            with self.assertRaises(COMPACT.WorkflowError) as raised:
                workflow.run("synthetic failure", [
                    sys.executable, "-c", "print('specific child error'); raise SystemExit(7)",
                ])
            self.assertEqual(raised.exception.stage, "synthetic failure")
            self.assertIn("specific child error", raised.exception.output_tail)
            self.assertIn("specific child error", str(raised.exception))

    def test_silent_subprocess_emits_workflow_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = COMPACT.Workflow(root / "working", root / "input")
            previous = COMPACT.HEARTBEAT_SECONDS
            COMPACT.HEARTBEAT_SECONDS = 0.01
            try:
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    workflow.run("silent stage", [
                        sys.executable, "-c", "import time; time.sleep(0.08)",
                    ])
            finally:
                COMPACT.HEARTBEAT_SECONDS = previous
            self.assertIn("[workflow heartbeat] silent stage is still running", captured.getvalue())

    def test_archive_hash_is_verified_before_checkpoint_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working, inputs = root / "working", root / "input"
            working.mkdir()
            inputs.mkdir()
            (working / "artifact.txt").write_text("retained result\n", encoding="utf-8")
            workflow = COMPACT.Workflow(working, inputs)
            archive = workflow.create_archive("result.tar.gz", ["artifact.txt"])
            self.assertTrue((working / "result.sha256").is_file())
            self.assertEqual(workflow.verify_archive(archive.name), COMPACT.sha256(archive))
            archive.write_bytes(archive.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(COMPACT.WorkflowError, "SHA-256 differs"):
                workflow.verify_archive(archive.name)


if __name__ == "__main__":
    unittest.main()
