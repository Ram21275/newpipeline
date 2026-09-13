"""Synthetic-only tests for resumable multiphase checkpoint recognition."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_multiphase_development.py"
SPEC = importlib.util.spec_from_file_location("_multiphase_runner_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load multiphase runner: {SCRIPT}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class MultiphaseResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root / "report.json"
        self.artifact = self.root / "artifact.csv"

    def write_report(self, **updates: object) -> None:
        payload = {"status": "PASS", "official_test_images_used": 0, **updates}
        self.report.write_text(json.dumps(payload), encoding="utf-8")

    def test_reuses_only_pass_report_with_complete_artifacts(self) -> None:
        self.write_report()
        self.artifact.write_text("value\n1\n", encoding="utf-8")
        self.assertTrue(
            RUNNER.completed_step("synthetic", self.report, (self.artifact,))
        )
        self.artifact.unlink()
        self.assertFalse(
            RUNNER.completed_step("synthetic", self.report, (self.artifact,))
        )

    def test_rejects_official_test_access(self) -> None:
        self.write_report(official_test_images_used=1)
        self.artifact.write_text("value\n1\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "official-test access"):
            RUNNER.completed_step("synthetic", self.report, (self.artifact,))


if __name__ == "__main__":
    unittest.main()
