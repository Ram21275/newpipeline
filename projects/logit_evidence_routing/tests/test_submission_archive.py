"""Tests for deterministic paper-result archive construction."""

from __future__ import annotations

import runpy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]


class SubmissionArchiveTests(unittest.TestCase):
    def test_same_inputs_produce_same_archive_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            (root / "results" / "a.txt").write_text("a\n")
            script = PROJECT / "scripts" / "build_submission_archive.py"
            hashes = []
            for name in ("first.tar.gz", "second.tar.gz"):
                output = root / name
                argv = [str(script), "--root", str(root), "--include", "results",
                        "--output", str(output)]
                with patch.object(sys, "argv", argv):
                    runpy.run_path(str(script), run_name="__main__")
                audit = json.loads(Path(str(output) + ".audit.json").read_text())
                self.assertTrue(audit["internal_manifest_verified"])
                hashes.append(output.read_bytes())
            self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
