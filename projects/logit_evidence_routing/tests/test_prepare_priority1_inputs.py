"""Tests for bounded Kaggle retained-input discovery."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_priority1_inputs", PROJECT / "scripts" / "prepare_priority1_inputs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PreparePriority1InputsTests(unittest.TestCase):
    def test_scan_finds_markers_without_descending_into_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "dataset" / "multiphase_development_5b55fde6a51f"
            (run / "phase7_bridge").mkdir(parents=True)
            (run / "multiphase_status.json").write_text("{}")
            cub = root / "birds" / "CUB_200_2011"
            (cub / "images" / "class").mkdir(parents=True)
            (cub / "images.txt").write_text("1 image.jpg\n")
            # A marker below a pruned payload must never be considered.
            hidden = cub / "images" / "class" / "phase2_stage_cache"
            hidden.mkdir()
            for name in ("validation_report.json", "index.json", "run_config.json"):
                (hidden / name).write_text("{}")
            found = MODULE.scan(root)
            self.assertEqual(found["run"], [run])
            self.assertEqual(found["cub"], [cub])
            self.assertEqual(found["stage_cache"], [])


if __name__ == "__main__":
    unittest.main()
