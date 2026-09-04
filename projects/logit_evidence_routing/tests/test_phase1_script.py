import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lger.cub import PilotRecord, save_pilot_manifest


class PhaseOneScriptTests(unittest.TestCase):
    def test_probe_script_writes_all_required_tables(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pilot.csv"
            records = [
                PilotRecord(1, "a/1.jpg", 1, "a", "train"),
                PilotRecord(2, "a/2.jpg", 1, "a", "val"),
                PilotRecord(3, "b/3.jpg", 2, "b", "train"),
                PilotRecord(4, "b/4.jpg", 2, "b", "val"),
            ]
            save_pilot_manifest(records, manifest, {"official_test_images_used": 0})
            cache_records = root / "cache" / "records"
            cache_records.mkdir(parents=True)
            for record in records:
                feature = torch.tensor([-1.0, 0.0]) if record.label == 1 else torch.tensor([1.0, 0.0])
                selections = {
                    "random_k2_seed0": torch.tensor([0, 1]),
                    "attention_k2": torch.tensor([1, 2]),
                    "logit_k2": torch.tensor([2, 3]),
                }
                torch.save(
                    {
                        "image_id": record.image_id,
                        "label": record.label,
                        "split": record.split,
                        "features": {
                            "random_k2_seed0": feature,
                            "attention_k2": feature,
                            "logit_k2": feature,
                        },
                        "selections": selections,
                    },
                    cache_records / f"{record.image_id:05d}.pt",
                )
            output = root / "results"
            arguments = [
                str(project_root / "scripts" / "run_phase1_probes.py"),
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(root / "cache"),
                "--output-dir",
                str(output),
                "--k",
                "2",
                "--seeds",
                "0",
                "--epochs",
                "2",
            ]
            with patch.object(sys, "argv", arguments):
                runpy.run_path(arguments[0], run_name="__main__")
            for filename in (
                "selector_metrics.csv",
                "selector_summary.csv",
                "patch_statistics.csv",
                "evaluation_config.json",
                "notes.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
