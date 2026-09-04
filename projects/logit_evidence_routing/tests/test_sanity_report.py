import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lger.cub import PilotRecord, save_pilot_manifest


class SanityReportTests(unittest.TestCase):
    def test_report_audits_split_cache_seeds_and_parts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cub = root / "CUB_200_2011"
            (cub / "images" / "001.a").mkdir(parents=True)
            (cub / "images" / "002.b").mkdir(parents=True)
            (cub / "parts").mkdir()
            records = [
                PilotRecord(1, "001.a/1.jpg", 1, "001.a", "train"),
                PilotRecord(2, "001.a/2.jpg", 1, "001.a", "val"),
                PilotRecord(3, "002.b/3.jpg", 2, "002.b", "train"),
                PilotRecord(4, "002.b/4.jpg", 2, "002.b", "val"),
            ]
            for record in records:
                (cub / "images" / record.relative_path).touch()
            (cub / "images.txt").write_text(
                "\n".join(f"{row.image_id} {row.relative_path}" for row in records) + "\n"
            )
            (cub / "image_class_labels.txt").write_text(
                "\n".join(f"{row.image_id} {row.label}" for row in records) + "\n"
            )
            (cub / "train_test_split.txt").write_text(
                "\n".join(f"{row.image_id} 1" for row in records) + "\n"
            )
            (cub / "classes.txt").write_text("1 001.a\n2 002.b\n")
            (cub / "bounding_boxes.txt").write_text(
                "\n".join(f"{row.image_id} 0 0 4 4" for row in records) + "\n"
            )
            (cub / "parts" / "part_locs.txt").write_text(
                "\n".join(f"{row.image_id} 1 1 1 1" for row in records) + "\n"
            )

            manifest = root / "pilot.csv"
            save_pilot_manifest(records, manifest, {"official_test_images_used": 0})
            cache = root / "cache"
            (cache / "records").mkdir(parents=True)
            (cache / "extraction_config.json").write_text(
                '{"schema_version": 2, "concept_tokenization_policy": '
                '"single_lexical_token_v1", "concept_token_ids": [1], '
                '"concept_tokens": ["▁bird"]}\n'
            )
            for record in records:
                torch.save(
                    {
                        "schema_version": 2,
                        "image_id": record.image_id,
                        "relative_path": record.relative_path,
                        "label": record.label,
                        "class_name": record.class_name,
                        "split": record.split,
                        "original_image_size": (24, 24),
                        "processed_image_size": (24, 24),
                        "grid_size": (24, 24),
                        "features": {
                            "global_all": torch.tensor([float(record.image_id), 0.0])
                        },
                        "selections": {
                            "vision_cls_attention_k16": torch.arange(16),
                            "vision_cls_attention_k32": torch.arange(32),
                        },
                    },
                    cache / "records" / f"{record.image_id:05d}.pt",
                )

            results = root / "results"
            (results / "qualitative").mkdir(parents=True)
            (results / "qualitative" / "sample.png").touch()
            (results / "evaluation_config.json").write_text(
                '{"k_values": [16, 32]}\n'
            )
            (results / "selector_summary.csv").write_text(
                "selector,K,accuracy_mean,accuracy_std,macro_f1_mean,macro_f1_std\n"
                "vision_cls_attention,16,0.925,0.0,0.92,0.0\n"
                "vision_cls_attention,32,0.95,0.0,0.95,0.0\n"
            )
            (results / "selector_metrics.csv").write_text(
                "selector,K,probe_seed\n"
                "vision_cls_attention,16,0\n"
                "vision_cls_attention,16,1\n"
                "vision_cls_attention,16,2\n"
                "vision_cls_attention,32,0\n"
                "vision_cls_attention,32,1\n"
                "vision_cls_attention,32,2\n"
            )
            (results / "localization_summary.csv").write_text(
                "selector,K,inside_fraction_mean\n"
                "vision_cls_attention,16,0.74\n"
                "vision_cls_attention,32,0.81\n"
            )

            output = results / "phase1_sanity_report.md"
            arguments = [
                str(project_root / "scripts" / "write_phase1_sanity_report.py"),
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(cache),
                "--results-dir",
                str(results),
                "--cub-root",
                str(cub),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", arguments):
                runpy.run_path(arguments[0], run_name="__main__")
            report = output.read_text()
            self.assertIn("Stage-gate status: PASS", report)
            self.assertTrue((results / "vision_cls_part_localization.csv").is_file())


if __name__ == "__main__":
    unittest.main()
