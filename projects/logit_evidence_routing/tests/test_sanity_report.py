import json
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
            extraction_config = {
                "schema_version": 2,
                "fixed_concepts": ["bird", "birds"],
                "concept_tokenization_policy": "single_lexical_token_v1",
                "concept_token_ids": [1, 2],
                "concept_tokens": ["▁bird", "▁birds"],
            }
            (cache / "extraction_config.json").write_text(
                json.dumps(extraction_config) + "\n"
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
            (results / "qualitative" / "00002.png").touch()
            (results / "qualitative" / "index.csv").write_text(
                "image_id,class_name,llm_attention_logit_concept_jaccard,figure\n"
                "2,001.a,0.1,00002.png\n"
            )
            (results / "evaluation_config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "manifest": str(manifest.resolve()),
                        "cache_dir": str(cache.resolve()),
                        "cache_config": extraction_config,
                        "selectors": [
                            "random",
                            "vision_cls_attention",
                            "logit_concept",
                            "attention_logit_fusion",
                        ],
                        "k_values": [16, 32],
                        "probe_seeds": [0, 1, 2],
                        "official_test_images_used": 0,
                    }
                )
                + "\n"
            )
            (results / "selector_summary.csv").write_text(
                "selector,K,accuracy_mean,accuracy_std,macro_f1_mean,macro_f1_std\n"
                "vision_cls_attention,16,0.925,0.0,0.92,0.0\n"
                "vision_cls_attention,32,0.95,0.0,0.95,0.0\n"
                "logit_concept,16,0.91,0.0,0.91,0.0\n"
                "logit_concept,32,0.95,0.0,0.95,0.0\n"
                "attention_logit_fusion,16,0.89,0.0,0.89,0.0\n"
                "attention_logit_fusion,32,0.95,0.0,0.95,0.0\n"
            )
            (results / "selector_metrics.csv").write_text(
                "selector,K,probe_seed\n"
                "vision_cls_attention,16,0\n"
                "vision_cls_attention,16,1\n"
                "vision_cls_attention,16,2\n"
                "vision_cls_attention,32,0\n"
                "vision_cls_attention,32,1\n"
                "vision_cls_attention,32,2\n"
                "logit_concept,16,0\n"
                "logit_concept,16,1\n"
                "logit_concept,16,2\n"
                "logit_concept,32,0\n"
                "logit_concept,32,1\n"
                "logit_concept,32,2\n"
                "attention_logit_fusion,16,0\n"
                "attention_logit_fusion,16,1\n"
                "attention_logit_fusion,16,2\n"
                "attention_logit_fusion,32,0\n"
                "attention_logit_fusion,32,1\n"
                "attention_logit_fusion,32,2\n"
            )
            (results / "localization_summary.csv").write_text(
                "selector,K,inside_fraction_mean,pointing_game_mean\n"
                "random,16,0.47,0.55\n"
                "random,32,0.47,0.55\n"
                "vision_cls_attention,16,0.74,0.40\n"
                "vision_cls_attention,32,0.81,0.40\n"
                "logit_concept,16,0.70,0.60\n"
                "logit_concept,32,0.72,0.60\n"
                "attention_logit_fusion,16,0.71,0.60\n"
                "attention_logit_fusion,32,0.73,0.60\n"
            )
            localization_rows = ["image_id,split,selector,K"]
            for image_id in (2, 4):
                for selector in (
                    "vision_cls_attention",
                    "logit_concept",
                    "attention_logit_fusion",
                ):
                    for k in (16, 32):
                        localization_rows.append(f"{image_id},val,{selector},{k}")
            (results / "localization_metrics.csv").write_text(
                "\n".join(localization_rows) + "\n"
            )
            prediction_rows = [
                "selector,K,probe_seed,image_id,target_label,target_class_name"
            ]
            expected_by_id = {2: (1, "001.a"), 4: (2, "002.b")}
            for selector in (
                "vision_cls_attention",
                "logit_concept",
                "attention_logit_fusion",
            ):
                for k in (16, 32):
                    for seed in (0, 1, 2):
                        for image_id, (label, class_name) in expected_by_id.items():
                            prediction_rows.append(
                                f"{selector},{k},{seed},{image_id},{label},{class_name}"
                            )
            (results / "validation_predictions.csv").write_text(
                "\n".join(prediction_rows) + "\n"
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
            self.assertIn("Stage-gate status: PASS WITH ANOMALY", report)
            self.assertIn("Top-K concentration/top-1 pointing mismatch", report)
            self.assertTrue((results / "vision_cls_part_localization.csv").is_file())
            gate = json.loads((results / "phase1_gate.json").read_text())
            self.assertEqual(gate["status"], "PASS WITH ANOMALY")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["concept_token_ids"], [1, 2])

    def test_invalid_concept_cache_is_rejected(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(
            str(project_root / "scripts" / "write_phase1_sanity_report.py")
        )
        errors = namespace["_concept_cache_errors"](
            {
                "fixed_concepts": ["bird", "birds"],
                "concept_tokenization_policy": "single_lexical_token_v1",
                "concept_token_ids": [1, 29871],
                "concept_tokens": ["▁bird", "▁"],
            }
        )
        self.assertTrue(any("29871" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
