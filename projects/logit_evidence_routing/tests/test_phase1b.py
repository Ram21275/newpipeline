import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lger.cub import PilotRecord, save_pilot_manifest
from lger.phase1b import feature_key, localizer_score_maps, refresh_concept_features


class PhaseOneBTests(unittest.TestCase):
    def test_score_maps_are_aligned_and_fusion_is_fixed(self) -> None:
        evidence = SimpleNamespace(
            attention_scores=torch.tensor([1.0, 3.0, 2.0]),
            localization_scores={
                "vision_cls_attention": torch.tensor([0.3, 0.2, 0.1]),
                "vision_attention_rollout": torch.tensor([0.1, 0.4, 0.2]),
            },
            evidence_scores={
                "maxprob": torch.tensor([0.2, 0.3, 0.1]),
                "margin": torch.tensor([0.0, 2.0, 1.0]),
                "negentropy": torch.tensor([-3.0, -1.0, -2.0]),
                "concept_logprob": torch.tensor([-4.0, -2.0, -3.0]),
            },
        )
        score_maps = localizer_score_maps(evidence)
        self.assertEqual({scores.numel() for scores in score_maps.values()}, {3})
        self.assertEqual(
            torch.argsort(score_maps["attention_logit_fusion"], descending=True)[0],
            1,
        )

    def test_feature_keys_separate_selection_and_probe_seeds(self) -> None:
        self.assertEqual(feature_key("global_all"), "global_all")
        self.assertEqual(feature_key("logit_concept", 16), "logit_concept_k16")
        self.assertEqual(
            feature_key("random", 16, 2), "random_k16_selection2"
        )
        with self.assertRaisesRegex(ValueError, "selection seed"):
            feature_key("random", 16)

    def test_refresh_concept_features_changes_only_concept_derivatives(self) -> None:
        original_random = torch.tensor([2, 1])
        record = {
            "patch_hidden_states": torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]
            ),
            "score_maps": {
                "llm_attention": torch.tensor([0.0, 1.0, 2.0]),
                "logit_concept": torch.zeros(3),
                "attention_logit_fusion": torch.zeros(3),
            },
            "features": {"random_k2_selection0": torch.tensor([9.0, 9.0])},
            "selections": {"random_k2_selection0": original_random.clone()},
        }
        refresh_concept_features(record, torch.tensor([3.0, 2.0, 1.0]), [2])
        self.assertEqual(record["selections"]["logit_concept_k2"].tolist(), [0, 1])
        self.assertEqual(
            record["selections"]["random_k2_selection0"].tolist(),
            original_random.tolist(),
        )
        torch.testing.assert_close(
            record["features"]["logit_concept_k2"], torch.tensor([0.5, 1.0])
        )

    def test_benchmark_writes_probe_and_localization_outputs(self) -> None:
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
            (root / "cache" / "extraction_config.json").write_text(
                '{"schema_version": 2, "concept_tokenization_policy": '
                '"single_lexical_token_v1", "concept_token_ids": [1, 2], '
                '"concept_tokens": ["▁bird", "▁birds"]}\n'
            )
            for record in records:
                feature = (
                    torch.tensor([-1.0, 0.0])
                    if record.label == 1
                    else torch.tensor([1.0, 0.0])
                )
                torch.save(
                    {
                        "schema_version": 2,
                        "image_id": record.image_id,
                        "relative_path": record.relative_path,
                        "label": record.label,
                        "class_name": record.class_name,
                        "split": record.split,
                        "grid_size": (2, 2),
                        "processed_image_size": (4, 4),
                        "bbox_xyxy_model": (0.0, 0.0, 2.0, 4.0),
                        "features": {
                            "random_k2_selection0": feature,
                            "logit_concept_k2": feature,
                            "global_all": feature,
                        },
                        "selections": {
                            "random_k2_selection0": torch.tensor([0, 1]),
                            "llm_attention_k2": torch.tensor([0, 2]),
                            "logit_concept_k2": torch.tensor([0, 2]),
                        },
                    },
                    cache_records / f"{record.image_id:05d}.pt",
                )

            output = root / "results"
            arguments = [
                str(project_root / "scripts" / "run_phase1b_benchmark.py"),
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(root / "cache"),
                "--output-dir",
                str(output),
                "--k",
                "2",
                "--selection-seeds",
                "0",
                "--probe-seeds",
                "0",
                "--selectors",
                "random",
                "logit_concept",
                "global_all",
                "--epochs",
                "2",
            ]
            with patch.object(sys, "argv", arguments):
                runpy.run_path(arguments[0], run_name="__main__")
            for filename in (
                "selector_metrics.csv",
                "selector_summary.csv",
                "validation_predictions.csv",
                "localization_metrics.csv",
                "localization_summary.csv",
                "evaluation_config.json",
                "notes.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_benchmark_rejects_legacy_concept_cache(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pilot.csv"
            save_pilot_manifest(
                [
                    PilotRecord(1, "a/1.jpg", 1, "a", "train"),
                    PilotRecord(2, "a/2.jpg", 1, "a", "val"),
                ],
                manifest,
                {"official_test_images_used": 0},
            )
            cache = root / "cache"
            cache.mkdir()
            (cache / "extraction_config.json").write_text(
                '{"schema_version": 2, "concept_token_ids": [1, 29871]}\n'
            )
            arguments = [
                str(project_root / "scripts" / "run_phase1b_benchmark.py"),
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(cache),
                "--output-dir",
                str(root / "results"),
                "--selectors",
                "logit_concept",
            ]
            with (
                patch.object(sys, "argv", arguments),
                self.assertRaisesRegex(RuntimeError, "predates"),
            ):
                runpy.run_path(arguments[0], run_name="__main__")


if __name__ == "__main__":
    unittest.main()
