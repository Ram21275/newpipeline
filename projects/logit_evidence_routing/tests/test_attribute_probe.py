import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from lger.attribute_probe import (
    binary_auroc,
    load_stage_attribute_dataset,
    mean_pool_probe_patch_contributions,
    random_project_features,
    run_masked_multilabel_probe,
    select_training_f1_threshold,
    shuffle_observed_targets,
)
from lger.stage_cache import config_digest


class AttributeProbeTests(unittest.TestCase):
    def test_binary_auroc_is_tie_aware(self) -> None:
        targets = torch.tensor([0, 0, 1, 1])
        self.assertEqual(binary_auroc(torch.tensor([0.0, 1.0, 2.0, 3.0]), targets), 1.0)
        self.assertEqual(binary_auroc(torch.ones(4), targets), 0.5)
        self.assertTrue(
            torch.isnan(torch.tensor(binary_auroc(torch.arange(3), torch.ones(3))))
        )

    def test_threshold_is_selected_on_training_scores(self) -> None:
        threshold = select_training_f1_threshold(
            torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([0, 1, 1])
        )
        self.assertEqual(threshold, 0.0)

    def test_shuffle_preserves_mask_and_column_counts(self) -> None:
        targets = torch.tensor(
            [[0.0, float("nan")], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )
        shuffled = shuffle_observed_targets(targets, seed=7)
        torch.testing.assert_close(torch.isfinite(shuffled), torch.isfinite(targets))
        for column in range(targets.shape[1]):
            mask = torch.isfinite(targets[:, column])
            self.assertEqual(
                sorted(shuffled[mask, column].tolist()),
                sorted(targets[mask, column].tolist()),
            )

    def test_random_projection_is_seed_deterministic(self) -> None:
        train = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        validation = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        first = random_project_features(train, validation, output_dim=2, seed=4)
        second = random_project_features(train, validation, output_dim=2, seed=4)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertEqual(first[0].shape, (6, 2))

    def test_masked_probe_returns_train_only_thresholds(self) -> None:
        train_features = torch.tensor(
            [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]]
        )
        train_targets = torch.tensor(
            [[0.0, 1.0], [0.0, float("nan")], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        result = run_masked_multilabel_probe(
            train_features,
            train_targets,
            train_features,
            seed=0,
            epochs=100,
            learning_rate=0.05,
            weight_decay=0.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.train_logits.shape, (6, 2))
        self.assertEqual(result.validation_logits.shape, (6, 2))
        self.assertEqual(result.thresholds.shape, (2,))
        self.assertTrue(torch.isfinite(result.thresholds).all())
        self.assertEqual(result.classifier_weight.shape, (2, 1))
        self.assertEqual(result.classifier_bias.shape, (2,))
        self.assertEqual(result.normalization_mean.shape, (1,))

    def test_mean_pool_probe_decomposition_reconstructs_margin(self) -> None:
        patch_features = torch.tensor(
            [
                [[-2.0], [-1.0]],
                [[-0.5], [0.5]],
                [[1.0], [2.0]],
                [[2.0], [3.0]],
            ]
        )
        targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        result = run_masked_multilabel_probe(
            patch_features.mean(dim=1),
            targets,
            patch_features.mean(dim=1),
            seed=3,
            epochs=40,
            learning_rate=0.05,
            weight_decay=0.0,
            device=torch.device("cpu"),
        )
        contributions = mean_pool_probe_patch_contributions(
            patch_features, result, thresholds=result.thresholds
        )
        self.assertEqual(contributions.shape, (4, 2, 1))
        torch.testing.assert_close(
            contributions.sum(dim=1).squeeze(1),
            result.validation_logits.squeeze(1) - result.thresholds[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_cache_reader_loads_only_requested_stage_tensors(self) -> None:
        class FakeSafeOpen:
            def __init__(self, _: Path, **__: object) -> None:
                pass

            def __enter__(self) -> "FakeSafeOpen":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def get_tensor(self, key: str) -> torch.Tensor:
                return tensors[key]

        package = types.ModuleType("safetensors")
        package.safe_open = FakeSafeOpen
        previous = sys.modules.get("safetensors")
        sys.modules["safetensors"] = package
        try:
            with tempfile.TemporaryDirectory() as temporary:
                cache_dir = Path(temporary)
                run_config = {
                    "purpose": "full_240_image_development_pilot_stage_cache",
                    "official_test_images": 0,
                }
                digest = config_digest(run_config)
                records = []
                packed_records = []
                tensors: dict[str, torch.Tensor] = {}
                for offset in range(240):
                    image_id = offset + 1
                    split = "train" if offset < 160 else "val"
                    key = f"image/{image_id:05d}/stages/vision.final"
                    tensors[key] = torch.tensor(
                        [[float(image_id), 1.0], [float(image_id), 3.0]]
                    )
                    records.append(
                        {"image_id": image_id, "development_split": split}
                    )
                    packed_records.append(
                        {
                            "image_id": image_id,
                            "packed_record": {
                                "image": {
                                    "image_id": image_id,
                                    "development_split": split,
                                    "selected_attribute_ids": [1, 2],
                                    "attributes": [
                                        {
                                            "attribute_id": 1,
                                            "name": "has_crown_color::red",
                                            "group": "has_crown_color",
                                            "primary_target": bool(image_id % 2),
                                            "certainty_name": "guess" if image_id == 1 else "probably",
                                        },
                                        {
                                            "attribute_id": 2,
                                            "name": "has_bill_shape::hooked",
                                            "group": "has_bill_shape",
                                            "primary_target": None,
                                            "certainty_name": "not visible",
                                        },
                                    ],
                                },
                                "stages": {
                                    "vision.final": {"__stage_cache_tensor__": key}
                                },
                            },
                        }
                    )
                index = {
                    "config_digest": digest,
                    "complete": True,
                    "records": records,
                    "shards": [
                        {
                            "tensor_path": "shards/stage.safetensors",
                            "metadata_path": "shards/stage.json",
                        }
                    ],
                }
                report = {
                    "config_digest": digest,
                    "status": "PASS",
                    "images": 240,
                    "official_test_images": 0,
                }
                metadata = {
                    "config_digest": digest,
                    "complete": True,
                    "records": packed_records,
                }
                (cache_dir / "shards").mkdir()
                (cache_dir / "run_config.json").write_text(json.dumps(run_config))
                (cache_dir / "index.json").write_text(json.dumps(index))
                (cache_dir / "validation_report.json").write_text(json.dumps(report))
                (cache_dir / "shards" / "stage.json").write_text(json.dumps(metadata))
                dataset = load_stage_attribute_dataset(
                    cache_dir,
                    stages=("vision.final",),
                    pooling=("mean", "max"),
                )
                self.assertEqual(dataset.targets.shape, (240, 2))
                self.assertTrue(torch.isnan(dataset.targets[0, 0]))
                self.assertEqual(dataset.certainty_policy_overrides_masked, 1)
                self.assertEqual(
                    dataset.features[("mean", "vision.final")].shape,
                    (240, 2),
                )
                torch.testing.assert_close(
                    dataset.features[("mean", "vision.final")][0],
                    torch.tensor([1.0, 2.0]),
                )
                torch.testing.assert_close(
                    dataset.features[("max", "vision.final")][0],
                    torch.tensor([1.0, 3.0]),
                )
        finally:
            if previous is None:
                sys.modules.pop("safetensors", None)
            else:
                sys.modules["safetensors"] = previous


if __name__ == "__main__":
    unittest.main()
