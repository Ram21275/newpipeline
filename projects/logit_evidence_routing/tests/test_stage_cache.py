import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from lger.hf_llava import HfLlavaPatchExtractor
from lger.hf_stage_cache import HfLlavaStageExtractor
from lger.stage_cache import (
    REQUIRED_STAGE_NAMES,
    config_digest,
    patch_centers,
    resolve_stage_plan,
    stage_metadata,
    validate_stage_record,
    write_or_validate_config,
)


class FakeVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, *_: object, **__: object) -> SimpleNamespace:
        hidden_states = tuple(
            torch.full((1, 5, 3), float(index)) for index in range(5)
        )
        return SimpleNamespace(hidden_states=hidden_states)


class FakeProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 12, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, sequence_length: int) -> SimpleNamespace:
        hidden_states = tuple(
            torch.arange(sequence_length * 4, dtype=torch.float32).reshape(
                1, sequence_length, 4
            )
            + 100 * index
            for index in range(5)
        )
        logits = torch.arange(
            sequence_length * 12, dtype=torch.float32
        ).reshape(1, sequence_length, 12)
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)


class FakeStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vision_tower = FakeVisionTower()
        self.multi_modal_projector = FakeProjector()
        self.language_model = FakeLanguageModel()
        self.config = SimpleNamespace(
            image_token_index=99,
            torch_dtype=torch.float32,
            _commit_hash="a" * 40,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
            vision_config=SimpleNamespace(num_hidden_layers=4),
            text_config=SimpleNamespace(num_hidden_layers=4),
        )

    def generate(self, **inputs: object) -> SimpleNamespace:
        pixels = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        assert isinstance(pixels, torch.Tensor)
        assert isinstance(input_ids, torch.Tensor)
        vision = self.vision_tower(pixels)
        projected = self.multi_modal_projector(vision.hidden_states[-2][:, 1:, :])
        self.language_model(int(input_ids.shape[1]))
        self.language_model(1)
        answer = torch.tensor([[7, 8]], dtype=torch.long)
        sequences = torch.cat((input_ids, answer), dim=1)
        self.last_projected = projected
        return SimpleNamespace(sequences=sequences)


class FakeTokenizer:
    all_special_ids: list[int] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [1]

    def decode(self, values: list[int], **_: object) -> str:
        return "mock answer"

    def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
        return [f"token_{value}" for value in values]


class FakeProcessor:
    chat_template = None

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(
            do_resize=True,
            size={"shortest_edge": 4},
            resample=3,
            do_center_crop=True,
            crop_size={"height": 4, "width": 4},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        )

    def __call__(self, **_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[10, 99, 99, 99, 99, 11, 12]]),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "pixel_values": torch.full((1, 3, 4, 4), 0.5),
        }


def mock_record() -> dict[str, object]:
    plan = resolve_stage_plan(
        vision_num_hidden_layers=4,
        llm_num_hidden_layers=4,
        vision_feature_layer=-2,
    )
    stages = {
        name: torch.ones(4, 3 if name.startswith("vision.") else 4).half()
        for name in REQUIRED_STAGE_NAMES
    }
    centers, normalized = patch_centers((2, 2), (4, 4))
    return {
        "schema_version": 1,
        "complete": True,
        "config_digest": "digest",
        "image": {
            "official_split": "train",
            "attributes": [
                {
                    "attribute_id": 1,
                    "state": "present",
                    "primary_target": True,
                }
            ],
            "selected_attribute_ids": [1],
        },
        "spatial": {
            "patch_count": 4,
            "grid_size": [2, 2],
            "processed_image_size_hw": [4, 4],
            "patch_centers_model_xy": centers,
            "patch_centers_normalized_xy": normalized,
            "visual_token_positions": torch.tensor([1, 2, 3, 4]),
        },
        "stages": stages,
        "stage_metadata": stage_metadata(stages, plan),
        "generation": {
            "answer_token_ids": torch.tensor([7, 8]),
            "answer_token_strings": ["token_7", "token_8"],
            "answer_token_logits": torch.ones(2, 12).half(),
        },
    }


class StageCacheTests(unittest.TestCase):
    def test_stage_resolution_preserves_projector_source(self) -> None:
        plan = resolve_stage_plan(
            vision_num_hidden_layers=24,
            llm_num_hidden_layers=32,
            vision_feature_layer=-2,
        )
        self.assertEqual(
            [plan[f"vision.{name}"] for name in ("early", "middle", "late", "final")],
            [6, 12, 23, 24],
        )
        self.assertEqual(
            [plan[f"llm.{name}"] for name in ("early", "middle", "late", "final")],
            [8, 16, 31, 32],
        )

    def test_patch_centers_are_row_major_and_normalized(self) -> None:
        centers, normalized = patch_centers((2, 2), (4, 8))
        torch.testing.assert_close(
            centers,
            torch.tensor([[2.0, 1.0], [6.0, 1.0], [2.0, 3.0], [6.0, 3.0]]),
        )
        torch.testing.assert_close(normalized[0], torch.tensor([0.25, 0.25]))

    def test_config_resumption_rejects_any_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_config.json"
            first = {"model": "a", "prompt": "fixed"}
            digest = write_or_validate_config(path, first)
            self.assertEqual(digest, config_digest(first))
            self.assertEqual(json.loads(path.read_text()), first)
            self.assertEqual(write_or_validate_config(path, first), digest)
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                write_or_validate_config(path, {"model": "a", "prompt": "changed"})

    def test_record_validation_rejects_patch_misalignment(self) -> None:
        record = mock_record()
        result = validate_stage_record(record, expected_config_digest="digest")
        self.assertEqual(result["stage_count"], 9)
        record["stages"]["llm.middle"] = torch.ones(3, 4).half()
        with self.assertRaisesRegex(RuntimeError, "lost patch correspondence"):
            validate_stage_record(record, expected_config_digest="digest")

    def test_record_validation_rejects_semantic_misalignment(self) -> None:
        record = mock_record()
        record["image"]["attributes"][0]["primary_target"] = False
        with self.assertRaisesRegex(RuntimeError, "primary target disagree"):
            validate_stage_record(record, expected_config_digest="digest")

        record = mock_record()
        record["spatial"]["visual_token_positions"] = torch.tensor([4, 3, 2, 1])
        with self.assertRaisesRegex(RuntimeError, "positions are not ordered"):
            validate_stage_record(record, expected_config_digest="digest")

        record = mock_record()
        record["generation"]["answer_token_strings"] = ["token_7"]
        with self.assertRaisesRegex(RuntimeError, "do not align"):
            validate_stage_record(record, expected_config_digest="digest")

    def test_mocked_generation_captures_all_aligned_stages(self) -> None:
        base = HfLlavaPatchExtractor(
            FakeStageModel(), FakeProcessor(), layer_offset=-2
        )
        extractor = HfLlavaStageExtractor(base)
        result = extractor.extract(object(), "Describe.", max_new_tokens=2)
        self.assertEqual(set(result.stages), set(REQUIRED_STAGE_NAMES))
        self.assertEqual({tensor.shape[0] for tensor in result.stages.values()}, {4})
        self.assertEqual(result.stage_plan["vision.late"], 3)
        self.assertEqual(
            extractor.spatial_preprocessing["crop_size"],
            {"height": 4, "width": 4},
        )
        self.assertEqual(result.answer_token_ids.tolist(), [7, 8])
        self.assertEqual(result.answer_token_logits.shape, (2, 12))
        torch.testing.assert_close(
            result.answer_token_logits[0],
            torch.arange(72, 84, dtype=torch.float16),
        )
        torch.testing.assert_close(
            result.answer_token_logits[1],
            torch.arange(12, dtype=torch.float16),
        )
        self.assertEqual(result.answer_text, "mock answer")


if __name__ == "__main__":
    unittest.main()
