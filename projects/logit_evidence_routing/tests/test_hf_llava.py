import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from lger.hf_llava import (
    HfLlavaPatchExtractor,
    square_grid,
    validate_bitsandbytes_4bit_runtime,
    visual_positions_and_query,
)


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(3)
        self.lm_head = nn.Linear(3, 6, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class FakeVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, *_: object, **__: object) -> SimpleNamespace:
        attention = torch.full((1, 2, 5, 5), 0.2)
        return SimpleNamespace(attentions=(attention, attention))


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.language_model = FakeLanguageModel()
        self.vision_tower = FakeVisionTower()
        self.config = SimpleNamespace(image_token_index=99, torch_dtype=torch.float32)

    def forward(self, **_: object) -> SimpleNamespace:
        sequence = 7
        hidden_states = tuple(
            torch.arange(sequence * 3, dtype=torch.float32).reshape(1, sequence, 3)
            + 100 * layer
            for layer in range(4)
        )
        attentions = []
        for layer in range(3):
            attention = torch.zeros(1, 2, sequence, sequence)
            attention[0, :, 6, 1:5] = torch.tensor(
                [0.1, 0.4, 0.2, 0.3]
            )
            attentions.append(attention + layer)
        return SimpleNamespace(hidden_states=hidden_states, attentions=tuple(attentions))


class FakeProcessor:
    chat_template = None

    def __init__(self) -> None:
        self.image_processor = SimpleNamespace(
            image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0]
        )
        self.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda values: [f"token_{value}" for value in values],
            encode=lambda text, add_special_tokens: [1 if text == " bird" else 2],
        )

    def __call__(self, **_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[10, 99, 99, 99, 99, 11, 12]]),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "pixel_values": torch.full((1, 3, 4, 4), 0.5),
        }


class HfLlavaHelperTests(unittest.TestCase):
    def test_bitsandbytes_preflight_exercises_nf4_kernel(self) -> None:
        functional = SimpleNamespace(
            quantize_4bit=lambda values, quant_type: (values, object()),
            dequantize_4bit=lambda values, state: values,
        )
        fake_bnb = SimpleNamespace(__version__="0.50.2", functional=functional)
        cpu_sample = torch.linspace(-1, 1, steps=128).reshape(2, 64)
        with (
            patch.dict(sys.modules, {"bitsandbytes": fake_bnb}),
            patch.object(torch, "linspace", return_value=cpu_sample),
            patch.object(torch.cuda, "empty_cache"),
        ):
            version = validate_bitsandbytes_4bit_runtime()
        self.assertEqual(version, "0.50.2")

    def test_bitsandbytes_preflight_reports_runtime_versions(self) -> None:
        def fail_quantization(*_: object, **__: object) -> None:
            raise AttributeError("missing native CUDA symbol")

        fake_bnb = SimpleNamespace(
            __version__="0.45.2",
            functional=SimpleNamespace(quantize_4bit=fail_quantization),
        )
        cpu_sample = torch.linspace(-1, 1, steps=128).reshape(2, 64)
        with (
            patch.dict(sys.modules, {"bitsandbytes": fake_bnb}),
            patch.object(torch, "linspace", return_value=cpu_sample),
            patch.object(torch.cuda, "empty_cache"),
            self.assertRaisesRegex(RuntimeError, "installed=0.45.2"),
        ):
            validate_bitsandbytes_4bit_runtime()

    def test_visual_positions_and_last_text_query(self) -> None:
        input_ids = torch.tensor([[10, 99, 99, 99, 11, 12]])
        attention_mask = torch.ones_like(input_ids)
        visual, query = visual_positions_and_query(input_ids, attention_mask, 99)
        self.assertEqual(visual.tolist(), [1, 2, 3])
        self.assertEqual(query, 5)

    def test_unexpanded_image_placeholder_is_rejected(self) -> None:
        input_ids = torch.tensor([[10, 99, 11]])
        with self.assertRaisesRegex(RuntimeError, "did not expand"):
            visual_positions_and_query(input_ids, torch.ones_like(input_ids), 99)

    def test_square_grid_rejects_any_resolution_layout(self) -> None:
        self.assertEqual(square_grid(576), (24, 24))
        with self.assertRaisesRegex(RuntimeError, "square"):
            square_grid(575)

    def test_mock_extraction_keeps_patches_aligned(self) -> None:
        extractor = HfLlavaPatchExtractor(
            FakeModel(), FakeProcessor(), layer_offset=-2, projection_chunk_size=2
        )
        evidence = extractor.extract(object(), "Describe the image briefly.")
        self.assertEqual(evidence.hidden_states.shape, (4, 3))
        self.assertEqual(evidence.grid_size, (2, 2))
        self.assertEqual(evidence.attention_scores.shape, (4,))
        self.assertEqual(evidence.localization_scores, {})
        self.assertEqual(evidence.top_token_ids.shape, (4, 5))
        self.assertEqual(set(evidence.evidence_scores), {"maxprob", "margin", "negentropy"})
        self.assertEqual(evidence.processed_image.shape, (3, 4, 4))

    def test_fixed_concepts_add_a_patch_aligned_token_mass(self) -> None:
        extractor = HfLlavaPatchExtractor(
            FakeModel(),
            FakeProcessor(),
            layer_offset=-2,
            projection_chunk_size=2,
            fixed_concepts=("bird", "birds"),
        )
        evidence = extractor.extract(object(), "Describe the image briefly.")
        self.assertEqual(extractor.concept_token_ids, (1, 2))
        self.assertEqual(evidence.evidence_scores["concept_logprob"].shape, (4,))
        self.assertTrue(torch.isfinite(evidence.evidence_scores["concept_logprob"]).all())

    def test_vision_localizers_are_patch_aligned(self) -> None:
        extractor = HfLlavaPatchExtractor(
            FakeModel(), FakeProcessor(), layer_offset=-2, projection_chunk_size=2
        )
        evidence = extractor.extract(
            object(),
            "Describe the image briefly.",
            include_vision_localizers=True,
        )
        self.assertEqual(
            set(evidence.localization_scores),
            {"vision_cls_attention", "vision_attention_rollout"},
        )
        self.assertEqual(
            {score.shape for score in evidence.localization_scores.values()},
            {(4,)},
        )


if __name__ == "__main__":
    unittest.main()
