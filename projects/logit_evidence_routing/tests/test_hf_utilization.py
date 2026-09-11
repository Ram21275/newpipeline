import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lger.hf_llava import HfLlavaPatchExtractor
from lger.hf_utilization import (
    HiddenIntervention,
    HfLlavaDecisionRunner,
    normalized_entropy,
)


class FakeTokenizer:
    all_special_ids: list[int] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return {"yes": [1], "no": [2], "very yes": [3, 1]}[text]

    def decode(self, values: list[int], **_: object) -> str:
        reverse = {(1,): "yes", (2,): "no", (3, 1): "very yes", (7,): "yes"}
        return reverse.get(tuple(values), "")

    def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
        return [str(value) for value in values]


class FakeProcessor:
    chat_template = None

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        if "images" in kwargs:
            ids = torch.tensor([[10, 99, 99, 99, 99, 11]])
            result = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            result["pixel_values"] = torch.zeros(1, 3, 4, 4)
            return result
        ids = torch.tensor([[10, 11]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class FakeDecisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.language_model = FakeLanguageModel()
        self.config = SimpleNamespace(
            image_token_index=99, torch_dtype=torch.float32, _commit_hash="a" * 40
        )

    def forward(self, **inputs: object) -> SimpleNamespace:
        ids = inputs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        logits = torch.zeros(1, ids.shape[1], 8)
        logits[..., 1] = 3.0
        logits[..., 2] = 1.0
        length = ids.shape[1]
        attentions = tuple(
            torch.full((1, 2, length, length), 1.0 / length) for _ in range(3)
        )
        return SimpleNamespace(logits=logits, attentions=attentions)

    def generate(self, **inputs: object) -> torch.Tensor:
        ids = inputs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        return torch.cat((ids, torch.tensor([[7]])), dim=1)


class FakeProjectorDecisionModel(FakeDecisionModel):
    def __init__(self) -> None:
        super().__init__()
        self.multi_modal_projector = nn.Identity()
        self.config.vision_config = SimpleNamespace(num_hidden_layers=4)
        self.config.text_config = SimpleNamespace(num_hidden_layers=4)
        self.config.vision_feature_layer = -2

    def forward(self, **inputs: object) -> SimpleNamespace:
        ids = inputs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        projected = self.multi_modal_projector(torch.ones(1, 4, 4))
        logits = torch.zeros(1, ids.shape[1], 8)
        logits[..., 1] = projected.sum()
        logits[..., 2] = 0.0
        return SimpleNamespace(logits=logits, attentions=None)


class HfUtilizationTests(unittest.TestCase):
    def runner(self, positive: str = "yes") -> HfLlavaDecisionRunner:
        base = HfLlavaPatchExtractor(
            FakeDecisionModel(), FakeProcessor(), layer_offset=-2
        )
        return HfLlavaDecisionRunner(base, positive_answer=positive)

    def test_binary_margin_generation_and_visual_attention(self) -> None:
        output = self.runner().evaluate(
            object(), "Question?", include_attention=True, generate=True
        )
        self.assertGreater(output.answer_margin, 0)
        self.assertEqual(output.generated_text, "yes")
        self.assertEqual(output.attention_scores.shape, (4,))
        self.assertAlmostEqual(output.attention_effective_tokens, 4.0, places=5)

    def test_prompt_only_control_has_no_visual_attention(self) -> None:
        output = self.runner().evaluate(
            None, "Question?", include_attention=False, generate=False
        )
        self.assertIsNone(output.generated_text)
        self.assertIsNone(output.attention_scores)
        self.assertGreater(output.answer_margin, 0)

    def test_multitoken_candidate_uses_sequence_log_likelihood(self) -> None:
        output = self.runner("very yes").evaluate(
            object(), "Question?", include_attention=False, generate=False
        )
        self.assertEqual(output.positive_token_ids, (3, 1))
        self.assertTrue(torch.isfinite(torch.tensor(output.answer_margin)))

    def test_normalized_entropy_rejects_negative_scores(self) -> None:
        entropy, effective = normalized_entropy(torch.ones(4))
        self.assertAlmostEqual(effective, 4.0, places=5)
        self.assertGreater(entropy, 0)
        with self.assertRaises(ValueError):
            normalized_entropy(torch.tensor([1.0, -1.0]))

    def test_projector_intervention_changes_only_selected_patch_states(self) -> None:
        base = HfLlavaPatchExtractor(
            FakeProjectorDecisionModel(), FakeProcessor(), layer_offset=-2
        )
        runner = HfLlavaDecisionRunner(base)
        baseline = runner.evaluate(
            object(), "Question?", include_attention=False, generate=False
        )
        changed = runner.evaluate(
            object(),
            "Question?",
            include_attention=False,
            generate=False,
            intervention=HiddenIntervention(
                stage="projector.output",
                patch_indices=torch.tensor([0, 2]),
                replacement=torch.zeros(4),
            ),
        )
        self.assertGreater(baseline.answer_margin, changed.answer_margin)


if __name__ == "__main__":
    unittest.main()
