import unittest

import torch
from torch import nn

from lger.multimodal_dola import (
    dynamic_layer_correction,
    dynamic_layer_contrast,
    project_query_hidden_states,
    semantic_binary_margin,
    stack_projected_layers,
    top_p_candidate_mask,
)


class MultimodalDoLaTests(unittest.TestCase):
    def test_dynamic_selection_uses_largest_jsd_and_contrasts(self) -> None:
        mature = torch.tensor([[4.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
        candidates = torch.tensor(
            [
                [[3.9, 1.1, -1.0], [-2.0, 0.0, 4.0]],
                [[0.1, 1.9, 1.0], [4.0, -2.0, 0.0]],
            ]
        )
        result = dynamic_layer_contrast(mature, candidates)
        self.assertEqual(result.premature_indices.tolist(), [1, 1])
        selected = candidates[torch.arange(2), result.premature_indices]
        expected = torch.log_softmax(mature, -1) - torch.log_softmax(selected, -1)
        torch.testing.assert_close(result.contrasted_logits, expected)
        self.assertTrue(bool(result.plausibility_mask.all()))

    def test_candidate_mask_and_plausibility_constraint(self) -> None:
        mature = torch.tensor([[5.0, 1.0, -4.0]])
        candidates = torch.tensor([[[4.0, 1.0, -3.0], [-5.0, 1.0, 5.0]]])
        result = dynamic_layer_contrast(
            mature,
            candidates,
            candidate_mask=torch.tensor([[True, False]]),
            plausibility_alpha=0.1,
        )
        self.assertEqual(result.premature_indices.item(), 0)
        self.assertEqual(result.plausibility_mask.tolist(), [[True, False, False]])
        self.assertEqual(result.contrasted_logits[0, 1:].tolist(), [-1000.0, -1000.0])

    def test_projection_and_semantic_margin(self) -> None:
        norm = nn.Identity()
        head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        layer_a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        layer_b = layer_a + 1.0
        projected = project_query_hidden_states(
            layer_a, final_norm=norm, lm_head=head, query_positions=-1
        )
        self.assertEqual(projected.tolist(), [[3.0, 4.0, 7.0]])
        stacked = stack_projected_layers(
            [layer_a, layer_b], final_norm=norm, lm_head=head, query_positions=-1
        )
        self.assertEqual(tuple(stacked.shape), (1, 2, 3))
        margin = semantic_binary_margin(projected, positive_index=2, negative_index=0)
        self.assertEqual(margin.tolist(), [4.0])

    def test_deco_selects_highest_candidate_probability_and_corrects(self) -> None:
        mature = torch.tensor([[2.0, 1.0, 0.0]])
        candidates = torch.tensor(
            [[[2.0, 1.5, 0.0], [0.0, 4.0, -1.0], [3.0, 0.0, 0.0]]]
        )
        token_mask = torch.tensor([[True, True, False]])
        result = dynamic_layer_correction(
            mature,
            candidates,
            alpha=0.6,
            candidate_token_mask=token_mask,
        )
        self.assertEqual(result.anchor_indices.tolist(), [1])
        selected = candidates[:, 1, :]
        max_probability = torch.softmax(selected, dim=-1).amax(dim=-1)
        expected = mature + 0.6 * max_probability.unsqueeze(-1) * selected
        torch.testing.assert_close(result.corrected_logits, expected)
        self.assertEqual(result.candidate_token_mask.tolist(), token_mask.tolist())

    def test_top_p_mask_includes_threshold_crossing_token(self) -> None:
        logits = torch.log(torch.tensor([[0.60, 0.25, 0.10, 0.05]]))
        self.assertEqual(
            top_p_candidate_mask(logits, top_p=0.80).tolist(),
            [[True, True, False, False]],
        )
        full = dynamic_layer_correction(
            logits,
            torch.stack([logits, logits + torch.tensor([[0.0, 0.2, 0.0, 0.0]])], dim=1),
            alpha=0.0,
            top_p=0.80,
        )
        torch.testing.assert_close(full.corrected_logits, logits.float())
        self.assertEqual(
            top_p_candidate_mask(logits, top_p=1.0, top_k=2).tolist(),
            [[True, True, False, False]],
        )

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dynamic_layer_contrast(torch.ones(2, 3), torch.ones(2, 0, 3))
        with self.assertRaises(ValueError):
            semantic_binary_margin(torch.ones(2, 3), positive_index=1, negative_index=1)
        with self.assertRaises(ValueError):
            dynamic_layer_correction(
                torch.ones(2, 3), torch.ones(2, 2, 3), alpha=-0.1
            )


if __name__ == "__main__":
    unittest.main()
