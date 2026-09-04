import unittest

import torch

from lger.scoring import (
    aggregate_layer_scores,
    logits_to_evidence,
    logits_to_token_mass,
    stable_topk,
)


class ScoringTests(unittest.TestCase):
    def test_margin_is_top_one_minus_top_two(self) -> None:
        logits = torch.tensor([[[1.0, 4.0, 2.0], [5.0, 1.0, 3.0]]])
        torch.testing.assert_close(
            logits_to_evidence(logits, "margin"), torch.tensor([[2.0, 2.0]])
        )

    def test_all_scores_are_finite_and_patch_aligned(self) -> None:
        logits = torch.tensor(
            [
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, -2.0], [1.0, 1.0, 1.0]],
            ]
        )
        for method in ("maxprob", "margin", "negentropy"):
            result = logits_to_evidence(logits, method)
            self.assertEqual(result.shape, (2, 2))
            self.assertTrue(torch.isfinite(result).all())

    def test_mean_layers_aligns_spatial_patch_identity(self) -> None:
        scores = torch.tensor(
            [[0.0, 10.0, 1.0], [0.0, 8.0, 2.0], [0.0, 9.0, 3.0]]
        )
        aggregate, normalized = aggregate_layer_scores(scores, "mean_layers")
        self.assertEqual(normalized.shape, scores.shape)
        self.assertEqual(stable_topk(aggregate, 1).item(), 1)

    def test_topk_uses_spatial_index_to_break_ties(self) -> None:
        indices = stable_topk(torch.tensor([2.0, 2.0, 1.0]), 2)
        self.assertEqual(indices.tolist(), [0, 1])

    def test_token_mass_sums_a_fixed_concept_vocabulary(self) -> None:
        logits = torch.tensor([[0.0, 1.0, 2.0, -1.0]])
        token_ids = torch.tensor([1, 2, 2], dtype=torch.long)
        expected = torch.logsumexp(
            torch.log_softmax(logits, dim=-1)[:, [1, 2]], dim=-1
        )
        torch.testing.assert_close(logits_to_token_mass(logits, token_ids), expected)


if __name__ == "__main__":
    unittest.main()
