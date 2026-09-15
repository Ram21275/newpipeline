"""CPU tests for the exploratory guided multi-layer fusion head."""

from __future__ import annotations

import unittest

import torch

from lger.guided_layer_fusion import (
    GuidedLayerAttentionFusion,
    MeanLayerPoolingBaseline,
)


class GuidedLayerFusionTests(unittest.TestCase):
    def test_shapes_weights_and_gradients(self):
        torch.manual_seed(4)
        model = GuidedLayerAttentionFusion(
            (6, 8, 10), query_dim=5, embed_dim=12, num_heads=3
        )
        states = [torch.randn(2, patches, width)
                  for patches, width in ((4, 6), (5, 8), (3, 10))]
        masks = [torch.ones(2, state.shape[1], dtype=torch.bool) for state in states]
        output = model(states, torch.randn(2, 5), valid_patch_masks=masks)
        self.assertEqual(output.logits.shape, (2, 1))
        self.assertEqual(output.layer_attention.shape, (2, 3, 3))
        self.assertEqual([value.shape for value in output.patch_attention], [
            torch.Size((2, 3, 4)), torch.Size((2, 3, 5)), torch.Size((2, 3, 3))
        ])
        self.assertTrue(torch.allclose(
            output.layer_attention.sum(dim=-1), torch.ones(2, 3), atol=1e-5
        ))
        output.logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_masked_patch_values_do_not_change_output(self):
        torch.manual_seed(7)
        model = GuidedLayerAttentionFusion(
            (4, 4), query_dim=3, embed_dim=8, num_heads=2
        ).eval()
        first = torch.randn(1, 4, 4)
        second = torch.randn(1, 3, 4)
        masks = [torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
                 torch.tensor([[1, 1, 1]], dtype=torch.bool)]
        query = torch.randn(1, 3)
        baseline = model([first, second], query, valid_patch_masks=masks).logits
        changed = first.clone()
        changed[:, 2:] = 10000.0
        repeated = model([changed, second], query, valid_patch_masks=masks).logits
        self.assertTrue(torch.allclose(baseline, repeated, atol=1e-5))

    def test_mean_pooling_baseline_supports_heterogeneous_widths(self):
        model = MeanLayerPoolingBaseline((3, 5), embed_dim=4)
        logits = model([torch.randn(2, 4, 3), torch.randn(2, 6, 5)])
        self.assertEqual(logits.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
