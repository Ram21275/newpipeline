"""CPU-only invariants for the Qwen2.5-VL replication adapter."""

from __future__ import annotations

import unittest

import torch

from lger.hf_qwen import qwen_visual_positions_and_query


class QwenPositionTests(unittest.TestCase):
    def test_visual_positions_and_query_respect_attention_mask(self):
        ids = torch.tensor([[10, 151655, 151655, 20, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        visual, query = qwen_visual_positions_and_query(ids, mask, 151655)
        self.assertEqual(visual.tolist(), [1, 2])
        self.assertEqual(query, 3)

    def test_missing_visual_tokens_and_malformed_batch_fail(self):
        with self.assertRaisesRegex(RuntimeError, "did not expand"):
            qwen_visual_positions_and_query(
                torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long), 99
            )
        with self.assertRaisesRegex(ValueError, "one aligned"):
            qwen_visual_positions_and_query(
                torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long), 1
            )


if __name__ == "__main__":
    unittest.main()
