import unittest

import torch

from lger.qwen_layout import expected_merged_visual_tokens, qwen_layout_assertions


class QwenLayoutTests(unittest.TestCase):
    def test_expected_tokens_respects_dynamic_grid_and_merger(self) -> None:
        grid = torch.tensor([[1, 24, 24], [2, 12, 16]])
        self.assertEqual(
            expected_merged_visual_tokens(grid, spatial_merge_size=2),
            (144, 96),
        )

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expected_merged_visual_tokens([[1, 3, 3]], spatial_merge_size=2)
        with self.assertRaises(ValueError):
            expected_merged_visual_tokens([[1, 4]], spatial_merge_size=2)

    def test_layout_assertions_report_all_invariants(self) -> None:
        checks = qwen_layout_assertions(
            input_image_tokens=144,
            expected_image_tokens=(144,),
            visual_output_tokens=144,
            input_sequence_tokens=160,
            hidden_sequence_tokens=160,
        )
        self.assertTrue(all(checks.values()))
        with self.assertRaises(RuntimeError):
            qwen_layout_assertions(
                input_image_tokens=143,
                expected_image_tokens=(144,),
                visual_output_tokens=144,
                input_sequence_tokens=160,
                hidden_sequence_tokens=159,
            )


if __name__ == "__main__":
    unittest.main()
