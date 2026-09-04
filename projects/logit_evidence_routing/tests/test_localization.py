import unittest

import torch

from lger.localization import (
    attention_rollout,
    final_cls_attention,
    patch_centers_in_box,
    selection_localization_metrics,
    selection_part_metrics,
)


class LocalizationTests(unittest.TestCase):
    def test_one_layer_rollout_includes_residual_attention(self) -> None:
        attention = torch.tensor(
            [
                [
                    [
                        [0.2, 0.3, 0.5],
                        [0.1, 0.8, 0.1],
                        [0.4, 0.1, 0.5],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(
            final_cls_attention((attention,)), torch.tensor([0.3, 0.5])
        )
        torch.testing.assert_close(
            attention_rollout((attention,)), torch.tensor([0.15, 0.25])
        )

    def test_patch_grid_localization_metrics(self) -> None:
        bbox_mask = patch_centers_in_box(
            (2, 2), (4, 4), (0.0, 0.0, 2.0, 4.0)
        )
        self.assertEqual(bbox_mask.tolist(), [True, False, True, False])
        metrics = selection_localization_metrics(torch.tensor([0, 1]), bbox_mask)
        self.assertEqual(metrics["inside_fraction"], 0.5)
        self.assertEqual(metrics["bbox_patch_recall"], 0.5)
        self.assertEqual(metrics["bbox_patch_iou"], 1 / 3)
        self.assertEqual(metrics["pointing_game"], 1.0)

    def test_duplicate_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selection_localization_metrics(
                torch.tensor([0, 0]), torch.tensor([True, False])
            )

    def test_part_metrics_cover_visible_part_cells(self) -> None:
        metrics = selection_part_metrics(
            torch.tensor([5, 10]),
            [(1.5, 1.5), (2.5, 2.5)],
            grid_size=(4, 4),
            image_size=(4, 4),
        )
        self.assertEqual(metrics["part_patch_recall"], 1.0)
        self.assertEqual(metrics["top1_part_hit"], 1.0)
        self.assertEqual(metrics["top1_nearest_part_distance_patches"], 0.0)


if __name__ == "__main__":
    unittest.main()
