import unittest

import torch

from lger.probe_localization import (
    exact_patch_margin,
    landmark_patch_mask,
    patch_evidence_summary,
)


class ProbeLocalizationTests(unittest.TestCase):
    def test_exact_patch_terms_reconstruct_mean_pooled_probe_margin(self) -> None:
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        mean = torch.tensor([2.0, 3.0])
        scale = torch.tensor([2.0, 4.0])
        weight = torch.tensor([0.5, -1.5])
        contributions = exact_patch_margin(
            states,
            normalization_mean=mean,
            normalization_scale=scale,
            weight=weight,
            bias=torch.tensor(0.7),
            threshold=torch.tensor(0.2),
        )
        expected = ((states.mean(dim=0) - mean) / scale) @ weight + 0.7 - 0.2
        torch.testing.assert_close(contributions.sum(), expected)

    def test_landmark_mask_deduplicates_points_in_one_patch(self) -> None:
        mask = landmark_patch_mask(
            [(1.0, 1.0), (1.5, 1.5), (7.0, 7.0)],
            grid_size=(2, 2),
            image_size=(8, 8),
        )
        self.assertEqual(mask.tolist(), [True, False, False, True])

    def test_summary_records_concentration_and_topk_localization(self) -> None:
        summary = patch_evidence_summary(
            torch.tensor([4.0, 1.0, 0.0, -1.0]),
            grid_size=(2, 2),
            image_size=(8, 8),
            bbox_xyxy=(0.0, 0.0, 4.0, 4.0),
            landmark_points=[(1.0, 1.0)],
            k_values=(1, 2),
        )
        self.assertEqual(summary["top1_indices"], [0])
        self.assertEqual(summary["top1_pointing_game"], 1.0)
        self.assertEqual(summary["top1_part_patch_recall"], 1.0)
        self.assertGreater(summary["bbox_evidence_mass"], 0.5)
        self.assertGreaterEqual(summary["evidence_effective_tokens"], 1.0)


if __name__ == "__main__":
    unittest.main()
