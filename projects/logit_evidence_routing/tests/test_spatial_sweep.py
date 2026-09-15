"""Tests for fixed-encoder multi-scale visual-token regions."""

from __future__ import annotations

import unittest

from lger.spatial_sweep import pooled_grid_regions, valid_window_centers, window_indices


class SpatialSweepTests(unittest.TestCase):
    def test_complete_windows_have_exact_token_count_and_no_wraparound(self):
        centers = valid_window_centers((24, 24), window_width=5, dilation=2)
        self.assertEqual(len(centers), 16 * 16)
        indices = window_indices((24, 24), centers[0], window_width=5, dilation=2)
        self.assertEqual(len(indices), 25)
        self.assertEqual(len(set(indices)), 25)
        self.assertTrue(all(0 <= value < 576 for value in indices))

    def test_border_center_is_rejected_instead_of_clipped(self):
        with self.assertRaises(ValueError):
            window_indices((24, 24), 0, window_width=3)

    def test_pooled_regions_are_an_exact_partition(self):
        regions = pooled_grid_regions((24, 24), (3, 3))
        self.assertEqual(len(regions), 9)
        self.assertTrue(all(len(region) == 64 for region in regions))
        flattened = [value for region in regions for value in region]
        self.assertEqual(sorted(flattened), list(range(576)))


if __name__ == "__main__":
    unittest.main()
