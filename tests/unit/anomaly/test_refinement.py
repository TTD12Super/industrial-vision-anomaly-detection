"""Tests for localization mask cleanup and heatmap gating."""

from __future__ import annotations

import unittest

import numpy as np

from mvtec_system.anomaly.refinement import gate_heatmap, refine_binary_mask


class RefinementTests(unittest.TestCase):
    def test_refine_removes_small_components(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = True
        mask[3:6, 3:6] = True

        refined = refine_binary_mask(mask, min_component_area=4, opening_kernel_size=3)

        self.assertFalse(refined[1, 1])
        self.assertTrue(refined[4, 4])
        self.assertEqual(int(refined.sum()), 9)

    def test_gate_heatmap_preserves_only_masked_values(self) -> None:
        heatmap = np.ones((2, 2), dtype=np.float32)
        mask = np.array([[True, False], [False, True]])

        gated = gate_heatmap(heatmap, mask)

        np.testing.assert_array_equal(gated, [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
