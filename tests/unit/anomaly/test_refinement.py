"""Tests for the DINO-affinity and mask-cleanup refinements."""

from __future__ import annotations

import unittest

import numpy as np

from mvtec_system import postprocess_mask, propagate_scores, refine_heatmap


class RefinementTests(unittest.TestCase):
    def test_eta_zero_is_score_preserving(self) -> None:
        scores = np.array([0.1, 0.9, 0.2], dtype=np.float32)
        features = np.eye(3, dtype=np.float32)
        actual = propagate_scores(scores, features, top_k=2, eta=0.0)
        self.assertTrue(np.array_equal(actual, scores))

    def test_affinity_propagation_is_finite_and_bounded(self) -> None:
        scores = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        actual = propagate_scores(scores, features, top_k=2, temperature=0.2, eta=1.0)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(actual[0], 0.0)
        self.assertLessEqual(float(actual.max()), 1.0)

    def test_opening_removes_single_pixel_component(self) -> None:
        heatmap = np.zeros((9, 9), dtype=np.float32)
        heatmap[4, 4] = 1.0
        mask = postprocess_mask(heatmap, 0.5, "open3")
        self.assertFalse(mask.any())

    def test_continuous_heatmap_opening_preserves_region_and_removes_spike(self) -> None:
        heatmap = np.zeros((15, 15), dtype=np.float32)
        heatmap[4:11, 4:11] = 0.5
        heatmap[1, 1] = 1.0

        actual = refine_heatmap(heatmap, "heatmap_open5")

        self.assertEqual(actual.shape, heatmap.shape)
        self.assertEqual(float(actual[1, 1]), 0.0)
        self.assertGreater(float(actual[7, 7]), 0.0)


if __name__ == "__main__":
    unittest.main()
