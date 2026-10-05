"""Tests for anomaly-map interpolation and heatmap rendering."""

from __future__ import annotations

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mvtec_system import make_anomaly_maps, plot_anomaly_examples


class AnomalyMapTests(unittest.TestCase):
    def test_interpolation_and_crop_match_notebook(self) -> None:
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        actual = make_anomaly_maps(scores, (2, 2), (4, 4), (3, 3))
        expected = F.interpolate(
            scores.reshape(1, 1, 2, 2), size=(4, 4), mode="bilinear", align_corners=False
        )[0, 0, :3, :3].numpy()

        self.assertTrue(np.allclose(actual, expected))
        self.assertEqual(actual.shape, (1, 3, 3))

    def test_invalid_patch_scores_are_reported(self) -> None:
        with self.assertRaises(ValueError):
            make_anomaly_maps(torch.zeros(1, 3), (2, 2), (4, 4), (3, 3))
        invalid = torch.zeros(1, 4)
        invalid[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            make_anomaly_maps(invalid, (2, 2), (4, 4), (3, 3))

    def test_heatmap_comparison_has_four_panels(self) -> None:
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        masks = [np.zeros((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
        maps = np.stack([np.zeros((8, 8)), np.ones((8, 8))])
        figure = plot_anomaly_examples(
            "bottle", images, masks, maps, [0.1, 0.9], [False, True]
        )
        self.assertEqual(len(figure.axes), 9)
        with tempfile.TemporaryDirectory() as directory:
            figure.savefig(f"{directory}/examples.png")
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
