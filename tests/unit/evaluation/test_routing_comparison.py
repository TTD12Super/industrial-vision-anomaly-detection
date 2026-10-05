"""Tests for oracle versus automatic routing comparison."""

from __future__ import annotations

import unittest

from mvtec_system import compare_routing_modes


class RoutingComparisonTests(unittest.TestCase):
    def test_reports_automatic_minus_oracle_deltas(self) -> None:
        labels = [0, 1, 0, 1]
        pixel_labels = [0, 1, 0, 1, 0, 1]
        report = compare_routing_modes(
            [0.1, 0.9, 0.2, 0.8],
            [0.1, 0.8, 0.7, 0.2],
            labels,
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [0.1, 0.8, 0.7, 0.2, 0.6, 0.3],
            pixel_labels,
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
        )

        self.assertEqual(report["oracle"]["image"]["ranking"]["auroc"], 1.0)
        self.assertLess(report["automatic_minus_oracle"]["image"]["auroc_delta"], 0.0)
        self.assertIn("f1_delta", report["automatic_minus_oracle"]["image"])
        self.assertIn("iou_delta", report["automatic_minus_oracle"]["pixel"])

    def test_different_sample_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compare_routing_modes(
                [0.1, 0.9],
                [0.1],
                [0, 1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0, 1],
            )


if __name__ == "__main__":
    unittest.main()
