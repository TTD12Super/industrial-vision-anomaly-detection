"""Tests for CLI argument and JSON output helpers."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from predict import build_parser, make_output_payload


class PredictCliTests(unittest.TestCase):
    def test_parser_requires_only_image(self) -> None:
        args = build_parser().parse_args(["--image", "input.png"])

        self.assertEqual(args.image, Path("input.png"))
        self.assertIsNone(args.image_thresholds)

    def test_output_payload_is_json_friendly_and_contains_thresholds(self) -> None:
        result = {
            "category": "bottle",
            "confidence": 0.99,
            "nearest_global_distance": 1.2,
            "anomaly_score": 0.3,
            "label": "GOOD",
            "patch_scores": np.array([0.1]),
            "anomaly_map": np.zeros((256, 256)),
            "binary_mask": np.zeros((256, 256), dtype=bool),
        }
        payload = make_output_payload(
            result,
            {"confidence_threshold": 0.9, "nearest_global_distance_threshold": 2.0},
            Path("output/heatmap.png"),
        )

        self.assertEqual(payload["classification_confidence"], 0.99)
        self.assertEqual(payload["reject_thresholds"]["minimum_product_confidence"], 0.5)
        self.assertEqual(payload["reject_thresholds"]["confidence"], 0.9)
        self.assertEqual(payload["anomaly_map_shape"], [256, 256])
        self.assertNotIn("patch_scores", payload)


if __name__ == "__main__":
    unittest.main()
