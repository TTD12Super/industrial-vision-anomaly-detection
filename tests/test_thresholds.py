"""Tests for normal-only anomaly threshold calibration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from mvtec_system import (
    calibrate_anomaly_thresholds,
    calibrate_quantile_threshold,
    save_anomaly_thresholds,
)


class ThresholdTests(unittest.TestCase):
    def test_quantile_threshold_records_acceptance_metadata(self) -> None:
        result = calibrate_quantile_threshold([1.0, 2.0, 3.0, 4.0], 0.75)

        self.assertEqual(result["threshold"], 3.25)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["quantile_interpolation"], "linear")
        self.assertEqual(result["observed_acceptance_rate"], 0.75)

    def test_image_and_pixel_protocol_is_normal_only(self) -> None:
        result = calibrate_anomaly_thresholds(
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            torch.tensor([[0.01, 0.02], [0.03, 0.04]]),
        )

        self.assertEqual(result["calibration_split"], "known_normal_validation")
        self.assertFalse(result["protocol"]["test_labels_used"])
        self.assertEqual(result["image"]["sample_count"], 4)
        self.assertEqual(result["pixel"]["sample_count"], 4)

    def test_invalid_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_quantile_threshold([], 0.99)
        with self.assertRaises(ValueError):
            calibrate_quantile_threshold([1.0, float("nan")], 0.99)
        with self.assertRaises(ValueError):
            calibrate_quantile_threshold([1.0], 0.0)

    def test_thresholds_can_be_saved(self) -> None:
        result = calibrate_anomaly_thresholds([1.0, 2.0], [0.1, 0.2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            save_anomaly_thresholds(path, result)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["artifact_version"], "v1")
        self.assertEqual(saved["pixel"]["threshold"], result["pixel"]["threshold"])


if __name__ == "__main__":
    unittest.main()
