"""Tests for validation-only open-set calibration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from mvtec_system import (
    UNKNOWN_PRODUCT_CATEGORY,
    calibrate_open_set_thresholds,
    nearest_global_distances,
    save_calibration,
)


class CalibrationTests(unittest.TestCase):
    def make_inputs(self):
        classifier = {
            "weights": torch.eye(2),
            "bias": torch.zeros(2),
        }
        reference = {
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "split": "train",
        }
        validation = {
            "embeddings": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.1, 0.0], [0.0, 0.9]]
            ),
            "split": "validation",
        }
        return classifier, reference, validation, {"bottle": 0, "cable": 1}

    def test_nearest_global_distance_is_exact_and_chunked(self) -> None:
        distances = nearest_global_distances(
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            chunk_size=1,
        )

        self.assertTrue(torch.allclose(distances, torch.tensor([0.0, 1.0])))

    def test_calibration_persists_reject_policy(self) -> None:
        classifier, reference, validation, mapping = self.make_inputs()
        calibration = calibrate_open_set_thresholds(
            classifier,
            reference,
            validation,
            mapping,
            target_acceptance_rate=0.75,
        )

        self.assertEqual(calibration["unknown_category"], UNKNOWN_PRODUCT_CATEGORY)
        self.assertEqual(calibration["reference_split"], "train")
        self.assertEqual(calibration["calibration_split"], "validation")
        self.assertGreaterEqual(calibration["distance_only_acceptance_rate"], 0.75)
        self.assertFalse(calibration["calibration_protocol"]["unknown_examples_used"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration(path, calibration)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["nearest_global_distance_threshold"], calibration["nearest_global_distance_threshold"])

    def test_target_acceptance_rate_must_be_positive(self) -> None:
        classifier, reference, validation, mapping = self.make_inputs()
        with self.assertRaises(ValueError):
            calibrate_open_set_thresholds(
                classifier, reference, validation, mapping, target_acceptance_rate=0.0
            )


if __name__ == "__main__":
    unittest.main()
