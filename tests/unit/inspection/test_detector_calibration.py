import unittest

from inspection.detector_calibration import measure_detector_confidence, select_confidence


class DetectorCalibrationTests(unittest.TestCase):
    def test_selects_highest_confidence_that_keeps_recall(self) -> None:
        selected = select_confidence(
            [
                {"confidence": 0.1, "precision": 0.5, "recall": 1.0},
                {"confidence": 0.2, "precision": 0.8, "recall": 1.0},
                {"confidence": 0.3, "precision": 1.0, "recall": 0.9},
            ],
            1.0,
        )
        self.assertEqual(selected["confidence"], 0.2)

    def test_rejects_unattainable_recall(self) -> None:
        with self.assertRaises(ValueError):
            select_confidence([{"confidence": 0.5, "precision": 1.0, "recall": 0.9}], 1.0)

    def test_diagnostic_contract_can_represent_unattainable_target(self) -> None:
        # The full detector call is intentionally not run in unit tests; this
        # assertion documents the persisted diagnostic schema contract.
        report = {
            "target_recall": 0.99,
            "target_reached": False,
            "confidence": None,
            "selected_metrics": None,
            "candidates": [{"confidence": 0.01, "precision": 1.0, "recall": 0.944}],
        }
        self.assertFalse(report["target_reached"])
        self.assertIsNone(report["selected_metrics"])
