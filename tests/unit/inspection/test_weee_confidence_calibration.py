from __future__ import annotations

import unittest

from scripts.calibrate_weee_detector_confidence import (
    filter_predictions,
    score_predictions,
    select_operating_point,
)


class WEEEConfidenceCalibrationTests(unittest.TestCase):
    def test_filter_predictions_keeps_only_selected_confidence(self) -> None:
        predictions = [{"confidence": 0.2}, {"confidence": 0.5}]
        self.assertEqual(filter_predictions(predictions, 0.5), [{"confidence": 0.5}])

    def test_selection_preserves_baseline_iou_and_prefers_low_unmatched_rate(self) -> None:
        rows = [
            {
                "confidence_threshold": 0.25,
                "ground_truth_recall": 0.80,
                "matched_crop_iou_mean": 0.70,
                "unmatched_prediction_rate": 0.20,
            },
            {
                "confidence_threshold": 0.10,
                "ground_truth_recall": 0.90,
                "matched_crop_iou_mean": 0.70,
                "unmatched_prediction_rate": 0.40,
            },
            {
                "confidence_threshold": 0.15,
                "ground_truth_recall": 0.895,
                "matched_crop_iou_mean": 0.70,
                "unmatched_prediction_rate": 0.10,
            },
        ]
        result = select_operating_point(rows)
        self.assertEqual(result["selected"]["confidence_threshold"], 0.15)

    def test_selection_rejects_recall_gain_that_increases_unmatched_rate(self) -> None:
        rows = [
            {
                "confidence_threshold": 0.25,
                "ground_truth_recall": 0.80,
                "matched_crop_iou_mean": 0.70,
                "unmatched_prediction_rate": 0.20,
            },
            {
                "confidence_threshold": 0.10,
                "ground_truth_recall": 0.90,
                "matched_crop_iou_mean": 0.70,
                "unmatched_prediction_rate": 0.40,
            },
        ]
        result = select_operating_point(rows)
        self.assertEqual(result["selected"]["confidence_threshold"], 0.25)

    def test_score_predictions_accounts_for_misses_and_unmatched(self) -> None:
        records = [{"ground_truth": [{"corners": [[0, 0], [10, 0], [10, 10], [0, 10]]}]}]
        predictions = [[
            {"confidence": 0.9, "corners": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"confidence": 0.9, "corners": [[20, 20], [30, 20], [30, 30], [20, 30]]},
        ]]
        metrics = score_predictions(records, predictions, 0.5, 0.01)
        self.assertEqual(metrics["matched_count"], 1)
        self.assertEqual(metrics["unmatched_prediction_count"], 1)
        self.assertEqual(metrics["unmatched_ground_truth_count"], 0)


if __name__ == "__main__":
    unittest.main()
