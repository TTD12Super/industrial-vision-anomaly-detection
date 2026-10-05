"""Tests for continuous and thresholded anomaly metrics."""

from __future__ import annotations

import unittest

from mvtec_system import evaluate_anomaly_detection


class AnomalyEvaluationTests(unittest.TestCase):
    def make_inputs(self):
        image_scores = [0.1, 0.9, 0.2, 0.8]
        image_labels = [0, 1, 0, 1]
        image_predictions = [0, 1, 0, 1]
        pixel_scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        pixel_labels = [0, 1, 0, 1, 0, 1]
        pixel_predictions = [0, 1, 0, 1, 0, 1]
        return (
            image_scores,
            image_labels,
            pixel_scores,
            pixel_labels,
            image_predictions,
            pixel_predictions,
        )

    def test_ranking_and_thresholded_metrics_are_reported(self) -> None:
        report = evaluate_anomaly_detection(*self.make_inputs())

        self.assertEqual(report["image"]["ranking"]["auroc"], 1.0)
        self.assertEqual(report["image"]["ranking"]["average_precision"], 1.0)
        self.assertEqual(report["image"]["thresholded"]["f1"], 1.0)
        self.assertEqual(report["image"]["thresholded"]["specificity"], 1.0)
        self.assertEqual(report["image"]["thresholded"]["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(report["pixel"]["thresholded"]["dice"], 1.0)
        self.assertEqual(report["pixel"]["thresholded"]["iou"], 1.0)

    def test_ranking_metrics_work_without_threshold_predictions(self) -> None:
        inputs = self.make_inputs()
        report = evaluate_anomaly_detection(*inputs[:4])

        self.assertIn("ranking", report["image"])
        self.assertNotIn("thresholded", report["image"])
        self.assertNotIn("thresholded", report["pixel"])

    def test_shape_and_single_class_errors_are_clear(self) -> None:
        inputs = self.make_inputs()
        with self.assertRaises(ValueError):
            evaluate_anomaly_detection(inputs[0], [0, 1], inputs[2], inputs[3])
        with self.assertRaises(ValueError):
            evaluate_anomaly_detection([0.1, 0.2], [0, 0], [0.1, 0.9], [0, 1])


if __name__ == "__main__":
    unittest.main()
