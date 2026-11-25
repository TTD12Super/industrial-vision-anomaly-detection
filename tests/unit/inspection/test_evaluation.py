"""Unit tests for integrated object matching and false-positive accounting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from inspection.evaluation import evaluate_end_to_end
from inspection.interfaces import KNOWN_PRODUCT, REVIEW_PRODUCT, UNKNOWN_PRODUCT
from inspection.policy import DEFECT, GOOD, REVIEW_REQUIRED


BOX_A = [[0, 0], [10, 0], [10, 10], [0, 10]]
BOX_B = [[20, 0], [30, 0], [30, 10], [20, 10]]
BOX_EXTRA = [[40, 0], [50, 0], [50, 10], [40, 10]]
BOX_UNKNOWN = [[60, 0], [70, 0], [70, 10], [60, 10]]
BOX_REVIEW = [[80, 0], [90, 0], [90, 10], [80, 10]]


class FakePipeline:
    def __init__(self, results: list[dict], recognizer_active: bool = False) -> None:
        self.results = iter(results)
        self.product_recognizer = object() if recognizer_active else None

    def inspect_path(self, _path: Path, inspection_id: str) -> dict:
        result = dict(next(self.results))
        result["inspection_id"] = inspection_id
        return result


class EvaluationMatchingTests(TestCase):
    config = SimpleNamespace(product_key="screw", dataset_path=lambda _path: Path("unused.png"))

    @staticmethod
    def image(image_id: str = "image-1") -> dict:
        return {"image_id": image_id, "split": "test", "relative_path": f"{image_id}.png"}

    @staticmethod
    def truth(object_id: str, box: list[list[int]], label: str, product_key: str = "screw") -> dict:
        return {
            "object_id": object_id,
            "image_id": "image-1",
            "obb_xyxyxyxy": box,
            "quality_label": label,
            "product_key": product_key,
        }

    @staticmethod
    def prediction(
        box: list[list[int]],
        status: str,
        label: str | None = None,
        recognition_status: str | None = None,
    ) -> dict:
        return {
            "obb_xyxyxyxy": box,
            "operational_status": status,
            "classifier_label": label or status,
            "product_recognition_status": recognition_status,
        }

    def evaluate(self, truths: list[dict], predictions: list[dict], recognizer_active: bool = False, iou_threshold: float = 0.5) -> dict:
        images = [self.image()]
        with patch("inspection.evaluation.load_dataset_records", return_value=(images, truths)):
            with TemporaryDirectory() as directory:
                return evaluate_end_to_end(
                    self.config,
                    FakePipeline([{"objects": predictions}], recognizer_active),
                    Path(directory) / "report.json",
                    iou_threshold=iou_threshold,
                )

    def test_matching_maximizes_valid_matches_not_truth_iteration_order(self) -> None:
        truths = [
            self.truth("gt-1", [[0, 0], [10, 0], [10, 10], [0, 10]], "GOOD"),
            self.truth("gt-2", [[8, 0], [18, 0], [18, 10], [8, 10]], "DEFECT"),
        ]
        predictions = [
            self.prediction([[5, 0], [15, 0], [15, 10], [5, 10]], GOOD),
            self.prediction([[0, 0], [3, 0], [3, 10], [0, 10]], GOOD),
        ]
        report = self.evaluate(truths, predictions, iou_threshold=0.2)
        self.assertEqual(report["integrated_metrics"]["matched_objects"], 2)
        self.assertEqual(report["integrated_metrics"]["missed_ground_truth_objects"], 0)

    def test_prediction_extra_is_unmatched_false_positive(self) -> None:
        report = self.evaluate(
            [self.truth("defect-1", BOX_A, "DEFECT")],
            [self.prediction(BOX_A, DEFECT), self.prediction(BOX_EXTRA, DEFECT)],
        )
        metrics = report["integrated_metrics"]
        self.assertEqual(metrics["matched_objects"], 1)
        self.assertEqual(metrics["missed_ground_truth_objects"], 0)
        self.assertEqual(metrics["unmatched_predictions"], 1)
        self.assertEqual(metrics["unmatched_false_positives"], 1)
        self.assertEqual(report["per_image"][0]["unmatched_false_positives"], 1)

    def test_ground_truth_miss_is_counted_as_missed_and_fn(self) -> None:
        report = self.evaluate([self.truth("defect-1", BOX_A, "DEFECT")], [])
        metrics = report["integrated_metrics"]
        self.assertEqual(metrics["missed_objects"], 1)
        self.assertEqual(metrics["missed_defect"], 1)
        self.assertEqual(metrics["missed_ground_truth_objects"], 1)
        self.assertEqual(metrics["unmatched_predictions"], 0)
        self.assertEqual(metrics["end_to_end_defect_recall"], 0.0)

    def test_latency_summary_is_reported_when_pipeline_provides_timings(self) -> None:
        truths = [self.truth("defect-1", BOX_A, "DEFECT")]
        predictions = [self.prediction(BOX_A, DEFECT)]
        images = [self.image()]
        result = {
            "objects": [dict(predictions[0], latency_ms={"total": 7.0})],
            "latency_ms": {"detector": 2.0, "total": 11.0},
        }
        with patch("inspection.evaluation.load_dataset_records", return_value=(images, truths)):
            with TemporaryDirectory() as directory:
                report = evaluate_end_to_end(
                    self.config,
                    FakePipeline([result]),
                    Path(directory) / "report.json",
                )

        self.assertEqual(report["latency_ms"]["image_total"]["p50"], 11.0)
        self.assertEqual(report["latency_ms"]["object_total"]["p95"], 7.0)

    def test_multiple_objects_preserve_good_alarm_and_matching_counts(self) -> None:
        report = self.evaluate(
            [
                self.truth("good-1", BOX_A, "GOOD"),
                self.truth("defect-1", BOX_B, "DEFECT"),
            ],
            [
                self.prediction(BOX_A, REVIEW_REQUIRED, GOOD),
                self.prediction(BOX_B, DEFECT),
                self.prediction(BOX_EXTRA, DEFECT),
            ],
        )
        metrics = report["integrated_metrics"]
        self.assertEqual(metrics["matched_objects"], 2)
        self.assertEqual(metrics["matched_good"], 1)
        self.assertEqual(metrics["matched_defect"], 1)
        self.assertEqual(metrics["good_flagged"], 1)
        self.assertEqual(metrics["good_false_alarm_count"], 1)
        self.assertEqual(metrics["unmatched_false_positives"], 1)

    def test_no_predictions_miss_all_ground_truth_objects(self) -> None:
        report = self.evaluate(
            [
                self.truth("good-1", BOX_A, "GOOD"),
                self.truth("defect-1", BOX_B, "DEFECT"),
            ],
            [],
        )
        metrics = report["integrated_metrics"]
        self.assertEqual(metrics["ground_truth_objects"], 2)
        self.assertEqual(metrics["predicted_objects"], 0)
        self.assertEqual(metrics["missed_ground_truth_objects"], 2)
        self.assertEqual(metrics["missed_good"], 1)
        self.assertEqual(metrics["missed_defect"], 1)
        self.assertEqual(metrics["unmatched_false_positives"], 0)

    def test_closed_world_keeps_existing_metrics_and_marks_recognition_inactive(self) -> None:
        report = self.evaluate(
            [self.truth("good-1", BOX_A, "GOOD"), self.truth("defect-1", BOX_B, "DEFECT")],
            [
                self.prediction(BOX_A, GOOD, GOOD),
                self.prediction(BOX_B, DEFECT, DEFECT),
            ],
        )

        metrics = report["integrated_metrics"]
        self.assertFalse(report["product_recognizer_active"])
        self.assertEqual(metrics["end_to_end_defect_recall"], 1.0)
        self.assertEqual(metrics["matched_known_product_count"], 0)
        self.assertEqual(metrics["matched_unknown_product_count"], 0)
        self.assertEqual(metrics["matched_review_product_count"], 0)
        self.assertEqual(metrics["unknown_false_accept_count"], 0)
        self.assertIsNone(metrics["known_product_recognition_coverage"])
        self.assertIsNone(metrics["known_only_defect_recall"])

    def test_recognition_metrics_separate_unknown_from_quality(self) -> None:
        truths = [
            self.truth("good-1", BOX_A, "GOOD"),
            self.truth("defect-1", BOX_B, "DEFECT"),
            self.truth("unknown-rejected", BOX_EXTRA, "DEFECT", "other_product"),
        ]
        predictions = [
            self.prediction(BOX_A, GOOD, GOOD, KNOWN_PRODUCT),
            self.prediction(BOX_B, DEFECT, DEFECT, KNOWN_PRODUCT),
            self.prediction(BOX_EXTRA, "NOT_ASSESSED", None, UNKNOWN_PRODUCT),
        ]

        metrics = self.evaluate(truths, predictions, recognizer_active=True)["integrated_metrics"]

        self.assertEqual(metrics["matched_known_product_count"], 2)
        self.assertEqual(metrics["matched_unknown_product_count"], 1)
        self.assertEqual(metrics["known_product_objects"], 2)
        self.assertEqual(metrics["unknown_product_objects"], 1)
        self.assertEqual(metrics["unknown_false_accept_count"], 0)
        self.assertEqual(metrics["unknown_rejection_rate"], 1.0)
        self.assertEqual(metrics["product_recognition_coverage"], 1.0)
        self.assertEqual(metrics["known_product_recognition_coverage"], 1.0)
        self.assertEqual(metrics["known_only_defect_recall"], 1.0)

    def test_active_recognizer_reports_review_and_unknown_false_accept(self) -> None:
        truths = [
            self.truth("good-1", BOX_A, "GOOD"),
            self.truth("defect-1", BOX_B, "DEFECT"),
            self.truth("unknown-accepted", BOX_EXTRA, "GOOD", "other_product"),
            self.truth("unknown-rejected", BOX_UNKNOWN, "DEFECT", "other_product"),
            self.truth("known-review", BOX_REVIEW, "GOOD"),
        ]
        predictions = [
            self.prediction(BOX_A, GOOD, GOOD, KNOWN_PRODUCT),
            self.prediction(BOX_B, DEFECT, DEFECT, KNOWN_PRODUCT),
            self.prediction(BOX_EXTRA, GOOD, GOOD, KNOWN_PRODUCT),
            self.prediction(BOX_UNKNOWN, "NOT_ASSESSED", None, UNKNOWN_PRODUCT),
            self.prediction(BOX_REVIEW, REVIEW_REQUIRED, None, REVIEW_PRODUCT),
        ]

        metrics = self.evaluate(truths, predictions, recognizer_active=True)["integrated_metrics"]

        self.assertEqual(metrics["matched_known_product_count"], 3)
        self.assertEqual(metrics["matched_unknown_product_count"], 1)
        self.assertEqual(metrics["matched_review_product_count"], 1)
        self.assertEqual(metrics["unknown_false_accept_count"], 1)
        self.assertEqual(metrics["unknown_false_accept_rate"], 0.5)
        self.assertEqual(metrics["unknown_rejection_rate"], 0.5)
        self.assertAlmostEqual(metrics["known_product_recognition_coverage"], 2 / 3)
        self.assertEqual(metrics["known_only_defect_recall"], 1.0)
