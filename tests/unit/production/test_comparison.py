"""Tests for detector-comparison aggregation."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from mvtec_system.production.comparison import run_detector_comparison, summarize_comparison
from mvtec_system.production.pipeline import _anomaly_report
from mvtec_system.production.pipeline import ProductionConfig


def _record(detector: str, offset: float) -> dict:
    return {
        "detector": detector,
        "run_seconds": 10.0 + offset,
        "summary": {
            "dataset_name": "tiny",
            "report_dir": f"reports/{detector}",
            "comparison_provenance": {
                "classification_manifest_sha256": "shared-manifest",
                "test_manifest_sha256": "shared-test",
                "dataset_provenance_sha256": "shared-dataset",
                "dataset_validation_sha256": "shared-validation",
            },
            "classification": {"total_test_images": 2, "top1_accuracy_all": 1.0},
            "anomaly": {
                "image": {"ranking": {"auroc": 0.8, "average_precision": 0.9}},
                "pixel": {
                    "ranking": {"auroc": 0.7, "average_precision": 0.6},
                    "thresholded": {"f1": 0.5},
                },
            },
            "performance": {
                "inference": {
                    "total_seconds": 2.0 + offset,
                    "mean_seconds": 1.0 + offset,
                    "p95_seconds": 1.2 + offset,
                }
            },
        },
    }


class ComparisonTests(unittest.TestCase):
    def test_requires_all_three_detectors_and_sums_combined_latency(self) -> None:
        result = summarize_comparison(
            [_record("exact_knn", 0.0), _record("patchcore", 1.0), _record("padim", 2.0)]
        )

        self.assertTrue(result["same_test_split"])
        self.assertTrue(result["split_identity_verified"])
        self.assertEqual(result["roles"]["heatmap_baseline"], "patchcore")
        self.assertEqual(result["combined"]["total_inference_seconds"], 9.0)
        self.assertFalse(result["combined"]["official_fusion_enabled"])

    def test_rejects_mismatched_test_splits(self) -> None:
        records = [_record(name, 0.0) for name in ("exact_knn", "patchcore", "padim")]
        records[2]["summary"]["classification"]["total_test_images"] = 3

        with self.assertRaises(ValueError):
            summarize_comparison(records)

    def test_rejects_mismatched_test_manifest(self) -> None:
        records = [_record(name, 0.0) for name in ("exact_knn", "patchcore", "padim")]
        records[2]["summary"]["comparison_provenance"]["test_manifest_sha256"] = "other"

        with self.assertRaises(ValueError):
            summarize_comparison(records)

    def test_rejects_duplicate_detectors(self) -> None:
        with self.assertRaises(ValueError):
            summarize_comparison([_record("exact_knn", 0.0), _record("exact_knn", 1.0), _record("padim", 2.0)])

    def test_anomaly_report_records_image_and_pixel_counts(self) -> None:
        rows = [
            {
                "anomaly_evaluated": True,
                "anomaly_score": 0.9,
                "decision_is_defect": True,
                "ground_truth_is_defect": True,
                "heatmap": np.array([[0.9, 0.1]], dtype=np.float32),
                "binary_mask": np.array([[1, 0]], dtype=np.uint8),
                "mask": np.array([[1, 0]], dtype=np.uint8),
            },
            {
                "anomaly_evaluated": True,
                "anomaly_score": 0.1,
                "decision_is_defect": False,
                "ground_truth_is_defect": False,
                "heatmap": np.array([[0.1, 0.9]], dtype=np.float32),
                "binary_mask": np.array([[0, 1]], dtype=np.uint8),
                "mask": np.array([[0, 0]], dtype=np.uint8),
            },
        ]

        report = _anomaly_report(rows)
        self.assertEqual(report["image"]["thresholded"]["counts"]["true_positive"], 1)
        self.assertEqual(report["image"]["thresholded"]["counts"]["true_negative"], 1)
        self.assertEqual(report["pixel"]["thresholded"]["counts"]["false_positive"], 1)

    def test_comparison_runs_independent_orchestration(self) -> None:
        summaries = {
            name: _record(name, float(index))["summary"]
            for index, name in enumerate(("exact_knn", "patchcore", "padim"))
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "mvtec_system.production.comparison.run_production",
                side_effect=lambda config: summaries[config.detector],
            ) as run:
                run_detector_comparison(
                    ProductionConfig(),
                    output_root=Path(directory),
                )

        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            {call.args[0].orchestration for call in run.call_args_list},
            {"independent"},
        )


if __name__ == "__main__":
    unittest.main()
