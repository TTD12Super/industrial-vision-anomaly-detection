"""Tests for detector-comparison aggregation."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from mvtec_system.production.comparison import run_detector_comparison, summarize_comparison
from mvtec_system.production.pipeline import ProductionConfig


def _record(detector: str, offset: float) -> dict:
    return {
        "detector": detector,
        "run_seconds": 10.0 + offset,
        "summary": {
            "dataset_name": "tiny",
            "report_dir": f"reports/{detector}",
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
        self.assertEqual(result["roles"]["heatmap_baseline"], "patchcore")
        self.assertEqual(result["combined"]["total_inference_seconds"], 9.0)
        self.assertFalse(result["combined"]["official_fusion_enabled"])

    def test_rejects_mismatched_test_splits(self) -> None:
        records = [_record(name, 0.0) for name in ("exact_knn", "patchcore", "padim")]
        records[2]["summary"]["classification"]["total_test_images"] = 3

        with self.assertRaises(ValueError):
            summarize_comparison(records)

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
                    ProductionConfig(heatmap_refiner="heatmap_open5"),
                    output_root=Path(directory),
                )

        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            {call.args[0].orchestration for call in run.call_args_list},
            {"independent"},
        )
        refiners = {
            call.args[0].detector: call.args[0].heatmap_refiner
            for call in run.call_args_list
        }
        self.assertEqual(
            refiners,
            {"exact_knn": "none", "patchcore": "heatmap_open5", "padim": "heatmap_open5"},
        )


if __name__ == "__main__":
    unittest.main()
