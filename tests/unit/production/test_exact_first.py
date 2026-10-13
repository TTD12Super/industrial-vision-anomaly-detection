"""Focused tests for Exact-first production orchestration."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from mvtec_system.production.pipeline import (
    _anomaly_report,
    _classification_report,
    infer_exact_first,
)


def _exact(decision: str) -> dict:
    return {
        "category": "bottle",
        "status": "known_product_category",
        "rejected": False,
        "decision": decision,
        "is_defect": decision == "DEFECT",
        "anomaly_score": 0.8 if decision == "DEFECT" else 0.1,
        "threshold": 0.5,
    }


class ExactFirstTests(TestCase):
    def test_good_short_circuits_localization(self) -> None:
        patchcore = Mock()
        padim = Mock()

        result = infer_exact_first(lambda: _exact("GOOD"), patchcore, padim)

        self.assertEqual(result["decision"], "GOOD")
        self.assertEqual(result["decision_source"], "exact_knn")
        self.assertEqual(result["localization_status"], {"patchcore": "not_run", "padim": "not_run"})
        patchcore.assert_not_called()
        padim.assert_not_called()

    def test_defect_runs_both_localization_branches(self) -> None:
        result = infer_exact_first(
            lambda: _exact("DEFECT"),
            lambda: np.ones((4, 4), dtype=np.float32),
            lambda: np.zeros((4, 4), dtype=bool),
        )

        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["localization_status"]["patchcore"], "available")
        self.assertEqual(result["localization_status"]["padim"], "empty")
        self.assertIsNotNone(result["heatmap"])
        self.assertIsNotNone(result["binary_mask"])

    def test_patchcore_failure_does_not_block_padim(self) -> None:
        padim = Mock(return_value=np.ones((4, 4), dtype=bool))

        def fail_patchcore() -> np.ndarray:
            raise RuntimeError("patchcore unavailable")

        result = infer_exact_first(lambda: _exact("DEFECT"), fail_patchcore, padim)

        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["localization_status"]["patchcore"], "failed")
        self.assertEqual(result["localization_status"]["padim"], "available")
        padim.assert_called_once()

    def test_padim_failure_does_not_remove_patchcore_heatmap(self) -> None:
        patchcore = Mock(return_value=np.ones((4, 4), dtype=np.float32))

        def fail_padim() -> np.ndarray:
            raise RuntimeError("padim unavailable")

        result = infer_exact_first(lambda: _exact("DEFECT"), patchcore, fail_padim)

        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["localization_status"]["patchcore"], "available")
        self.assertEqual(result["localization_status"]["padim"], "failed")
        self.assertIsNotNone(result["heatmap"])
        self.assertIsNone(result["binary_mask"])

    def test_exact_failure_is_unavailable_and_never_good(self) -> None:
        patchcore = Mock()
        padim = Mock()

        def fail_exact() -> dict:
            raise RuntimeError("exact bank missing")

        result = infer_exact_first(fail_exact, patchcore, padim)

        self.assertIsNone(result["decision"])
        self.assertEqual(result["decision_status"], "unavailable")
        self.assertEqual(result["localization_status"], {"patchcore": "not_run", "padim": "not_run"})
        patchcore.assert_not_called()
        padim.assert_not_called()

    def test_unknown_category_does_not_fallback_to_a_bank(self) -> None:
        patchcore = Mock()
        padim = Mock()
        exact = {
            "category": None,
            "status": "unknown_product_category",
            "rejected": True,
            "reason": "confidence below threshold",
        }

        result = infer_exact_first(lambda: exact, patchcore, padim)

        self.assertIsNone(result["decision"])
        self.assertEqual(result["status"], "unknown_product_category")
        self.assertEqual(result["decision_status"], "unavailable")
        patchcore.assert_not_called()
        padim.assert_not_called()

    def test_missing_exact_artifact_is_explicitly_unavailable(self) -> None:
        result = infer_exact_first(
            lambda: {
                "category": "bottle",
                "status": "known_product_category",
                "rejected": False,
                "bank_status": "missing",
                "reason": "feature bank is missing",
            },
            Mock(),
            Mock(),
        )

        self.assertIsNone(result["decision"])
        self.assertEqual(result["decision_status"], "unavailable")
        self.assertEqual(result["primary_error"], "feature bank is missing")

    def test_metrics_use_prediction_not_ground_truth_as_prediction(self) -> None:
        report = _anomaly_report(
            [
                {
                    "anomaly_evaluated": True,
                    "anomaly_score": 0.9,
                    "ground_truth_is_defect": False,
                    "decision_is_defect": True,
                    "heatmap": np.ones((2, 2), dtype=np.float32),
                    "binary_mask": np.zeros((2, 2), dtype=bool),
                    "mask": np.zeros((2, 2), dtype=bool),
                }
            ]
        )

        self.assertEqual(report["image"]["thresholded"]["accuracy"], 0.0)
        self.assertEqual(report["image"]["thresholded"]["f1"], 0.0)

    def test_unavailable_decision_does_not_break_category_report(self) -> None:
        report, _ = _classification_report(
            [
                {
                    "true_category": "bottle",
                    "category": None,
                    "rejected": False,
                    "ground_truth_is_defect": True,
                }
            ],
            ["bottle"],
        )

        self.assertEqual(report["accepted_known_predictions"], 0)
        self.assertEqual(report["top1_accuracy_all"], 0.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
