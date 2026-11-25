"""Unit tests for the reconciled open-set benchmark helpers."""

from __future__ import annotations

import unittest

import numpy as np

from benchmark_screw_open_set import _set_metrics


class OpenSetBenchmarkTests(unittest.TestCase):
    def test_known_metrics_report_acceptance_without_false_known_rate(self) -> None:
        result = _set_metrics(np.asarray([0.9, 0.2, 0.8]), 0.8, known=True)
        self.assertEqual(result["accepted_as_known"], 2)
        self.assertAlmostEqual(result["accepted_rate"], 2 / 3)
        self.assertIsNone(result["false_known_rate"])

    def test_unknown_metrics_report_false_known_rate(self) -> None:
        result = _set_metrics(np.asarray([0.9, 0.2, 0.8]), 0.8, known=False)
        self.assertEqual(result["accepted_as_known"], 2)
        self.assertAlmostEqual(result["false_known_rate"], 2 / 3)
        self.assertAlmostEqual(result["rejection_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
