from __future__ import annotations

import unittest

from inspection.classifier_sweep import select_operating_point


class ClassifierSweepTests(unittest.TestCase):
    def test_selection_is_validation_only_and_recall_first(self) -> None:
        points = [
            {"threshold": 0.8, "validation": {"recall": 0.9, "good_fp": 0.01}, "test_diagnostic": {"recall": 1.0, "good_fp": 1.0}},
            {"threshold": 0.4, "validation": {"recall": 1.0, "good_fp": 0.06}, "test_diagnostic": {"recall": 0.1, "good_fp": 0.0}},
            {"threshold": 0.3, "validation": {"recall": 1.0, "good_fp": 0.06}, "test_diagnostic": {"recall": 0.99, "good_fp": 0.0}},
        ]
        result = select_operating_point(points, target_recall=0.90, max_good_fp=0.20)
        self.assertEqual(result["selected"]["threshold"], 0.4)

    def test_no_validation_candidate_passes_both_gates(self) -> None:
        result = select_operating_point(
            [{"threshold": 0.4, "validation": {"recall": 0.89, "good_fp": 0.01}, "test_diagnostic": {}}],
            target_recall=0.90,
            max_good_fp=0.20,
        )
        self.assertEqual(result["status"], "no_validation_gate")
        self.assertIsNone(result["selected"])


if __name__ == "__main__":
    unittest.main()
