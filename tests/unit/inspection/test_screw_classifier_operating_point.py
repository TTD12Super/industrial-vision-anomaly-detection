from __future__ import annotations

import unittest

from scripts.select_screw_classifier_operating_point import select_operating_point


class ScrewClassifierOperatingPointTests(unittest.TestCase):
    def test_maximizes_validation_recall_under_good_fp_cap(self) -> None:
        points = [
            {"threshold": 0.8, "validation": {"recall": 0.9, "good_fp": 0.1}},
            {"threshold": 0.4, "validation": {"recall": 1.0, "good_fp": 0.15}},
            {"threshold": 0.2, "validation": {"recall": 1.0, "good_fp": 0.25}},
        ]
        result = select_operating_point(points, 0.9, 0.2)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected"]["threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
