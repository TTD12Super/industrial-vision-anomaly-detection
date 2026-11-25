from __future__ import annotations

from unittest import TestCase

from scripts.record_classifier_sweep import _candidate_rows


class ClassifierSweepAdapterTests(TestCase):
    def test_converts_threshold_points_to_counted_validation_candidates(self) -> None:
        rows = _candidate_rows(
            {
                "points": [
                    {
                        "threshold": 0.4,
                        "validation": {"recall": 0.9333333333333333, "good_fp": 0.06},
                    }
                ]
            },
            defect_count=15,
            good_count=50,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metrics"]["threshold"], 0.4)
        self.assertEqual(rows[0]["metric_numerators"], {
            "defect_true_positive": 14,
            "good_false_alarm_count": 3,
        })
        self.assertEqual(rows[0]["split_role"], "validation")
        self.assertFalse(rows[0]["holdout_used_for_selection"])

    def test_preserves_exact_counts_from_trainer_report(self) -> None:
        rows = _candidate_rows(
            {
                "points": [
                    {
                        "threshold": 0.4,
                        "validation": {
                            "recall": 1.0,
                            "good_fp": 0.06,
                            "defect_true_positive": 15,
                            "good_false_alarm_count": 3,
                        },
                    }
                ]
            },
            defect_count=15,
            good_count=50,
        )
        self.assertEqual(rows[0]["metric_numerators"], {
            "defect_true_positive": 15,
            "good_false_alarm_count": 3,
        })


if __name__ == "__main__":
    import unittest

    unittest.main()
