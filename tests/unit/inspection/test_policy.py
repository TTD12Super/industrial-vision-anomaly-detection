import unittest

from inspection.policy import DEFECT, GOOD, REVIEW_REQUIRED, DecisionPolicy


class DecisionPolicyTests(unittest.TestCase):
    def test_binary_label_is_preserved_for_review_band(self) -> None:
        policy = DecisionPolicy(defect_threshold=0.10, review_low_threshold=0.05)

        good = policy.decide(0.01)
        review = policy.decide(0.05)
        defect = policy.decide(0.10)

        self.assertEqual((good.classifier_label, good.operational_status), (GOOD, GOOD))
        self.assertEqual((review.classifier_label, review.operational_status), (GOOD, REVIEW_REQUIRED))
        self.assertEqual((defect.classifier_label, defect.operational_status), (DEFECT, DEFECT))

    def test_invalid_threshold_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DecisionPolicy(defect_threshold=0.10, review_low_threshold=0.10)
