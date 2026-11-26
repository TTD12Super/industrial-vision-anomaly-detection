"""Detector input-validation tests (require torch for import; skipped otherwise)."""

import unittest

try:
    from model.inference.detector import AnomalyDetector  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipIf(not HAS_TORCH, "torch not installed")
class TestDetectorValidation(unittest.TestCase):
    def test_unknown_category_raises_before_model_load(self):
        with self.assertRaises(ValueError):
            AnomalyDetector(category="phone", bank_path="dummy.pt")

    def test_missing_bank_path_and_dir_raises(self):
        with self.assertRaises(ValueError):
            AnomalyDetector(category="bottle")


if __name__ == "__main__":
    unittest.main()
