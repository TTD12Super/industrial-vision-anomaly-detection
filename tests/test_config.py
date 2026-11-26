"""Config / constants tests (stdlib only; always runnable)."""

import unittest

from model.inference import config


class TestConstants(unittest.TestCase):
    def test_model_identity_matches_notebook(self):
        self.assertEqual(config.MODEL_NAME, "dinov2_vitb14")
        self.assertEqual(
            config.MODEL_REVISION, "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        )
        self.assertEqual(
            config.MODEL_REPOSITORY,
            "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        )
        self.assertEqual(config.FEATURE_KEY, "x_norm_patchtokens")

    def test_inference_hyperparameters_match_notebook(self):
        self.assertEqual(config.IMAGE_SIZE, 256)
        self.assertEqual(config.PROCESSED_SIZE, (256, 256))
        self.assertEqual(config.IMAGENET_MEAN, (0.485, 0.456, 0.406))
        self.assertEqual(config.IMAGENET_STD, (0.229, 0.224, 0.225))
        self.assertTrue(config.L2_NORMALIZE_FEATURES)
        self.assertEqual(config.K_NEIGHBORS, 1)
        self.assertEqual(config.QUERY_CHUNK_SIZE, 512)
        self.assertEqual(config.BANK_CHUNK_SIZE, 16384)

    def test_categories(self):
        self.assertEqual(len(config.CATEGORIES), 15)
        for expected in (
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
            "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
            "transistor", "wood", "zipper",
        ):
            self.assertIn(expected, config.CATEGORIES)


class TestCategoryValidation(unittest.TestCase):
    def test_valid_category_returns_name(self):
        self.assertEqual(config.validate_category("bottle"), "bottle")

    def test_unknown_category_raises(self):
        with self.assertRaises(ValueError):
            config.validate_category("phone")

    def test_empty_category_raises(self):
        with self.assertRaises(ValueError):
            config.validate_category("")

    def test_default_bank_path(self):
        from pathlib import Path

        path = config.default_bank_path("bottle", "artifacts/banks")
        self.assertEqual(path, Path("artifacts/banks/bottle/feature_bank.pt"))
        with self.assertRaises(ValueError):
            config.default_bank_path("unknown", "artifacts/banks")


class TestDeviceSelection(unittest.TestCase):
    def test_get_device_never_raises(self):
        device = config.get_device()
        self.assertIn(str(device), ("cpu", "cuda", "mps"))

    def test_get_bank_device_never_raises(self):
        device = config.get_bank_device()
        self.assertIn(str(device), ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()
