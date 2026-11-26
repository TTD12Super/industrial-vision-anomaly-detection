"""Feature-bank metadata + persistence tests.

Metadata tests always run (pure). Save/load roundtrip requires torch and is
skipped otherwise; it uses a tiny synthetic bank, never real data.
"""

import tempfile
import unittest
from pathlib import Path

from model.inference.feature_bank import bank_metadata

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestBankMetadata(unittest.TestCase):
    def _metadata(self):
        return bank_metadata(
            "bottle",
            embedding_dimension=768,
            patch_grid=(19, 19),
            padding=(10, 10),
            train_paths=["bottle/train/good/000.png"],
        )

    def test_metadata_contents(self):
        metadata = self._metadata()
        self.assertEqual(metadata["category"], "bottle")
        self.assertEqual(metadata["model_name"], "dinov2_vitb14")
        self.assertEqual(
            metadata["model_revision"], "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        )
        self.assertEqual(metadata["feature_key"], "x_norm_patchtokens")
        self.assertEqual(metadata["image_size"], [256, 256])
        self.assertEqual(metadata["padding"], [10, 10])
        self.assertEqual(metadata["embedding_dimension"], 768)
        self.assertEqual(metadata["patch_grid"], [19, 19])
        self.assertIn("train/good", metadata["sampling"])

    def test_unknown_category_raises(self):
        with self.assertRaises(ValueError):
            bank_metadata(
                "phone",
                embedding_dimension=8,
                patch_grid=(2, 2),
                padding=(0, 0),
                train_paths=[],
            )

    def test_list_train_images_missing_dir(self):
        from model.inference.feature_bank import list_train_images

        with self.assertRaises(FileNotFoundError):
            list_train_images("does/not/exist")


@unittest.skipIf(not HAS_TORCH, "torch not installed")
class TestBankPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        from model.inference.feature_bank import load_feature_bank, save_feature_bank

        metadata = bank_metadata(
            "bottle",
            embedding_dimension=4,
            patch_grid=(1, 2),
            padding=(0, 0),
            train_paths=["a.png"],
        )
        bank = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_bank.pt"
            save_feature_bank(path, bank, metadata)
            loaded, loaded_metadata = load_feature_bank(
                path, expected_metadata=metadata
            )
            self.assertTrue(torch.equal(loaded.cpu(), bank))
            self.assertEqual(loaded_metadata, metadata)

    def test_metadata_mismatch_raises(self):
        from model.inference.feature_bank import load_feature_bank, save_feature_bank

        metadata = bank_metadata(
            "bottle",
            embedding_dimension=4,
            patch_grid=(1, 2),
            padding=(0, 0),
            train_paths=["a.png"],
        )
        bank = torch.zeros(2, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_bank.pt"
            save_feature_bank(path, bank, metadata)
            tampered = dict(metadata)
            tampered["model_name"] = "other"
            with self.assertRaises(ValueError):
                load_feature_bank(path, expected_metadata=tampered)

    def test_missing_file_raises(self):
        from model.inference.feature_bank import load_feature_bank

        with self.assertRaises(FileNotFoundError):
            load_feature_bank("does/not/exist.pt")


if __name__ == "__main__":
    unittest.main()
