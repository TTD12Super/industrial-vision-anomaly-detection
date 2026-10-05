"""Tests for category routing and bank compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from mvtec_system import resolve_category_bank, route_product


class RouterTests(unittest.TestCase):
    def make_inputs(self):
        classifier = {
            "weights": torch.eye(2),
            "bias": torch.zeros(2),
        }
        reference = {"embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        mapping = {"bottle": 0, "cable": 1}
        calibration = {
            "confidence_threshold": 0.5,
            "nearest_global_distance_threshold": 0.5,
        }
        return classifier, reference, mapping, calibration

    def encoder_metadata(self):
        return {
            "model_name": "dinov2_vitb14",
            "model_revision": "revision",
            "patch_feature_key": "x_norm_patchtokens",
            "image_size": [256, 256],
            "padding": [10, 10],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "l2_normalize_patches": True,
            "embedding_dimension": 768,
            "patch_grid": [19, 19],
        }

    def write_compatible_bank(self, root: Path) -> None:
        bank_dir = root / "bottle"
        bank_dir.mkdir(parents=True)
        (bank_dir / "feature_bank.pt").write_bytes(b"bank placeholder")
        metadata = {
            "category": "bottle",
            "model_name": "dinov2_vitb14",
            "model_revision": "revision",
            "feature_key": "x_norm_patchtokens",
            "image_size": [256, 256],
            "padding": [10, 10],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "l2_normalization": True,
            "embedding_dimension": 768,
            "patch_grid": [19, 19],
        }
        (bank_dir / "bank_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_known_product_resolves_compatible_bank(self) -> None:
        classifier, reference, mapping, calibration = self.make_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_compatible_bank(root)
            result = route_product(
                torch.tensor([1.0, 0.0]),
                classifier,
                reference,
                mapping,
                calibration,
                root,
                self.encoder_metadata(),
            )

        self.assertEqual(result.category, "bottle")
        self.assertFalse(result.rejected)
        self.assertEqual(result.bank_status, "compatible")
        self.assertFalse(result.requires_rebuild)

    def test_rejected_product_skips_bank_lookup(self) -> None:
        classifier, reference, mapping, calibration = self.make_inputs()
        calibration["confidence_threshold"] = 0.9
        result = route_product(
            torch.tensor([0.0, 0.0]),
            classifier,
            reference,
            mapping,
            calibration,
            "/path/that/does/not/exist",
        )

        self.assertTrue(result.rejected)
        self.assertEqual(result.category, "unknown_product_category")
        self.assertEqual(result.bank_status, "skipped")
        self.assertIsNone(result.bank_path)

    def test_known_product_reports_missing_bank_without_rebuild(self) -> None:
        classifier, reference, mapping, calibration = self.make_inputs()
        with tempfile.TemporaryDirectory() as directory:
            result = route_product(
                torch.tensor([1.0, 0.0]),
                classifier,
                reference,
                mapping,
                calibration,
                directory,
            )

        self.assertFalse(result.rejected)
        self.assertEqual(result.bank_status, "missing")
        self.assertTrue(result.requires_rebuild)

    def test_bank_metadata_mismatch_is_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_compatible_bank(root)
            metadata_path = root / "bottle" / "bank_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["model_revision"] = "different"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = resolve_category_bank(root, "bottle", self.encoder_metadata())

        self.assertEqual(result.status, "incompatible")
        self.assertTrue(result.requires_rebuild)


if __name__ == "__main__":
    unittest.main()
