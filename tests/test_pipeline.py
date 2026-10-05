"""Tests for the one-forward unified inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from mvtec_system import (
    DINOv2Config,
    DINOv2Features,
    calibrate_category_image_thresholds,
    calibrate_category_pixel_thresholds,
    infer_image,
)


class FakeEncoder:
    def __init__(self, accepted: bool = True) -> None:
        self.config = DINOv2Config()
        self.patch_grid = (1, 2)
        self.padded_size = (256, 256)
        self.calls = 0
        self.accepted = accepted

    def metadata(self):
        return {
            "model_name": self.config.model_name,
            "model_revision": self.config.model_revision,
            "patch_feature_key": self.config.patch_feature_key,
            "global_feature_key": self.config.global_feature_key,
            "image_size": list(self.config.image_size),
            "mean": list(self.config.mean),
            "std": list(self.config.std),
            "l2_normalize_patches": True,
            "padding": [0, 0],
            "patch_grid": list(self.patch_grid),
            "embedding_dimension": 2,
        }

    def forward(self, images):
        del images
        self.calls += 1
        global_embedding = torch.tensor([[5.0, 0.0]]) if self.accepted else torch.tensor([[0.0, 5.0]])
        patch_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        return DINOv2Features(global_embedding, patch_embeddings)


class PipelineTests(unittest.TestCase):
    def make_artifacts(self, root: Path):
        encoder = FakeEncoder()
        bank_directory = root / "bottle"
        bank_directory.mkdir()
        metadata = {
            "category": "bottle",
            **encoder.metadata(),
            "feature_key": encoder.config.patch_feature_key,
            "l2_normalization": True,
        }
        torch.save(
            {"features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "metadata": metadata},
            bank_directory / "feature_bank.pt",
        )
        (bank_directory / "bank_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return encoder

    def make_common_inputs(self):
        classifier = {"weights": torch.eye(2), "bias": torch.zeros(2)}
        reference = {"embeddings": torch.tensor([[5.0, 0.0], [0.0, 5.0]])}
        mapping = {"bottle": 0, "cable": 1}
        calibration = {
            "confidence_threshold": 0.9,
            "nearest_global_distance_threshold": 1.0,
        }
        return classifier, reference, mapping, calibration

    def test_pipeline_forwards_encoder_once_and_returns_anomaly_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_path)
            encoder = self.make_artifacts(root)
            image_thresholds = calibrate_category_image_thresholds({"bottle": [0.1, 0.2]}, 0.5)
            pixel_thresholds = calibrate_category_pixel_thresholds({"bottle": [0.1, 0.2]}, 0.5)
            result = infer_image(
                image_path,
                encoder,
                *self.make_common_inputs(),
                root,
                image_thresholds,
                pixel_thresholds,
            )

        self.assertEqual(encoder.calls, 1)
        self.assertEqual(result["category"], "bottle")
        self.assertEqual(result["bank_status"], "compatible")
        self.assertEqual(result["label"], "GOOD")
        self.assertEqual(result["anomaly_score"], 0.0)
        self.assertEqual(tuple(result["anomaly_map"].shape), (256, 256))
        self.assertEqual(result["binary_mask"].dtype, bool)

    def test_rejected_product_stops_before_bank_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_path)
            encoder = self.make_artifacts(root)
            encoder.accepted = False
            classifier, reference, mapping, calibration = self.make_common_inputs()
            calibration["confidence_threshold"] = 0.99999
            result = infer_image(
                image_path,
                encoder,
                classifier,
                reference,
                mapping,
                calibration,
                root,
            )

        self.assertTrue(result["rejected"])
        self.assertEqual(result["category"], "unknown_product_category")
        self.assertIsNone(result["anomaly_score"])


if __name__ == "__main__":
    unittest.main()
