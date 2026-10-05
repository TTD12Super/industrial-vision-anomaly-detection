"""Tests for the shared DINOv2 wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from mvtec_system import DINOv2Config, SharedDINOv2, image_tensor, select_device


class FakeBackbone(torch.nn.Module):
    """Small deterministic backbone for fast contract tests."""

    def __init__(self, missing_key: str | None = None, wrong_patch_count: bool = False):
        super().__init__()
        self.patch_embed = SimpleNamespace(patch_size=14)
        self.embed_dim = 768
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.missing_key = missing_key
        self.wrong_patch_count = wrong_patch_count
        self.forward_calls = 0

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        batch_size = images.shape[0]
        patch_count = 360 if self.wrong_patch_count else 361
        patch_values = torch.arange(
            patch_count * self.embed_dim,
            dtype=torch.float32,
            device=images.device,
        ).reshape(1, patch_count, self.embed_dim)
        patch_values = patch_values.expand(batch_size, -1, -1)
        global_values = torch.arange(self.embed_dim, dtype=torch.float32, device=images.device)
        global_values = global_values.expand(batch_size, -1)
        outputs = {
            "x_norm_patchtokens": patch_values,
            "x_norm_clstoken": global_values,
        }
        if self.missing_key:
            outputs.pop(self.missing_key, None)
        return outputs


class SharedDINOv2ContractTests(unittest.TestCase):
    def make_encoder(self, **backbone_options) -> tuple[SharedDINOv2, FakeBackbone]:
        backbone = FakeBackbone(**backbone_options)
        encoder = SharedDINOv2(backbone=backbone, device="cpu")
        return encoder, backbone

    def make_batch(self, batch_size: int = 2) -> torch.Tensor:
        return torch.zeros(batch_size, 3, 256, 256)

    def test_one_forward_returns_global_and_patch_features(self):
        encoder, backbone = self.make_encoder()
        features = encoder.forward(self.make_batch())

        self.assertEqual(tuple(features.global_embedding.shape), (2, 768))
        self.assertEqual(tuple(features.patch_embeddings.shape), (2, 361, 768))
        self.assertEqual(backbone.forward_calls, 1)
        self.assertTrue(torch.isfinite(features.global_embedding).all().item())
        self.assertTrue(torch.isfinite(features.patch_embeddings).all().item())

    def test_patch_features_are_l2_normalized(self):
        encoder, _ = self.make_encoder()
        features = encoder.forward(self.make_batch())
        norms = torch.linalg.vector_norm(features.patch_embeddings, dim=-1)

        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_padding_and_patch_geometry_match_baseline(self):
        encoder, _ = self.make_encoder()

        self.assertEqual(encoder.patch_size, (14, 14))
        self.assertEqual(encoder.padding, (10, 10))
        self.assertEqual(encoder.padded_size, (266, 266))
        self.assertEqual(encoder.patch_grid, (19, 19))
        self.assertEqual(encoder.patch_count, 361)

    def test_backbone_is_frozen_and_in_eval_mode(self):
        encoder, backbone = self.make_encoder()

        self.assertFalse(backbone.training)
        self.assertEqual(sum(p.numel() for p in encoder.backbone.parameters() if p.requires_grad), 0)

    def test_metadata_contains_compatibility_settings(self):
        encoder, _ = self.make_encoder()
        metadata = encoder.metadata()

        self.assertEqual(metadata["model_name"], "dinov2_vitb14")
        self.assertEqual(metadata["model_revision"], "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
        self.assertEqual(metadata["patch_feature_key"], "x_norm_patchtokens")
        self.assertEqual(metadata["global_feature_key"], "x_norm_clstoken")
        self.assertEqual(metadata["image_size"], [256, 256])
        self.assertEqual(metadata["patch_grid"], [19, 19])
        self.assertEqual(metadata["embedding_dimension"], 768)

    def test_input_shape_is_validated(self):
        encoder, _ = self.make_encoder()

        with self.assertRaises(ValueError):
            encoder.forward(torch.zeros(3, 256, 256))
        with self.assertRaises(ValueError):
            encoder.forward(torch.zeros(1, 3, 224, 224))

        invalid = self.make_batch()
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            encoder.forward(invalid)

    def test_missing_global_feature_is_reported(self):
        encoder, _ = self.make_encoder(missing_key="x_norm_clstoken")

        with self.assertRaises(KeyError):
            encoder.forward(self.make_batch(1))

    def test_unexpected_patch_shape_is_reported(self):
        encoder, _ = self.make_encoder(wrong_patch_count=True)

        with self.assertRaises(ValueError):
            encoder.forward(self.make_batch(1))

    def test_image_tensor_keeps_preprocessing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "good.png"
            pixels = np.zeros((256, 256, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path)
            tensor = image_tensor(path)

        self.assertEqual(tuple(tensor.shape), (3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all().item())

    def test_image_tensor_rejects_wrong_size_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_size = Path(directory) / "wrong_size.png"
            Image.new("RGB", (128, 128)).save(wrong_size)
            with self.assertRaises(ValueError):
                image_tensor(wrong_size)

            wrong_mode = Path(directory) / "wrong_mode.png"
            Image.new("L", (256, 256)).save(wrong_mode)
            with self.assertRaises(ValueError):
                image_tensor(wrong_mode)

    def test_device_selector_returns_torch_device(self):
        self.assertIsInstance(select_device(), torch.device)


class CachedDINOv2IntegrationTests(unittest.TestCase):
    def test_cached_model_produces_expected_features(self):
        artifact_root = Path("artifacts/dinov2_knn")
        checkpoint = artifact_root / "torch_hub/checkpoints/dinov2_vitb14_pretrain.pth"
        repository = artifact_root / "torch_hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        if not checkpoint.is_file() or not repository.is_dir():
            self.skipTest("local pinned DINOv2 cache is not available")

        path = Path("datasets/mvtec_ad_processed/bottle/train/good/000.png")
        encoder = SharedDINOv2(device=select_device())
        features = encoder.forward(image_tensor(path).unsqueeze(0))

        self.assertEqual(tuple(features.global_embedding.shape), (1, 768))
        self.assertEqual(tuple(features.patch_embeddings.shape), (1, 361, 768))
        self.assertTrue(torch.isfinite(features.global_embedding).all().item())
        self.assertTrue(torch.isfinite(features.patch_embeddings).all().item())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(features.patch_embeddings, dim=-1),
                torch.ones(1, 361, device=features.patch_embeddings.device),
                atol=1e-5,
            )
        )
        self.assertEqual(sum(p.numel() for p in encoder.backbone.parameters() if p.requires_grad), 0)


if __name__ == "__main__":
    unittest.main()
