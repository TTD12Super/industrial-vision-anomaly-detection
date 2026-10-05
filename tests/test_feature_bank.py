"""Tests for normal patch feature-bank reuse."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from mvtec_system import (
    SharedDINOv2,
    build_feature_bank,
    load_feature_bank,
    load_or_build_feature_bank,
    make_bank_metadata,
    save_feature_bank,
)


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = SimpleNamespace(patch_size=14)
        self.embed_dim = 4
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        patches = torch.arange(361 * 4, dtype=torch.float32).reshape(1, 361, 4)
        patches = patches.expand(batch_size, -1, -1).to(images.device)
        global_features = torch.zeros(batch_size, 4, device=images.device)
        return {
            "x_norm_patchtokens": patches,
            "x_norm_clstoken": global_features,
        }


class FeatureBankTests(unittest.TestCase):
    def make_fixture(self) -> tuple[Path, Path, list[Path]]:
        root = Path(tempfile.mkdtemp())
        processed = root / "processed"
        train_dir = processed / "bottle" / "train" / "good"
        train_dir.mkdir(parents=True)
        paths = []
        for index in range(2):
            path = train_dir / f"{index:03d}.png"
            Image.fromarray(np.full((256, 256, 3), index, dtype=np.uint8), mode="RGB").save(path)
            paths.append(path)
        return root, processed, paths

    def make_encoder(self) -> SharedDINOv2:
        return SharedDINOv2(backbone=TinyBackbone(), device="cpu")

    def test_build_save_and_load_preserve_notebook_bank_contract(self) -> None:
        root, processed, paths = self.make_fixture()
        encoder = self.make_encoder()
        features, metadata = build_feature_bank("bottle", paths, processed, encoder, batch_size=1)

        self.assertEqual(tuple(features.shape), (722, 4))
        self.assertEqual(metadata["feature_key"], "x_norm_patchtokens")
        self.assertEqual(metadata["sampling"], "none; all train/good patches")
        path = root / "banks" / "bottle" / "feature_bank.pt"
        save_feature_bank(path, features, metadata)
        loaded = load_feature_bank(path, metadata, "cpu")

        self.assertTrue(torch.equal(loaded, features.cpu()))

    def test_load_or_build_reuses_existing_compatible_bank(self) -> None:
        root, processed, paths = self.make_fixture()
        first = load_or_build_feature_bank(
            root / "banks",
            "bottle",
            paths,
            processed,
            self.make_encoder(),
            batch_size=1,
        )
        second = load_or_build_feature_bank(
            root / "banks",
            "bottle",
            paths,
            processed,
            self.make_encoder(),
            batch_size=1,
        )

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertTrue(torch.equal(first.features, second.features))

    def test_incompatible_bank_requires_explicit_rebuild(self) -> None:
        root, processed, paths = self.make_fixture()
        encoder = self.make_encoder()
        metadata = make_bank_metadata("bottle", paths, processed, encoder)
        path = root / "banks" / "bottle" / "feature_bank.pt"
        save_feature_bank(path, torch.zeros(722, 4), {**metadata, "model_revision": "different"})

        with self.assertRaises(ValueError):
            load_or_build_feature_bank(
                root / "banks",
                "bottle",
                paths,
                processed,
                self.make_encoder(),
            )


if __name__ == "__main__":
    unittest.main()
