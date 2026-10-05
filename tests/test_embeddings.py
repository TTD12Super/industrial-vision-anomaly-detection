"""Tests for the classification embedding cache."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from mvtec_system import (
    SharedDINOv2,
    build_global_embedding_artifact,
    category_mapping,
    load_compatible_embedding_artifacts,
    make_embedding_metadata,
    read_classification_manifest,
    save_embedding_artifacts,
)


class TinyBackbone(torch.nn.Module):
    """Small deterministic backbone with the same feature contract."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = type("PatchEmbed", (), {"patch_size": 14})()
        self.embed_dim = 4
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        patch_values = torch.zeros(batch_size, 361, self.embed_dim, device=images.device)
        global_values = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, self.embed_dim)
        return {
            "x_norm_patchtokens": patch_values,
            "x_norm_clstoken": global_values,
        }


class EmbeddingArtifactTests(unittest.TestCase):
    def make_fixture(self) -> tuple[Path, Path]:
        root = Path(tempfile.mkdtemp())
        processed = root / "processed"
        rows = []
        for category, values in {"bottle": (10, 20), "cable": (30, 40)}.items():
            path = processed / category / "train" / "good" / "000.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.full((256, 256, 3), values[0], dtype=np.uint8), mode="RGB"
            ).save(path)
            rows.append(
                {
                    "image_path": path.relative_to(processed).as_posix(),
                    "category": category,
                    "split": "train",
                }
            )
            path = processed / category / "train" / "good" / "001.png"
            Image.fromarray(
                np.full((256, 256, 3), values[1], dtype=np.uint8), mode="RGB"
            ).save(path)
            rows.append(
                {
                    "image_path": path.relative_to(processed).as_posix(),
                    "category": category,
                    "split": "validation",
                }
            )

        manifest = root / "classification_split.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "category", "split"])
            writer.writeheader()
            writer.writerows(rows)
        return processed, manifest

    def test_manifest_mapping_and_embedding_artifacts(self) -> None:
        processed, manifest = self.make_fixture()
        rows = read_classification_manifest(manifest, processed)
        mapping = category_mapping(rows)
        encoder = SharedDINOv2(backbone=TinyBackbone(), device="cpu")

        train = build_global_embedding_artifact(
            rows, "train", processed, encoder, mapping, batch_size=1
        )
        validation = build_global_embedding_artifact(rows, "validation", processed, encoder, mapping)

        self.assertEqual(mapping, {"bottle": 0, "cable": 1})
        self.assertEqual(tuple(train["embeddings"].shape), (2, 4))
        self.assertEqual(tuple(validation["embeddings"].shape), (2, 4))
        self.assertEqual(train["labels"].tolist(), [0, 1])
        self.assertEqual(train["split"], "train")
        self.assertTrue(torch.isfinite(train["embeddings"]).all().item())

        metadata = make_embedding_metadata(rows, mapping, manifest, processed, encoder)
        self.assertEqual(metadata["artifact_version"], "v1")
        self.assertEqual(metadata["splits"]["train"]["count"], 2)
        self.assertEqual(metadata["encoder"]["embedding_dimension"], 4)

        output = processed.parent / "artifacts"
        save_embedding_artifacts(output, train, validation, mapping, metadata)
        cached = load_compatible_embedding_artifacts(output, metadata)
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["image_paths"], train["image_paths"])
        saved_train = torch.load(output / "train_embeddings.pt", map_location="cpu", weights_only=True)
        saved_mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
        self.assertTrue(torch.equal(saved_train["labels"], train["labels"]))
        self.assertEqual(saved_mapping["category_to_index"], mapping)

    def test_manifest_rejects_non_train_good_path(self) -> None:
        processed, manifest = self.make_fixture()
        lines = manifest.read_text(encoding="utf-8").replace(
            "bottle/train/good/000.png", "bottle/test/defect/000.png"
        )
        manifest.write_text(lines, encoding="utf-8")
        with self.assertRaises(ValueError):
            read_classification_manifest(manifest, processed)


if __name__ == "__main__":
    unittest.main()
