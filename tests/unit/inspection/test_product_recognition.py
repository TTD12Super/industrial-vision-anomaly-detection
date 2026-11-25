import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from inspection.backends import BackendUnavailable, DinoV2GalleryRecognizer, build_product_recognizer
from inspection.config import ProductConfig
from inspection.interfaces import REVIEW_PRODUCT, RecognitionResult
from inspection.product_recognition import (
    CALIBRATION_ARTIFACT_TYPE,
    CALIBRATION_SCHEMA_VERSION,
    GALLERY_ARTIFACT_TYPE,
    GALLERY_SCHEMA_VERSION,
    calibrate_gallery_threshold,
    build_product_gallery,
    gallery_artifact_ready,
    load_gallery_artifact,
)
from mvtec_system.core.dinov2 import DINOv2Config


class _FakeEncoder:
    config = DINOv2Config(image_size=(4, 4))

    @staticmethod
    def metadata() -> dict[str, object]:
        return {
            "model_name": "fake-dino",
            "model_revision": "test",
            "patch_feature_key": "x_norm_patchtokens",
            "global_feature_key": "x_norm_clstoken",
            "image_size": [4, 4],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "l2_normalize_patches": True,
            "feature_layers": [],
            "embedding_dimension": 2,
        }

    @staticmethod
    def forward(images: torch.Tensor) -> SimpleNamespace:
        means = images.mean(dim=(1, 2, 3))
        rows = []
        for value in means:
            if float(value) < -1.0:
                rows.append([1.0, 0.0])
            elif float(value) > 1.0:
                rows.append([0.0, 1.0])
            else:
                rows.append([1.0, 1.0])
        return SimpleNamespace(global_embedding=torch.tensor(rows, dtype=torch.float32))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProductRecognitionTests(unittest.TestCase):
    def test_unknown_result_cannot_carry_a_known_product(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not carry"):
            RecognitionResult("screw", 0.1, "test-recognizer", "UNKNOWN_PRODUCT")

    def test_review_result_cannot_carry_a_product(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not carry"):
            RecognitionResult("screw", 0.9, "test-recognizer", REVIEW_PRODUCT)

    def test_known_result_requires_a_product(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires product_key"):
            RecognitionResult(None, 0.9, "test-recognizer", "KNOWN_PRODUCT")

    def test_missing_gallery_or_calibration_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(gallery_artifact_ready(root / "missing.pt"))
            with self.assertRaisesRegex(BackendUnavailable, "artifacts are not ready"):
                build_product_recognizer(
                    {
                        "backend": "gallery",
                        "gallery_path": root / "missing.pt",
                        "calibration_path": root / "missing.json",
                    }
                )

    def test_calibration_uses_known_target_and_unknown_examples(self) -> None:
        result = calibrate_gallery_threshold(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
            torch.tensor([[0.0, 1.0]]),
            target_known_recall=1.0,
        )

        self.assertAlmostEqual(result["known_threshold"], 0.8, places=6)
        self.assertEqual(result["known_validation_count"], 2)
        self.assertEqual(result["unknown_validation_count"], 1)
        self.assertEqual(result["unknown_validation_rejection_rate"], 1.0)

    def test_multi_product_calibration_derives_min_product_margin(self) -> None:
        result = calibrate_gallery_threshold(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.707, 0.707]]),
            target_known_recall=1.0,
            known_product_keys=["screw", "nut"],
            gallery_product_keys=["screw", "nut"],
        )

        self.assertEqual(result["min_product_margin"], 1.0)
        self.assertEqual(result["known_product_validation_recall"], 1.0)

    def test_dino_gallery_returns_known_or_unknown_without_fallback_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gallery_path = root / "gallery.pt"
            torch.save(
                {
                    "artifact_type": GALLERY_ARTIFACT_TYPE,
                    "schema_version": GALLERY_SCHEMA_VERSION,
                    "model_version": "fake-gallery",
                    "product_keys": ["screw", "nut"],
                    "object_ids": ["s1", "n1"],
                    "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "encoder": _FakeEncoder.metadata(),
                },
                gallery_path,
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "score_type": "cosine_similarity",
                        "known_threshold": 0.8,
                        "min_product_margin": 0.1,
                        "known_validation_count": 2,
                        "unknown_validation_count": 1,
                        "gallery_sha256": _sha256(gallery_path),
                    }
                ),
                encoding="utf-8",
            )
            recognizer = DinoV2GalleryRecognizer(
                gallery_path,
                calibration_path,
                encoder=_FakeEncoder(),
            )

            known = recognizer.recognize(np.zeros((8, 8, 3), dtype=np.uint8))
            unknown = recognizer.recognize(np.full((8, 8, 3), 127, dtype=np.uint8))

            self.assertEqual(known.status, "KNOWN_PRODUCT")
            self.assertEqual(known.product_key, "screw")
            self.assertEqual(unknown.status, "UNKNOWN_PRODUCT")
            self.assertIsNone(unknown.product_key)

    def test_multi_product_gallery_uses_product_level_margin_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gallery_path = root / "gallery.pt"
            torch.save(
                {
                    "artifact_type": GALLERY_ARTIFACT_TYPE,
                    "schema_version": GALLERY_SCHEMA_VERSION,
                    "model_version": "fake-gallery",
                    "product_keys": ["screw", "screw", "nut"],
                    "object_ids": ["s1", "s2", "n1"],
                    "embeddings": torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
                    "encoder": _FakeEncoder.metadata(),
                },
                gallery_path,
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "score_type": "cosine_similarity",
                        "known_threshold": 0.6,
                        "min_product_margin": 0.3,
                        "known_validation_count": 2,
                        "unknown_validation_count": 1,
                        "gallery_sha256": _sha256(gallery_path),
                    }
                ),
                encoding="utf-8",
            )
            recognizer = DinoV2GalleryRecognizer(gallery_path, calibration_path, encoder=_FakeEncoder())

            known = recognizer.recognize(np.zeros((8, 8, 3), dtype=np.uint8))
            ambiguous = recognizer.recognize(np.full((8, 8, 3), 127, dtype=np.uint8))

            self.assertEqual(known.status, "KNOWN_PRODUCT")
            self.assertEqual(known.product_key, "screw")
            self.assertIsNotNone(known.margin)
            self.assertEqual(ambiguous.status, REVIEW_PRODUCT)
            self.assertIsNone(ambiguous.product_key)
            self.assertLess(ambiguous.margin, 0.3)

    def test_one_product_gallery_ignores_product_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gallery_path = root / "gallery.pt"
            torch.save(
                {
                    "artifact_type": GALLERY_ARTIFACT_TYPE,
                    "schema_version": GALLERY_SCHEMA_VERSION,
                    "model_version": "fake-gallery",
                    "product_keys": ["screw", "screw"],
                    "object_ids": ["s1", "s2"],
                    "embeddings": torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
                    "encoder": _FakeEncoder.metadata(),
                },
                gallery_path,
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "score_type": "cosine_similarity",
                        "known_threshold": 0.8,
                        "known_validation_count": 2,
                        "unknown_validation_count": 1,
                        "gallery_sha256": _sha256(gallery_path),
                    }
                ),
                encoding="utf-8",
            )
            recognizer = DinoV2GalleryRecognizer(gallery_path, calibration_path, encoder=_FakeEncoder())

            result = recognizer.recognize(np.zeros((8, 8, 3), dtype=np.uint8))

            self.assertEqual(result.status, "KNOWN_PRODUCT")
            self.assertEqual(result.product_key, "screw")
            self.assertIsNone(result.margin)

    def test_builder_uses_train_good_crops_and_writes_gallery_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            image_manifest = dataset_root / "manifests/images.jsonl"
            object_manifest = dataset_root / "manifests/objects.jsonl"
            image_manifest.parent.mkdir(parents=True)
            rows = []
            objects = []
            for index, split in enumerate(("train", "validation", "test"), 1):
                image_path = dataset_root / f"image_{index}.png"
                Image.fromarray(np.full((10, 10, 3), index * 50, dtype=np.uint8)).save(image_path)
                rows.append(
                    {
                        "image_id": f"image_{index}",
                        "relative_path": image_path.relative_to(dataset_root).as_posix(),
                        "split": split,
                        "width": 10,
                        "height": 10,
                        "sha256": _sha256(image_path),
                        "capture_group_id": f"group_{index}",
                    }
                )
                objects.append(
                    {
                        "object_id": f"object_{index}",
                        "image_id": f"image_{index}",
                        "product_key": "screw",
                        "quality_label": "GOOD" if split != "validation" else "DEFECT",
                        "obb_xyxyxyxy": [[1, 1], [9, 1], [9, 9], [1, 9]],
                    }
                )
            image_manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            object_manifest.write_text("\n".join(json.dumps(row) for row in objects) + "\n", encoding="utf-8")
            config = ProductConfig(
                source_path=root / "configs/products/screw.yaml",
                schema_version=1,
                product_key="screw",
                dataset_root=dataset_root,
                image_manifest=Path("manifests/images.jsonl"),
                object_manifest=Path("manifests/objects.jsonl"),
                split_policy="grouped",
                models={"classifier": {"encoder_artifact_root": "artifacts/dinov2_knn"}},
                policy={"quality_labels": ["GOOD", "DEFECT"]},
                outputs={},
            )
            output = root / "gallery.pt"

            summary = build_product_gallery([config], output, encoder=_FakeEncoder())

            payload = load_gallery_artifact(output)
            self.assertTrue(gallery_artifact_ready(output))
            self.assertEqual(summary["embedding_count"], 1)
            self.assertEqual(payload["product_keys"], ["screw"])


if __name__ == "__main__":
    unittest.main()
