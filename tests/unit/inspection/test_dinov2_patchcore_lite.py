from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
from PIL import Image
import torch

from inspection.backends import DinoV2PatchMemoryClassifier
from inspection.dinov2_patchcore_lite import (
    _score_records,
    aggregate_patch_scores,
    build_patch_memory_bank,
    deterministic_cad_good_split,
    image_metrics,
    load_patch_memory_artifact,
    patchcore_scores,
    per_group_metrics,
    save_patch_memory_artifact,
    select_good_quantile_threshold,
)
from mvtec_system.core.dinov2 import DINOv2Config


class _FakeEncoder:
    config = DINOv2Config(image_size=(4, 4))
    patch_grid = (1, 2)
    padded_size = (4, 4)
    patch_count = 2

    def __init__(self) -> None:
        self.forward_calls = 0

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        self.forward_calls += 1
        batch = images.shape[0]
        patch = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        return SimpleNamespace(patch_embeddings=patch.repeat(batch, 1, 1))

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": "fake",
            "model_revision": "unit",
            "patch_feature_key": "patch",
            "global_feature_key": "global",
            "image_size": [4, 4],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "l2_normalize_patches": True,
            "feature_layers": [],
            "patch_size": [4, 2],
            "padding": [0, 0],
            "padded_size": [4, 4],
            "patch_grid": [1, 2],
            "patch_count": 2,
            "embedding_dimension": 2,
        }


class DINOv2PatchCoreLiteTests(TestCase):
    def test_memory_bank_build_and_nearest_patch_score(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = []
            for index in range(2):
                path = root / f"good-{index}.png"
                Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(path)
                image_paths.append(path)

            bank = build_patch_memory_bank(image_paths, _FakeEncoder(), batch_size=1)
            image_score, patch_scores = patchcore_scores(
                torch.tensor([[1.0, 0.0], [1.0, 1.0]]), bank, k=1
            )

            self.assertEqual(tuple(bank.shape), (4, 2))
            self.assertEqual(patch_scores.shape, (2,))
            self.assertAlmostEqual(image_score, 1.0, places=6)

    def test_score_records_batches_encoder_forwards_and_preserves_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(5):
                path = root / f"good-{index}.png"
                Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(path)
                records.append({"path": path, "label": 0, "group": "good", "mask_path": None})

            encoder = _FakeEncoder()
            scored = _score_records(
                records,
                encoder,
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                k=1,
                batch_size=2,
            )

            self.assertEqual(encoder.forward_calls, 3)
            self.assertEqual([record["path"] for record in scored], [record["path"] for record in records])
            self.assertEqual(len(scored), 5)
            self.assertEqual(scored[0]["patch_scores"].shape, (2,))
            self.assertTrue(all(record["anomaly_map"].shape == (4, 4) for record in scored))

    def test_patch_score_aggregation_uses_declared_top_fractions(self) -> None:
        patch_scores = np.arange(1.0, 101.0)

        self.assertEqual(aggregate_patch_scores(patch_scores, "max"), 100.0)
        self.assertEqual(aggregate_patch_scores(patch_scores, "top_1pct"), 100.0)
        self.assertEqual(aggregate_patch_scores(patch_scores, "top_5pct"), 98.0)
        self.assertEqual(aggregate_patch_scores(patch_scores, "top_10pct"), 95.5)
        self.assertEqual(aggregate_patch_scores(patch_scores, "top_20pct"), 90.5)

    def test_patch_memory_artifact_round_trip_validates_contract(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.pt"
            encoder = _FakeEncoder()
            summary = save_patch_memory_artifact(
                path,
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                encoder,
                threshold=0.4,
                k=1,
                aggregation="max",
            )
            loaded = load_patch_memory_artifact(path)

            self.assertEqual(summary["memory_count"], 2)
            self.assertEqual(loaded["artifact_type"], "dinov2_patch_memory_classifier")
            self.assertEqual(tuple(loaded["memory_bank"].shape), (2, 2))
            self.assertAlmostEqual(loaded["threshold"], 0.4, places=6)

            classifier = DinoV2PatchMemoryClassifier(path, encoder=encoder)
            result = classifier.classify(np.full((8, 8, 3), 128, dtype=np.uint8))
            self.assertGreaterEqual(result.p_defect, 0.0)
            self.assertLessEqual(result.p_defect, 1.0)
            self.assertEqual(result.model_version, "dinov2-patch-memory")

    def test_cad_good_hash_split_is_deterministic_and_disjoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "train" / "0_good" / f"image-{index}.png" for index in range(10)]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"unique-{index}".encode())
            first = deterministic_cad_good_split(paths, root=root, memory_fraction=0.6, salt="unit-test")
            second = deterministic_cad_good_split(list(reversed(paths)), root=root, memory_fraction=0.6, salt="unit-test")

            first_memory = set(first["memory_paths"])
            first_calibration = set(first["calibration_paths"])
            self.assertEqual(first_memory, set(second["memory_paths"]))
            self.assertEqual(first_calibration, set(second["calibration_paths"]))
            self.assertTrue(first_memory.isdisjoint(first_calibration))
            self.assertEqual(first_memory | first_calibration, set(path.resolve() for path in paths))
            self.assertEqual(first["split_hash"], second["split_hash"])
            self.assertEqual(first["memory_path_hash"], second["memory_path_hash"])
            self.assertEqual(first["calibration_path_hash"], second["calibration_path_hash"])

    def test_cad_hash_split_excludes_duplicate_and_test_content(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "train" / "0_good" / f"image-{index}.png" for index in range(6)]
            contents = [b"a", b"a", b"b", b"c", b"d", b"e"]
            for path, content in zip(paths, contents):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            test_duplicate = root / "test" / "1_scratch" / "test.png"
            test_duplicate.parent.mkdir(parents=True, exist_ok=True)
            test_duplicate.write_bytes(b"b")

            split = deterministic_cad_good_split(
                paths,
                root=root,
                memory_fraction=0.5,
                salt="unit-test-content",
                excluded_paths=[test_duplicate],
            )

            memory_hashes = set(split["memory_content_hashes"])
            calibration_hashes = set(split["calibration_content_hashes"])
            excluded_hashes = set(split["excluded_content_hashes"])
            self.assertTrue(split["content_hash_disjoint"])
            self.assertTrue(memory_hashes.isdisjoint(calibration_hashes))
            self.assertTrue(memory_hashes.isdisjoint(excluded_hashes))
            self.assertTrue(calibration_hashes.isdisjoint(excluded_hashes))
            self.assertEqual(split["exclusion_counts"], {"duplicate_candidate_content": 1, "excluded_test_content": 1})
            self.assertEqual(split["retained_unique_content_count"], 4)
            self.assertEqual(split["excluded_path_count"], 2)
            self.assertEqual(split["exclusion_reference_path_count"], 1)
            self.assertEqual(split["exclusion_reference_unique_content_count"], 1)
            self.assertIn("sha256 exact file bytes", split["hash_group_policy"])

    def test_good_only_quantile_threshold_respects_declared_fpr_rule(self) -> None:
        calibration = select_good_quantile_threshold([0.1, 0.2, 0.3, 0.4], allowed_fpr=0.25)

        self.assertEqual(calibration["threshold"], 0.4)
        self.assertEqual(calibration["good_quantile"], 0.75)
        self.assertEqual(calibration["empirical_good_fpr"], 0.25)
        self.assertIn("CAD-SD train/0_good", calibration["rule"])

    def test_locked_image_metrics_report_auroc_ap_recall_and_good_fp(self) -> None:
        metrics = image_metrics(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            threshold=0.5,
        )

        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["defect_recall"], 1.0)
        self.assertEqual(metrics["good_false_positive_count"], 0)
        self.assertEqual(metrics["good_false_positive_rate"], 0.0)

    def test_per_group_metrics_keep_single_class_group_counts(self) -> None:
        metrics = per_group_metrics(
            [
                {"group": "0_good", "label": 0, "image_score": 0.1},
                {"group": "1_scratch", "label": 1, "image_score": 0.9},
            ],
            threshold=0.5,
        )

        self.assertEqual(metrics["0_good"]["good_false_positive_rate"], 0.0)
        self.assertEqual(metrics["1_scratch"]["defect_recall"], 1.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
