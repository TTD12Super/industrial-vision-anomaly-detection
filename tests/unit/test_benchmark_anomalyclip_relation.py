"""Focused pure-math checks for the isolated AnomalyCLIP relation experiment."""

import numpy as np
import pytest

import scripts.benchmark_anomalyclip_relation as benchmark
from scripts.benchmark_anomalyclip_relation import (
    METHODS,
    MVTEC_CATEGORIES,
    _aggregate_rows,
    _build_prototypes,
    _category_metrics,
    _require_clean_split_hashes,
    _thresholds,
    fuse_maps,
    relation_features,
    relation_scores,
)


def test_relation_math_is_m_a_at_f_and_fusion_is_per_image_normalized() -> None:
    attention = np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    prototype = np.zeros_like(features)

    relation = relation_features(attention, features)
    np.testing.assert_allclose(relation, attention @ features)
    np.testing.assert_allclose(
        relation_scores(attention, features, prototype),
        np.linalg.norm(attention @ features, axis=1),
    )

    raw = np.asarray([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    contextual = np.asarray([[10.0, 20.0], [15.0, 25.0]], dtype=np.float32)
    expected = (raw - raw.min()) / (raw.max() - raw.min())
    expected += (contextual - contextual.min()) / (contextual.max() - contextual.min())
    np.testing.assert_allclose(fuse_maps(raw, contextual, beta=1.0), expected)


def test_metric_memory_preflight_rejects_insufficient_available_ram(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_available_memory_bytes", lambda: 3 * 1024**3)
    result = benchmark._preflight_metric_memory(10)
    assert result["available_bytes"] == 3 * 1024**3
    assert result["estimated_peak_bytes"] == 2 * 1024**3 + 320

    monkeypatch.setattr(benchmark, "_available_memory_bytes", lambda: 2 * 1024**3)
    with pytest.raises(MemoryError, match="lower --metric-size"):
        benchmark._preflight_metric_memory(10)


def test_clip_backbone_provenance_hashes_cached_model(tmp_path) -> None:
    cache = tmp_path / "clip"
    cache.mkdir()
    weights = cache / "ViT-L-14-336px.pt"
    weights.write_bytes(b"clip weights")

    resolved_path, digest = benchmark._clip_weight_provenance(cache)
    assert resolved_path == weights
    assert digest == benchmark._sha256_file(weights)

    with pytest.raises(FileNotFoundError, match="CLIP backbone weights"):
        benchmark._clip_weight_provenance(tmp_path / "empty")


def test_split_audit_records_data_and_masks_and_rejects_duplicate_content(tmp_path) -> None:
    train = tmp_path / "category/train.png"
    validation = tmp_path / "category/validation.png"
    test_image = tmp_path / "category/test.png"
    test_mask = tmp_path / "category/test_mask.png"
    for path, content in (
        (train, b"train"),
        (validation, b"validation"),
        (test_image, b"test"),
        (test_mask, b"mask"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {"category": {"train": [train], "validation": [validation]}}
    test = [{"path": test_image, "mask": test_mask}]

    audit = _require_clean_split_hashes(manifest, test, tmp_path)
    assert audit["overlap_entry_count"] == 0
    assert set(audit["split_tree_sha256"]) == {"train", "validation", "test"}
    assert audit["test_mask_count"] == 1
    assert len(audit["test_masks_tree_sha256"]) == 64

    validation.write_bytes(b"train")
    with pytest.raises(RuntimeError, match="Exact byte-hash overlap"):
        _require_clean_split_hashes(manifest, test, tmp_path)


def test_category_metrics_rejects_single_class_masks_and_macro_records_category_count() -> None:
    with pytest.raises(ValueError, match="both GOOD and DEFECT pixels"):
        _category_metrics(
            "bottle",
            "relation_map",
            [np.asarray([0.1, 0.2], dtype=np.float32)],
            [np.zeros(2, dtype=np.uint8)],
            [np.zeros(2, dtype=bool)],
            threshold=0.5,
            image_count=1,
            mask_images=0,
        )

    rows = [
        {
            "pixel_ap": score,
            "pixel_auroc": score,
            "precision": score,
            "recall": score,
            "f1": score,
            "dice": score,
            "iou": score,
            "specificity": score,
            "good_fp": score,
            "image_count": 1,
            "map_images": 1,
            "mask_images": 1,
            "map_coverage": 1.0,
            "mask_coverage": 1.0,
            "pixel_count": 2,
            "good_fp_pixels": 0,
            "good_pixel_count": 1,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
        }
        for score in (0.2, 0.4, 0.6)
    ]
    macro = _aggregate_rows(rows, "relation_map", "macro")
    assert macro["macro_category_count"] == 3
    assert macro["pixel_ap"] == pytest.approx(0.4)


def test_prototypes_and_thresholds_read_only_their_assigned_splits(monkeypatch) -> None:
    manifest = {
        category: {
            "train": [f"{category}/train/{index}" for index in range(2)],
            "validation": [f"{category}/validation/{index}" for index in range(2)],
        }
        for category in MVTEC_CATEGORIES
    }
    monkeypatch.setattr(benchmark, "_load_rgb", lambda path: path)

    class Runner:
        def __init__(self):
            self.images = []

        def infer(self, image):
            self.images.append(image)
            return np.zeros((1, 1), dtype=np.float32), np.ones((1, 1)), np.asarray([[float(len(self.images))]])

    runner = Runner()
    prototypes = _build_prototypes(runner, manifest)
    expected_train = [path for rows in manifest.values() for path in rows["train"]]
    assert runner.images == expected_train
    assert set(prototypes) == set(MVTEC_CATEGORIES)

    validation_images = []

    def fake_maps(_runner, image, _prototype):
        validation_images.append(image)
        return {
            method: np.full((2, 2), index + 1, dtype=np.float32)
            for index, method in enumerate(METHODS)
        }

    monkeypatch.setattr(benchmark, "_maps", fake_maps)
    thresholds = _thresholds(runner, manifest, prototypes)
    expected_validation = [
        path for rows in manifest.values() for path in rows["validation"]
    ]
    assert validation_images == expected_validation
    assert set(thresholds) == set(MVTEC_CATEGORIES)
