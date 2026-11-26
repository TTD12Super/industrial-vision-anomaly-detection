"""Evaluation-metrics and offline-evaluator tests.

Metric/label/mask tests need only numpy + scikit-learn + Pillow and run
without torch. The end-to-end category tests use a synthetic bank and a stub
extractor, so they never download DINOv2 or touch real feature banks.
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SIZE = (256, 256)


def _write_png(path, array, mode="L"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode=mode).save(path)


@unittest.skipIf(not (HAS_NUMPY and HAS_PIL), "numpy/Pillow required")
class TestLabelsAndMaskMapping(unittest.TestCase):
    """Reference rules: labels from directory names, masks from ground_truth."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="eval_ds_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self._make_fixture("bottle")

    def _make_fixture(self, category):
        base = self.root / category
        for name in ("000", "001", "002"):
            _write_png(base / "train" / "good" / f"{name}.png", np.full(SIZE, 128))
        _write_png(base / "test" / "good" / "000.png", np.full(SIZE, 128))
        _write_png(base / "test" / "broken_large" / "001.png", np.full(SIZE, 90))
        mask = np.zeros(SIZE, dtype=np.uint8)
        mask[10:40, 10:40] = 255
        _write_png(base / "ground_truth" / "broken_large" / "001_mask.png", mask)

    def test_image_labels_follow_directory_names(self):
        from model.evaluation.metrics import image_labels

        paths = [
            self.root / "bottle/test/good/000.png",
            self.root / "bottle/test/broken_large/001.png",
        ]
        self.assertEqual(image_labels(paths).tolist(), [False, True])

    def test_good_image_has_no_mask_path(self):
        from model.evaluation.metrics import find_mask_path

        path = self.root / "bottle/test/good/000.png"
        self.assertIsNone(find_mask_path("bottle", path, self.root))

    def test_defective_mask_path_maps_unambiguously(self):
        from model.evaluation.metrics import find_mask_path

        path = self.root / "bottle/test/broken_large/001.png"
        found = find_mask_path("bottle", path, self.root)
        self.assertEqual(found.name, "001_mask.png")

    def test_good_image_mask_is_all_zero(self):
        from model.evaluation.metrics import load_mask

        mask = load_mask("bottle", self.root / "bottle/test/good/000.png", self.root)
        self.assertEqual(mask.shape, SIZE)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertFalse(mask.any())

    def test_defective_mask_is_binarized(self):
        from model.evaluation.metrics import load_mask

        mask = load_mask("bottle", self.root / "bottle/test/broken_large/001.png", self.root)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask.any())
        self.assertEqual(int(mask.sum()), 30 * 30)

    def test_missing_mask_raises(self):
        from model.evaluation.metrics import load_mask

        path = self.root / "bottle/test/broken_large/009.png"
        _write_png(path, np.full(SIZE, 90))
        with self.assertRaises(ValueError):
            load_mask("bottle", path, self.root)

    def test_duplicate_masks_raise(self):
        from model.evaluation.metrics import find_mask_path

        folder = self.root / "bottle/ground_truth/broken_large"
        _write_png(folder / "001_mask.jpg", np.zeros(SIZE, dtype=np.uint8))
        with self.assertRaises(ValueError):
            find_mask_path(
                "bottle", self.root / "bottle/test/broken_large/001.png", self.root
            )

    def test_wrong_size_mask_raises(self):
        from model.evaluation.metrics import load_mask

        path = self.root / "bottle/test/broken_large/003.png"
        _write_png(path, np.full(SIZE, 90))
        _write_png(
            self.root / "bottle/ground_truth/broken_large/003_mask.png",
            np.zeros((8, 8), dtype=np.uint8),
        )
        with self.assertRaises(ValueError):
            load_mask("bottle", path, self.root)

    def test_empty_mask_raises(self):
        from model.evaluation.metrics import load_mask

        path = self.root / "bottle/test/broken_large/004.png"
        _write_png(path, np.full(SIZE, 90))
        _write_png(
            self.root / "bottle/ground_truth/broken_large/004_mask.png",
            np.zeros(SIZE, dtype=np.uint8),
        )
        with self.assertRaises(ValueError):
            load_mask("bottle", path, self.root)

    def test_non_binary_mask_raises(self):
        from model.evaluation.metrics import load_mask

        path = self.root / "bottle/test/broken_large/005.png"
        _write_png(path, np.full(SIZE, 90))
        bad = np.zeros(SIZE, dtype=np.uint8)
        bad[0, 0] = 3
        _write_png(self.root / "bottle/ground_truth/broken_large/005_mask.png", bad)
        with self.assertRaises(ValueError):
            load_mask("bottle", path, self.root)

    def test_unreadable_mask_raises(self):
        from model.evaluation.metrics import load_mask

        path = self.root / "bottle/test/broken_large/006.png"
        _write_png(path, np.full(SIZE, 90))
        broken = self.root / "bottle/ground_truth/broken_large/006_mask.png"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"not an image")
        with self.assertRaises(Exception):
            load_mask("bottle", path, self.root)

    def test_category_paths_are_sorted_and_disjoint(self):
        from model.evaluation.evaluate import category_paths

        train_paths, test_paths = category_paths(self.root, "bottle")
        self.assertEqual(train_paths, sorted(train_paths))
        self.assertEqual(test_paths, sorted(test_paths))
        self.assertEqual(len(train_paths), 3)
        self.assertEqual(len(test_paths), 2)
        # Sorted by full path: "broken_large" sorts before "good".
        self.assertEqual([path.name for path in test_paths], ["001.png", "000.png"])


@unittest.skipIf(not HAS_NUMPY, "numpy required")
class TestRankingMetrics(unittest.TestCase):
    def test_perfect_separation(self):
        from model.evaluation.metrics import ranking_metrics

        result = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["average_precision"], 1.0)

    def test_known_value(self):
        from model.evaluation.metrics import ranking_metrics

        result = ranking_metrics([0, 1], [0.2, 0.7])
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["average_precision"], 1.0)

    def test_shape_mismatch_raises(self):
        from model.evaluation.metrics import ranking_metrics

        with self.assertRaises(ValueError):
            ranking_metrics([0, 1], [0.1])

    def test_non_finite_scores_raise(self):
        from model.evaluation.metrics import ranking_metrics

        with self.assertRaises(ValueError):
            ranking_metrics([0, 1], [0.1, float("nan")])

    def test_single_class_returns_none(self):
        from model.evaluation.metrics import ranking_metrics

        result = ranking_metrics([1, 1, 1], [0.1, 0.2, 0.3])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["average_precision"])

    def test_pixel_metrics_match_reference_population(self):
        from model.evaluation.metrics import pixel_metrics

        masks = np.array([[False, False], [True, True]])
        maps = np.array([[0.1, 0.2], [0.8, 0.9]])
        result = pixel_metrics(masks, maps)
        self.assertAlmostEqual(result["auroc"], 1.0)

    def test_pixel_memory_guard_raises_above_limit(self):
        from model.evaluation.metrics import (
            MAX_PIXEL_METRIC_GIB,
            pixel_metric_memory_gib,
            pixel_metrics,
        )

        # Element count (not dtype) drives the reference budget formula.
        maps = np.zeros((800, 256, 256), dtype=bool)
        masks = np.zeros_like(maps)
        self.assertGreater(pixel_metric_memory_gib(maps), MAX_PIXEL_METRIC_GIB)
        with self.assertRaises(MemoryError):
            pixel_metrics(masks, maps)

    def test_macro_average_excludes_none(self):
        from model.evaluation.metrics import macro_average

        mean, count = macro_average([1.0, None, 0.5])
        self.assertAlmostEqual(mean, 0.75)
        self.assertEqual(count, 2)

    def test_macro_average_all_none(self):
        from model.evaluation.metrics import macro_average

        self.assertEqual(macro_average([None, None]), (None, 0))


class StubExtractor:
    """Minimal stand-in for DINOv2FeatureExtractor (no download, no torch model)."""

    patch_grid = (19, 19)
    padded_size = (266, 266)
    patch_count = 361
    pad_right = 10
    pad_bottom = 10
    device = "cpu"

    def __init__(self, embedding_dimension=8):
        self.embedding_dimension = embedding_dimension

    def extract_batch(self, images):
        batch = images.shape[0]
        flat = images.reshape(batch, -1).mean(dim=1, keepdim=True)
        features = flat.unsqueeze(1).repeat(1, self.patch_count, self.embedding_dimension)
        return features


@unittest.skipIf(not (HAS_NUMPY and HAS_PIL and HAS_TORCH), "numpy/Pillow/torch required")
class TestEvaluateCategoryEndToEnd(unittest.TestCase):
    """Full offline loop with a synthetic bank and a stub extractor."""

    def setUp(self):
        from model.inference.feature_bank import bank_metadata, save_feature_bank

        self.root = Path(tempfile.mkdtemp(prefix="eval_run_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.processed = self.root / "processed"
        self.banks = self.root / "banks"
        category = "bottle"
        base = self.processed / category
        for index in range(2):
            _write_png(base / "train" / "good" / f"{index:03d}.png", np.full(SIZE, 100 + index))
        _write_png(base / "test" / "good" / "000.png", np.full(SIZE, 100))
        _write_png(base / "test" / "broken_large" / "001.png", np.full(SIZE, 40))
        mask = np.zeros(SIZE, dtype=np.uint8)
        mask[20:60, 20:60] = 255
        _write_png(base / "ground_truth" / "broken_large" / "001_mask.png", mask)

        train_paths = sorted((base / "train" / "good").glob("*.png"))
        relative = [str(path.relative_to(self.processed)) for path in train_paths]
        metadata = bank_metadata(
            category,
            embedding_dimension=8,
            patch_grid=(19, 19),
            padding=(10, 10),
            train_paths=relative,
        )
        bank = torch.zeros(len(train_paths) * 361, 8, dtype=torch.float32)
        bank_path = self.banks / category / "feature_bank.pt"
        save_feature_bank(bank_path, bank, metadata)

    def _evaluate(self, **kwargs):
        from model.evaluation.evaluate import evaluate_category

        return evaluate_category(
            "bottle",
            processed_root=self.processed,
            bank_dir=self.banks,
            extractor=StubExtractor(),
            batch_size=2,
            **kwargs,
        )

    def test_end_to_end_counts_and_metrics(self):
        result = self._evaluate()
        self.assertEqual(result.status, "ok", result.error_message)
        self.assertEqual(result.train_images, 2)
        self.assertEqual(result.test_images, 2)
        self.assertEqual(result.normal_images, 1)
        self.assertEqual(result.defective_images, 1)
        for value in result.metrics.values():
            self.assertTrue(value is None or np.isfinite(value))
        self.assertEqual(len(result.rows), 2)
        self.assertEqual([row["is_anomaly"] for row in result.rows], [1, 0])
        # Zero bank => score is the L2 norm of the normalized patch features, so
        # the darker (defective) image must score strictly higher than the normal.
        defect_score = result.rows[0]["anomaly_score"]
        normal_score = result.rows[1]["anomaly_score"]
        self.assertGreater(defect_score, normal_score)
        self.assertTrue(np.isfinite(defect_score) and np.isfinite(normal_score))
        self.assertAlmostEqual(result.image_auroc, 1.0)
        self.assertAlmostEqual(result.image_ap, 1.0)

    def test_rows_preserve_sorted_order(self):
        result = self._evaluate()
        paths = [row["image_path"] for row in result.rows]
        self.assertEqual(paths, sorted(paths))

    def test_failed_bank_is_reported_not_swallowed(self):
        result = evaluate_category_broken_bank(self)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "FileNotFoundError")
        self.assertIn("not available", result.error_message)

    def test_pixel_metrics_can_be_skipped(self):
        result = self._evaluate(compute_pixel_metrics=False)
        self.assertEqual(result.status, "ok", result.error_message)
        self.assertIsNotNone(result.image_auroc)
        self.assertIsNone(result.pixel_auroc)
        self.assertIsNone(result.pixel_ap)

    def test_outputs_and_aggregates(self):
        from model.evaluation.evaluate import (
            write_aggregate_outputs,
            write_category_outputs,
        )

        result = self._evaluate()
        run_dir = self.root / "run"
        run_dir.mkdir()
        write_category_outputs(run_dir, result, save_maps=False)
        write_aggregate_outputs(run_dir, [result])

        scores_csv = run_dir / "bottle" / "image_scores.csv"
        with scores_csv.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["defect_type"], "broken_large")

        metrics = json.loads((run_dir / "bottle" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["status"], "ok")
        self.assertIn("threshold_strategy", metrics)
        self.assertEqual(
            metrics["threshold_strategy"], "continuous scores only; no test-label calibration"
        )
        self.assertIn("aupro", metrics)
        self.assertEqual(metrics["aupro"], "not implemented")

        macro = (run_dir / "macro_metrics.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(macro), 5)
        for line in macro[1:]:
            self.assertEqual(len(line.split(",")), 3)

        self.assertEqual(json.loads((run_dir / "failures.json").read_text(encoding="utf-8")), [])

    def test_failure_reported_in_aggregate_outputs(self):
        from model.evaluation.evaluate import CategoryResult, write_aggregate_outputs

        failure = CategoryResult(
            category="zipper",
            status="failed",
            error_type="MemoryError",
            error_message="boom",
        )
        run_dir = self.root / "fail_run"
        run_dir.mkdir()
        write_aggregate_outputs(run_dir, [failure])
        failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(failures[0]["category"], "zipper")
        self.assertEqual(failures[0]["error_type"], "MemoryError")
        macro = (run_dir / "macro_metrics.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertTrue(all(line.endswith(",0") for line in macro[1:]))

    def test_run_config_hash_is_deterministic(self):
        from model.evaluation.evaluate import build_run_config, configuration_hash

        config = build_run_config(
            processed_root=self.processed,
            bank_dir=self.banks,
            run_dir=self.root / "cfg",
            extractor=StubExtractor(),
            categories=["bottle"],
            batch_size=8,
            limit=None,
            save_maps=False,
            device="cpu",
        )
        self.assertEqual(config["threshold_strategy"], "continuous scores only; no test-label calibration")
        self.assertEqual(config["aupro"], "not implemented")
        self.assertEqual(
            configuration_hash({"a": 1, "b": [1, 2]}), configuration_hash({"b": [1, 2], "a": 1})
        )
        self.assertNotEqual(configuration_hash({"a": 1}), configuration_hash({"a": 2}))


def evaluate_category_broken_bank(case):
    """Helper: evaluate with an empty bank directory so the loader must fail."""
    from model.evaluation.evaluate import evaluate_category

    return evaluate_category(
        "bottle",
        processed_root=case.processed,
        bank_dir=case.root / "missing_banks",
        extractor=StubExtractor(),
        batch_size=2,
    )


if __name__ == "__main__":
    unittest.main()
