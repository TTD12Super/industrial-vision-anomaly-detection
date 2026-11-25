from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from inspection.interfaces import Localization
from inspection.segad_lite import (
    classification_metrics,
    derive_object_interior_mask,
    evaluate_segad_lite,
    fit_validation_logistic,
    masked_heatmap_stats,
)
from inspection.segad_lite import _samples


class _StaticLocalizer:
    def localize(self, crop: np.ndarray) -> Localization:
        value = float(crop.mean() / 255.0)
        return Localization(np.full(crop.shape[:2], value, dtype=np.float32), None, value, "completed")


class SegADLiteTests(TestCase):
    def test_interior_mask_is_a_conservative_central_support(self) -> None:
        mask = derive_object_interior_mask((10, 8), border_fraction=0.20)

        self.assertEqual(mask.shape, (10, 8))
        self.assertEqual(int(mask.sum()), 24)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[8, 2])

    def test_masked_heatmap_stats_ignore_border_and_report_shape_stats(self) -> None:
        heatmap = np.zeros((6, 6), dtype=np.float32)
        heatmap[2:4, 2:4] = 1.0
        mask = derive_object_interior_mask((6, 6), border_fraction=0.10)

        stats = masked_heatmap_stats(heatmap, mask)

        self.assertEqual(stats["masked_mean"], 0.25)
        self.assertEqual(stats["masked_max"], 1.0)
        self.assertEqual(stats["masked_area_fraction_at_p90"], 0.25)
        self.assertGreater(stats["masked_compactness_at_p90"], 0.0)

    def test_missing_mask_means_zero_truth_for_good_but_unavailable_for_defect(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            objects = []
            for index, label in enumerate(("GOOD", "DEFECT"), 1):
                image_id = f"image-{index}"
                relative = Path(f"{image_id}.png")
                Image.fromarray(np.full((12, 12, 3), 100, dtype=np.uint8)).save(root / relative)
                images.append({"image_id": image_id, "split": "validation", "relative_path": relative.as_posix()})
                objects.append(
                    {
                        "object_id": f"object-{index}",
                        "image_id": image_id,
                        "quality_label": label,
                        "obb_xyxyxyxy": [[2, 2], [10, 2], [10, 10], [2, 10]],
                    }
                )
            config = SimpleNamespace(
                dataset_path=lambda relative: root / relative,
            )
            with patch("inspection.segad_lite.load_dataset_records", return_value=(images, objects)):
                samples = _samples(config, "validation", _StaticLocalizer(), None, 0.10)

            self.assertTrue(np.array_equal(samples[0]["truth"], np.zeros_like(samples[0]["interior_mask"])))
            self.assertIsNone(samples[1]["truth"])

    def test_logistic_threshold_is_fit_on_validation_and_locked_for_test(self) -> None:
        validation_features = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
        validation_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        calibration = fit_validation_logistic(validation_features, validation_labels, target_defect_recall=1.0)
        test_features = np.asarray([[0.05], [0.95]], dtype=np.float32)
        test_scores = calibration["model"].predict_proba(test_features)[:, 1]

        test_metrics = classification_metrics(np.asarray([0, 1]), test_scores, calibration["threshold"])

        self.assertEqual(calibration["validation_metrics"]["defect_recall_at_threshold"], 1.0)
        self.assertEqual(test_metrics["threshold"], calibration["threshold"])
        self.assertIn(test_metrics["defect_recall_at_threshold"], (0.0, 1.0))

    def test_report_omits_pixel_metrics_without_complete_ground_truth_masks(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            objects = []
            for index, (split, label, value) in enumerate(
                (("validation", "GOOD", 20), ("validation", "DEFECT", 220), ("test", "GOOD", 20), ("test", "DEFECT", 220)),
                1,
            ):
                image_id = f"image-{index}"
                relative = Path(f"{image_id}.png")
                Image.fromarray(np.full((12, 12, 3), value, dtype=np.uint8)).save(root / relative)
                images.append({"image_id": image_id, "split": split, "relative_path": relative.as_posix()})
                objects.append(
                    {
                        "object_id": f"object-{index}",
                        "image_id": image_id,
                        "quality_label": label,
                        "obb_xyxyxyxy": [[2, 2], [10, 2], [10, 10], [2, 10]],
                    }
                )
            config = SimpleNamespace(
                product_key="screw",
                source_path=root / "configs/products/screw.yaml",
                image_manifest=Path("manifests/images.jsonl"),
                object_manifest=Path("manifests/objects.jsonl"),
                dataset_path=lambda relative: root / relative,
            )
            report_path = root / "segad_lite.json"
            with patch("inspection.segad_lite.load_dataset_records", return_value=(images, objects)):
                report = evaluate_segad_lite(
                    config,
                    report_path,
                    localizer=_StaticLocalizer(),
                )

            self.assertIsNone(report["validation"]["pixel_metrics"])
            self.assertIsNone(report["test"]["pixel_metrics"])
            self.assertTrue(any("ground-truth masks" in warning for warning in report["warnings"]))
            self.assertEqual(report["provenance"]["detector_used"], False)
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    import unittest

    unittest.main()
