"""Tests for the Python MVTec preprocessing workflow."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from mvtec_system import (
    create_classification_manifest,
    dataset_manifest,
    discover_dataset_root,
    preprocess_dataset,
    prepare_mvtec_dataset,
    validate_processed_dataset,
    validate_raw_dataset,
)


class PreprocessingTests(unittest.TestCase):
    def make_raw_dataset(self) -> tuple[Path, Path]:
        root = Path(tempfile.mkdtemp())
        raw = root / "raw"
        category = raw / "bottle"
        for relative, color in (
            ("train/good/000.png", (10, 20, 30)),
            ("test/good/000.png", (10, 20, 30)),
            ("test/scratch/000.png", (200, 20, 30)),
            ("ground_truth/scratch/000_mask.png", 255),
        ):
            path = category / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "L" if "mask" in path.name else "RGB"
            Image.new(mode, (8, 6), color=color).save(path)
        return root, raw

    def test_validation_and_preprocessing_preserve_raw_manifest(self) -> None:
        root, raw = self.make_raw_dataset()
        processed = root / "processed"
        before = dataset_manifest(raw)

        raw_report = validate_raw_dataset(raw)
        self.assertTrue(raw_report["valid"])
        self.assertEqual(raw_report["categories"], ["bottle"])

        first = prepare_mvtec_dataset(raw, processed, image_size=16)
        self.assertTrue(first["processing"]["raw_dataset_unchanged"])
        self.assertEqual(first["processing"]["created"], 4)
        self.assertTrue(validate_processed_dataset(raw, processed, 16)["valid"])
        self.assertEqual(before, dataset_manifest(raw))

        second = prepare_mvtec_dataset(raw, processed, image_size=16)
        self.assertEqual(second["processing"]["created"], 0)
        self.assertEqual(second["processing"]["reused"], 4)

        with Image.open(processed / "bottle/train/good/000.png") as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (16, 16))
        with Image.open(processed / "bottle/ground_truth/scratch/000_mask.png") as mask:
            self.assertEqual(mask.size, (16, 16))

    def test_missing_defect_mask_is_reported(self) -> None:
        root, raw = self.make_raw_dataset()
        mask = raw / "bottle/ground_truth/scratch/000_mask.png"
        mask.unlink()
        report = validate_raw_dataset(raw)
        self.assertFalse(report["valid"])
        self.assertTrue(report["failures"]["ground_truth"])

    def test_dataset_root_is_discovered_without_a_fixed_dataset_name(self) -> None:
        root, raw = self.make_raw_dataset()
        datasets = root / "datasets"
        datasets.mkdir()
        renamed = datasets / "customer_images"
        raw.rename(renamed)
        (datasets / "customer_images_processed").mkdir()
        self.assertEqual(discover_dataset_root(datasets), renamed.resolve())

    def test_manifest_uses_only_train_good_paths(self) -> None:
        root, raw = self.make_raw_dataset()
        processed = root / "processed"
        preprocess_dataset(raw, processed, image_size=16)
        report = create_classification_manifest(processed, root / "split.csv", 0.30, 7)
        self.assertEqual(report["categories"], ["bottle"])
        self.assertEqual(report["row_count"], 1)
        self.assertIn("bottle/train/good/000.png", (root / "split.csv").read_text())


if __name__ == "__main__":
    unittest.main()
