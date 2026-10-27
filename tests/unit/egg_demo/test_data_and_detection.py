from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image

from egg_demo.data import inspect_quality_archive, prepare_quality_dataset, validate_yolo_dataset, write_local_yolo_yaml
from egg_demo.detection import EggDetection, crop_detection


class EggDemoDataTests(unittest.TestCase):
    def test_validate_yolo_and_write_local_config(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "data/images/train"
            label_dir = root / "data/labels/train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (100, 80), "white").save(image_dir / "one.jpg")
            (label_dir / "one.txt").write_text("\ufeff0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (root / "data/images/val").mkdir(parents=True)
            (root / "data/labels/val").mkdir(parents=True)
            (root / "data/data.yaml").write_text("names:\n  0: egg\n", encoding="utf-8")
            result = validate_yolo_dataset(root)
            self.assertEqual(result["splits"]["train"]["boxes"], 1)
            config = write_local_yolo_yaml(root, root / "local.yaml")
            self.assertIn(f"path: {root.resolve()}", config.read_text())

    def test_quality_archive_is_image_level_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "quality.zip"
            with ZipFile(archive, "w", ZIP_DEFLATED) as output:
                for label in ("Good Eggs", "Bad Eggs"):
                    image_path = root / f"{label}.jpg"
                    Image.new("RGB", (32, 32), "white").save(image_path)
                    output.write(image_path, f"Original Images(Eggs)/{label}/{label}.jpg")
            report = inspect_quality_archive(archive)
            self.assertEqual(report["image_counts"], {"good": 1, "bad": 1})
            self.assertFalse(report["has_pixel_masks"])
            prepared = prepare_quality_dataset(archive, root / "processed", image_size=16)
            self.assertEqual(prepared["image_count"], 2)
            self.assertTrue((root / "processed/manifest.csv").is_file())

    def test_crop_clips_to_image(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        crop = crop_detection(image, EggDetection((-10, -5, 50, 45), 0.9, 0, "egg"), margin=0)
        self.assertEqual(crop.size, (50, 45))


if __name__ == "__main__":
    unittest.main()
