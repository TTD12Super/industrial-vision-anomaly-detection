"""Read-only checks for the local MVTec dataset and processed baseline."""

from __future__ import annotations

from pathlib import Path
import unittest

from PIL import Image


EXPECTED_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPOSITORY_ROOT / "datasets" / "mvtec_ad"
PROCESSED_ROOT = REPOSITORY_ROOT / "datasets" / "mvtec_ad_processed"


class DatasetValidationTests(unittest.TestCase):
    def test_raw_and_processed_roots_discover_all_categories(self) -> None:
        raw_categories = {path.name for path in RAW_ROOT.iterdir() if path.is_dir()}
        processed_categories = {
            path.name for path in PROCESSED_ROOT.iterdir() if path.is_dir()
        }

        self.assertEqual(raw_categories, EXPECTED_CATEGORIES)
        self.assertEqual(processed_categories, EXPECTED_CATEGORIES)

    def test_processed_images_and_one_mask_match_baseline_shape(self) -> None:
        for category in sorted(EXPECTED_CATEGORIES):
            train_image = next((PROCESSED_ROOT / category / "train" / "good").glob("*.png"))
            test_root = PROCESSED_ROOT / category / "test"
            defect_directory = next(
                path for path in test_root.iterdir() if path.is_dir() and path.name != "good"
            )
            defect_image = next(defect_directory.glob("*.png"))
            mask_path = (
                PROCESSED_ROOT
                / category
                / "ground_truth"
                / defect_directory.name
                / f"{defect_image.stem}_mask.png"
            )

            with Image.open(train_image) as image, Image.open(defect_image) as defect, Image.open(mask_path) as mask:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (256, 256))
                self.assertEqual(defect.mode, "RGB")
                self.assertEqual(defect.size, (256, 256))
                self.assertEqual(mask.size, (256, 256))


if __name__ == "__main__":
    unittest.main()
