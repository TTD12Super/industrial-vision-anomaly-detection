"""Tests for full MVTec product-category test discovery and routing outputs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from mvtec_system import discover_mvtec_test_images, product_category_from_mvtec_path


class ProductTestEvaluationTests(unittest.TestCase):
    def test_category_is_derived_from_directory_before_test(self) -> None:
        path = Path("datasets/mvtec_ad_processed/bottle/test/good/000.png")
        self.assertEqual(product_category_from_mvtec_path(path), "bottle")

    def test_discovery_includes_good_and_defect_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "bottle/test/good/000.png",
                "bottle/test/broken_large/001.png",
                "cable/test/good/000.png",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (256, 256), color=(255, 255, 255)).save(path)
            records = discover_mvtec_test_images(root)

        self.assertEqual(len(records), 3)
        self.assertEqual({record["true_category"] for record in records}, {"bottle", "cable"})
        self.assertEqual(sum(record["is_defect"] for record in records), 1)
        defect = next(record for record in records if record["is_defect"])
        self.assertEqual(defect["defect_type"], "broken_large")


if __name__ == "__main__":
    unittest.main()
