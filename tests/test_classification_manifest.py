"""Validation tests for the deterministic classification split manifest."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import random
import unittest


class ClassificationManifestTests(unittest.TestCase):
    processed = Path("datasets/mvtec_ad_processed")
    manifest = Path("manifests/classification_split_v1.csv")
    seed = 42
    validation_fraction = 0.30

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.processed.is_dir():
            raise unittest.SkipTest("processed MVTec dataset is not available")
        if not cls.manifest.is_file():
            raise unittest.SkipTest("classification manifest is not available")
        with cls.manifest.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_schema_and_total_row_count(self):
        self.assertEqual(
            set(self.rows[0]), {"image_path", "category", "split"}
        )
        expected_count = sum(
            len(list((self.processed / category / "train" / "good").glob("*.png")))
            for category in self.categories()
        )
        self.assertEqual(len(self.rows), expected_count)

    def test_paths_are_existing_train_good_images(self):
        for row in self.rows:
            relative_path = Path(row["image_path"])
            path = self.processed / relative_path
            self.assertFalse(relative_path.is_absolute())
            self.assertEqual(relative_path.parts[-4:-1], (row["category"], "train", "good"))
            self.assertTrue(path.is_file(), path)
            self.assertNotIn("test", relative_path.parts)

    def test_each_category_has_disjoint_train_and_validation_splits(self):
        grouped = defaultdict(lambda: defaultdict(set))
        for row in self.rows:
            grouped[row["category"]][row["split"]].add(row["image_path"])

        self.assertEqual(set(grouped), set(self.categories()))
        for category in self.categories():
            train_paths = grouped[category]["train"]
            validation_paths = grouped[category]["validation"]
            self.assertTrue(train_paths)
            self.assertTrue(validation_paths)
            self.assertTrue(train_paths.isdisjoint(validation_paths))

    def test_seed_42_recreates_the_same_assignment(self):
        actual = {(row["category"], row["image_path"]): row["split"] for row in self.rows}
        expected = {}
        rng = random.Random(self.seed)

        for category in self.categories():
            paths = sorted((self.processed / category / "train" / "good").glob("*.png"))
            shuffled = list(paths)
            rng.shuffle(shuffled)
            validation_count = max(1, int(round(len(shuffled) * self.validation_fraction)))
            validation_paths = set(shuffled[:validation_count])
            for path in paths:
                relative_path = path.relative_to(self.processed).as_posix()
                expected[(category, relative_path)] = (
                    "validation" if path in validation_paths else "train"
                )

        self.assertEqual(actual, expected)

    def categories(self):
        return sorted(
            path.name
            for path in self.processed.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )


if __name__ == "__main__":
    unittest.main()
