"""Tests for Python-generated dataset charts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mvtec_system import plot_dataset_distribution


class VisualizationTests(unittest.TestCase):
    def test_dataset_distribution_chart_is_saved(self) -> None:
        report = {
            "categories": ["bottle", "cable"],
            "counts": {
                "bottle": {"train": 2, "test": 3, "ground_truth": 1},
                "cable": {"train": 4, "test": 5, "ground_truth": 2},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = plot_dataset_distribution(
                report, Path(directory) / "dataset_split_sizes.png"
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
