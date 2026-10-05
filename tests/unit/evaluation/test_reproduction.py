"""Tests for saved-baseline metric reproduction."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from mvtec_system import reproduce_saved_baseline, save_reproduction_report


class ReproductionTests(unittest.TestCase):
    def test_reproduces_saved_scores_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            processed_root = root / "processed"
            category_dir = run_root / "bottle"
            category_dir.mkdir(parents=True)
            (processed_root / "bottle" / "ground_truth" / "scratch").mkdir(parents=True)
            rows = [
                ["category", "image_path", "defect_type", "is_anomaly", "anomaly_score"],
                ["bottle", "bottle/test/good/000.png", "good", "0", "0.1"],
                ["bottle", "bottle/test/scratch/000.png", "scratch", "1", "0.9"],
            ]
            with (category_dir / "image_scores.csv").open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
                processed_root / "bottle" / "ground_truth" / "scratch" / "000_mask.png"
            )
            Image.fromarray(np.ones((2, 2), dtype=np.uint8) * 255).save(
                processed_root / "bottle" / "ground_truth" / "scratch" / "000_mask.png"
            )
            patch_scores = np.asarray([[0.1], [0.9]], dtype=np.float32)
            anomaly_maps = np.asarray(
                [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
            )
            np.savez(category_dir / "anomaly_maps.npz", patch_scores=patch_scores, anomaly_maps=anomaly_maps)
            report = reproduce_saved_baseline(run_root, processed_root)
            output = root / "report.json"
            save_reproduction_report(output, report)
            saved_report = output.read_text(encoding="utf-8")

        self.assertEqual(report["category_count"], 1)
        self.assertTrue(report["all_image_score_reconstructions_within_tolerance"])
        self.assertEqual(report["observed_macro"]["image_auroc"], 1.0)
        self.assertIn('"observed_macro"', saved_report)


if __name__ == "__main__":
    unittest.main()
