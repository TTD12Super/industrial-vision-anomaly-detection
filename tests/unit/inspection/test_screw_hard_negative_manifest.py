from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_screw_hard_negative_manifest import build_manifest


class ScrewHardNegativeManifestTests(unittest.TestCase):
    def test_keeps_test_locked_and_splits_prior_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            fields = ["image_path", "label", "source_class", "split", "image_id", "capture_group_id", "sha256"]
            rows = []
            for index, (label, split) in enumerate([
                ("GOOD", "train"), ("GOOD", "train"), ("DEFECT", "train"), ("DEFECT", "train"),
                ("GOOD", "validation"), ("DEFECT", "validation"), ("GOOD", "test"), ("DEFECT", "test"),
            ]):
                rows.append({"image_path": f"{index}.png", "label": label, "source_class": "GOOD" if label == "GOOD" else "scratch_head", "split": split, "image_id": str(index), "capture_group_id": f"g{index}", "sha256": f"h{index}"})
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            audit = build_manifest(source, root / "out.csv", root / "audit.json", calibration_fraction=0.5, seed=42)
            self.assertEqual(audit["split_counts"]["test"], 2)
            self.assertTrue(audit["exact_content_disjoint"])
            with (root / "out.csv").open(newline="", encoding="utf-8") as handle:
                output = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in output if row["image_id"] in {"6", "7"}}, {"test"})


if __name__ == "__main__":
    unittest.main()
