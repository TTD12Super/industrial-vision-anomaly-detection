from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from evaluate_weee_geometry_fair import enclosing_aabb, match_aabb, write_detector_crop_manifest  # noqa: E402


class FairGeometryTest(unittest.TestCase):
    def test_enclosing_aabb_projects_rotated_corners(self) -> None:
        self.assertEqual(enclosing_aabb([(10, 20), (50, 10), (60, 40), (20, 50)]), [
            [10.0, 10.0],
            [60.0, 10.0],
            [60.0, 50.0],
            [10.0, 50.0],
        ])

    def test_aabb_matching_is_one_to_one_and_uses_iou_floor(self) -> None:
        ground_truth = [{"corners": enclosing_aabb([(0, 0), (10, 0), (10, 10), (0, 10)])}]
        predictions = [
            {"confidence": 0.9, "corners": enclosing_aabb([(0, 0), (10, 0), (10, 10), (0, 10)])},
            {"confidence": 0.8, "corners": enclosing_aabb([(20, 20), (30, 20), (30, 30), (20, 30)])},
        ]
        result = match_aabb(ground_truth, predictions, iou_threshold=0.5)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["prediction_index"], 0)
        self.assertEqual(result["unmatched_prediction_indices"], [1])

    def test_crop_manifest_preserves_matched_missed_and_unmatched_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "source.png"
            Image.new("RGB", (40, 40), (120, 120, 120)).save(image_path)
            records = [{
                "image_id": 7,
                "file_name": "source.png",
                "path": image_path,
                "ground_truth": [
                    {"annotation_id": "gt-1", "corners": enclosing_aabb([(2, 2), (12, 2), (12, 12), (2, 12)])},
                    {"annotation_id": "gt-2", "corners": enclosing_aabb([(20, 20), (30, 20), (30, 30), (20, 30)])},
                ],
            }]
            predictions = [[
                {"confidence": 0.9, "corners": enclosing_aabb([(2, 2), (12, 2), (12, 12), (2, 12)])},
                {"confidence": 0.8, "corners": enclosing_aabb([(32, 2), (38, 2), (38, 8), (32, 8)])},
            ]]
            output = root / "manifest.jsonl"

            result = write_detector_crop_manifest(
                records, predictions, 0.5, "yolo_obb", output, "split-hash", "dataset-hash", "annotations-hash"
            )

            self.assertEqual(result["status_counts"], {
                "matched": 1,
                "missed_ground_truth": 1,
                "unmatched_prediction": 1,
            })
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({row["match_status"] for row in rows}, {
                "matched", "missed_ground_truth", "unmatched_prediction"
            })
            self.assertTrue((root / result["crop_root"] / "image_0000" / "prediction_0000.png").is_file())
