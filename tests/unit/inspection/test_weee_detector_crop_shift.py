from __future__ import annotations

from unittest import TestCase

from scripts.evaluate_weee_detector_crop_shift import (
    _load_detector_crop_manifest,
    match_objects,
    polygon_iou,
    score_comparison_summary,
)

from PIL import Image


class WEEEDetectorCropShiftTests(TestCase):
    def test_polygon_iou_is_one_for_identical_boxes_and_zero_when_disjoint(self) -> None:
        box = [[0, 0], [10, 0], [10, 4], [0, 4]]
        other = [[20, 20], [30, 20], [30, 24], [20, 24]]

        self.assertAlmostEqual(polygon_iou(box, box), 1.0)
        self.assertEqual(polygon_iou(box, other), 0.0)

    def test_matching_reports_unmatched_ground_truth_and_predictions(self) -> None:
        matching = match_objects(
            [{"annotation_id": 1, "corners": [[0, 0], [10, 0], [10, 4], [0, 4]]}],
            [
                {"prediction_index": 0, "corners": [[0, 0], [10, 0], [10, 4], [0, 4]]},
                {"prediction_index": 1, "corners": [[20, 20], [30, 20], [30, 24], [20, 24]]},
            ],
            iou_threshold=0.5,
        )

        self.assertEqual(len(matching["matches"]), 1)
        self.assertAlmostEqual(matching["matches"][0]["crop_iou"], 1.0)
        self.assertEqual(matching["unmatched_gt_indices"], [])
        self.assertEqual(matching["unmatched_prediction_indices"], [1])

    def test_score_report_is_stress_only_and_has_no_defect_metric(self) -> None:
        pairs = [
            {
                "crop_iou": 0.8,
                "oracle_score": {"max": 0.2, "top_1pct": 0.1},
                "actual_score": {"max": 0.4, "top_1pct": 0.3},
            },
            {
                "crop_iou": 0.6,
                "oracle_score": {"max": 0.3, "top_1pct": 0.2},
                "actual_score": {"max": 0.2, "top_1pct": 0.25},
            },
        ]

        summary = score_comparison_summary(pairs, "max", threshold=0.25)

        self.assertEqual(summary["matched_pair_count"], 2)
        self.assertEqual(summary["actual_detector_crop"]["above_threshold_count"], 1)
        self.assertAlmostEqual(summary["score_shift_actual_minus_oracle"]["mean"], 0.05)
        self.assertNotIn("defect_recall", summary)

    def test_crop_manifest_loads_saved_crops_and_preserves_misses(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            crop = root / "crop.png"
            Image.new("RGB", (40, 40), (120, 120, 120)).save(source)
            Image.new("RGB", (10, 10), (140, 140, 140)).save(crop)
            manifest = root / "crops.jsonl"
            rows = [
                {
                    "source_image_id": 7,
                    "source_file_name": "source.png",
                    "detector_name": "yolo",
                    "confidence_threshold": 0.4,
                    "prediction_index": 0,
                    "ground_truth_index": 0,
                    "ground_truth_object_id": "gt-1",
                    "ground_truth_box": [[2, 2], [12, 2], [12, 12], [2, 12]],
                    "prediction_confidence": 0.9,
                    "match_iou": 0.8,
                    "crop_path": "crop.png",
                    "match_status": "matched",
                    "split_manifest_sha256": "split",
                    "test_annotations_sha256": "annotations-hash",
                },
                {
                    "source_image_id": 7,
                    "source_file_name": "source.png",
                    "detector_name": "yolo",
                    "confidence_threshold": 0.4,
                    "prediction_index": None,
                    "ground_truth_index": 1,
                    "ground_truth_object_id": "gt-2",
                    "ground_truth_box": [[20, 20], [30, 20], [30, 30], [20, 30]],
                    "prediction_confidence": None,
                    "match_iou": None,
                    "crop_path": None,
                    "match_status": "missed_ground_truth",
                    "split_manifest_sha256": "split",
                    "test_annotations_sha256": "annotations-hash",
                },
                {
                    "source_image_id": 7,
                    "source_file_name": "source.png",
                    "detector_name": "yolo",
                    "confidence_threshold": 0.4,
                    "prediction_index": 1,
                    "ground_truth_index": None,
                    "ground_truth_object_id": None,
                    "ground_truth_box": None,
                    "prediction_confidence": 0.7,
                    "match_iou": None,
                    "crop_path": None,
                    "match_status": "unmatched_prediction",
                    "split_manifest_sha256": "split",
                    "test_annotations_sha256": "annotations-hash",
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            loaded = _load_detector_crop_manifest(
                manifest,
                [{
                    "image_id": 7,
                    "file_name": "source.png",
                    "path": source,
                    "ground_truth": [
                        {"annotation_id": "gt-1", "corners": rows[0]["ground_truth_box"]},
                        {"annotation_id": "gt-2", "corners": rows[1]["ground_truth_box"]},
                    ],
                }],
                "annotations-hash",
            )

            self.assertEqual(len(loaded["matched_pairs"]), 1)
            self.assertEqual(len(loaded["unmatched_gt"]), 1)
            self.assertEqual(len(loaded["unmatched_predictions"]), 1)
            self.assertEqual(loaded["total_predictions"], 2)
            self.assertTrue(loaded["matched_pairs"][0]["actual_crop"].size > 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
