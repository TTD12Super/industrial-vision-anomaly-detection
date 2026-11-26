"""Repository unit tests over the in-memory fake collection (no server)."""

import unittest
from datetime import datetime, timezone

from backend.database.repository import (
    DetectionRepository,
    ensure_indexes,
    serialize_detection,
)

from .fakes import DuplicateKeyError, FakeCollection


def _doc(prediction_id, category="bottle", score=0.1, day=1):
    return {
        "prediction_id": prediction_id,
        "category": category,
        "anomaly_score": score,
        "processing_time_ms": 10,
        "created_at": datetime(2026, 9, day, tzinfo=timezone.utc),
        "result": {"input_url": "x", "heatmap_url": "y", "overlay_url": "z"},
    }


class TestRepository(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.repo = DetectionRepository(self.collection)

    def test_create_and_get(self):
        self.repo.create_detection(_doc("a"))
        found = self.repo.get_detection("a")
        self.assertEqual(found["category"], "bottle")
        self.assertTrue(self.repo.exists("a"))
        self.assertFalse(self.repo.exists("missing"))
        self.assertIsNone(self.repo.get_detection("missing"))

    def test_duplicate_prediction_id_rejected(self):
        self.repo.create_detection(_doc("a"))
        with self.assertRaises(DuplicateKeyError):
            self.repo.create_detection(_doc("a"))

    def test_list_sort_and_pagination(self):
        for day, pid in enumerate(["a", "b", "c"], start=1):
            self.repo.create_detection(_doc(pid, day=day))
        items, total = self.repo.list_detections(page=1, page_size=10)
        self.assertEqual(total, 3)
        self.assertEqual([item["prediction_id"] for item in items], ["c", "b", "a"])
        ascending, _ = self.repo.list_detections(page=1, page_size=10, sort_order="asc")
        self.assertEqual([item["prediction_id"] for item in ascending], ["a", "b", "c"])
        page2, _ = self.repo.list_detections(page=2, page_size=2)
        self.assertEqual([item["prediction_id"] for item in page2], ["a"])

    def test_filters_and_count(self):
        self.repo.create_detection(_doc("a", category="bottle", score=0.1))
        self.repo.create_detection(_doc("b", category="cable", score=0.9))
        items, total = self.repo.list_detections(category="cable")
        self.assertEqual(total, 1)
        self.assertEqual(items[0]["prediction_id"], "b")
        ranged, total = self.repo.list_detections(min_score=0.5, max_score=1.0)
        self.assertEqual(total, 1)
        self.assertEqual(self.repo.count_detections(min_score=0.0), 2)
        with self.assertRaises(ValueError):
            self.repo.list_detections(page=0)

    def test_delete(self):
        self.repo.create_detection(_doc("a"))
        self.assertTrue(self.repo.delete_detection("a"))
        self.assertFalse(self.repo.delete_detection("a"))

    def test_ensure_indexes(self):
        ensure_indexes(self.collection)
        fields = [keys[0][0] for keys, _ in self.collection.created_indexes]
        self.assertIn("prediction_id", fields)
        unique = [
            options.get("unique")
            for keys, options in self.collection.created_indexes
            if keys[0][0] == "prediction_id"
        ]
        self.assertEqual(unique, [True])

    def test_serialize_hides_internals(self):
        document = _doc("a")
        document["_id"] = "internal"
        document["artifacts"] = {"input_path": "/secret/path"}
        serialized = serialize_detection(document)
        self.assertNotIn("_id", serialized)
        self.assertNotIn("artifacts", serialized)
        self.assertTrue(serialized["created_at"].endswith("Z"))
        self.assertIsNone(serialize_detection(None))


if __name__ == "__main__":
    unittest.main()
