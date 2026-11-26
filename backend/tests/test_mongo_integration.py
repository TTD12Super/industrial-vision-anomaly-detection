"""MongoDB integration tests (REAL server; skipped when unavailable).

Uses the isolated ``industrial_anomaly_detection_test`` database — never the
development/production database — and removes its documents afterwards.
"""

import os
import unittest
from datetime import datetime, timezone

try:
    from pymongo import MongoClient

    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False


def _server_available() -> bool:
    if not HAS_PYMONGO:
        return False
    try:
        client = MongoClient(
            os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
            serverSelectionTimeoutMS=2000,
        )
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


@unittest.skipIf(not _server_available(), "MongoDB server is unavailable")
class TestMongoIntegration(unittest.TestCase):
    DB_NAME = "industrial_anomaly_detection_test"

    @classmethod
    def setUpClass(cls):
        from backend.database.repository import DetectionRepository, ensure_indexes

        uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        cls.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        cls.collection = cls.client[cls.DB_NAME]["detections"]
        ensure_indexes(cls.collection)
        cls.repo = DetectionRepository(cls.collection)

    @classmethod
    def tearDownClass(cls):
        cls.collection.delete_many({})
        cls.client.close()

    def tearDown(self):
        self.collection.delete_many({})

    def test_ping_and_indexes(self):
        version = self.client.server_info()["version"]
        self.assertTrue(version)
        names = [index["name"] for index in self.collection.list_indexes()]
        self.assertIn("prediction_id_1", names)

    def test_full_history_flow(self):
        for index in range(3):
            self.repo.create_detection(
                {
                    "prediction_id": f"itest-{index}",
                    "category": "bottle" if index < 2 else "cable",
                    "anomaly_score": 0.1 * (index + 1),
                    "processing_time_ms": 5,
                    "created_at": datetime.now(timezone.utc),
                    "result": {"input_url": "i", "heatmap_url": "h", "overlay_url": "o"},
                }
            )
        self.assertEqual(self.repo.count_detections(), 3)
        items, total = self.repo.list_detections(category="bottle", page=1, page_size=10)
        self.assertEqual(total, 2)
        ranged, total = self.repo.list_detections(min_score=0.2, max_score=0.3)
        self.assertEqual(total, 1)
        self.assertIsNotNone(self.repo.get_detection("itest-0"))
        self.assertTrue(self.repo.delete_detection("itest-0"))
        self.assertEqual(self.repo.count_detections(), 2)


if __name__ == "__main__":
    unittest.main()
