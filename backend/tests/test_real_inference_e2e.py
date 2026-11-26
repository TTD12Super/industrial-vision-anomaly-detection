"""REAL end-to-end inference test (NO mocks anywhere in this file).

Runs only when genuine prerequisites exist; otherwise SKIPPED with reasons:

* torch + real DINOv2 weights loadable
* ``artifacts/dinov2_knn/banks/bottle/feature_bank.pt`` built from train/good
* a real ``datasets/mvtec_ad_processed/bottle/test`` image
* a reachable MongoDB server (isolated ``..._test`` database, cleaned up)

Flow: POST /api/detect -> artifacts -> MongoDB -> history/detail ->
artifact downloads -> DELETE. Uses the real ``DetectorService``,
``DetectionService``, ``FileStorage`` and ``DetectionRepository``.
"""

import os
import tempfile
import unittest
from pathlib import Path

REASONS: list[str] = []

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    REASONS.append("torch not installed")

BANK_FILE = Path("artifacts/dinov2_knn/banks/bottle/feature_bank.pt")
if not BANK_FILE.is_file():
    REASONS.append(f"feature bank missing: {BANK_FILE}")

TEST_IMAGE: Path | None = None
try:
    from model.inference.config import SUPPORTED_EXTENSIONS

    test_dir = Path("datasets/mvtec_ad_processed/bottle/test")
    candidates = sorted(
        path
        for path in test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if candidates:
        TEST_IMAGE = candidates[0]
except Exception:
    TEST_IMAGE = None
if TEST_IMAGE is None:
    REASONS.append("no real bottle test image under datasets/mvtec_ad_processed")

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
try:
    from pymongo import MongoClient

    _probe = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    _probe.admin.command("ping")
    _probe.close()
    HAS_MONGO = True
except Exception:
    HAS_MONGO = False
    REASONS.append("MongoDB server unavailable")

TEST_DB = "industrial_anomaly_detection_test"


@unittest.skipIf(bool(REASONS), "real E2E prerequisites missing: " + "; ".join(REASONS))
class TestRealInferenceEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from fastapi.testclient import TestClient
        from PIL import Image

        from backend.database.repository import DetectionRepository
        from backend.dependencies import (
            get_detection_repository,
            get_detection_service,
            get_detector_service,
            get_optional_repository,
            get_storage,
        )
        from backend.main import app
        from backend.services.detection_service import DetectionService
        from backend.services.detector_service import DetectorService
        from backend.storage.file_storage import FileStorage

        cls.np, cls.Image = np, Image
        cls.tmp = tempfile.TemporaryDirectory()
        cls.storage = FileStorage(root=Path(cls.tmp.name) / "predictions")
        cls.detectors = DetectorService(bank_dir=BANK_FILE.parent.parent)
        cls.client_mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        cls.collection = cls.client_mongo[TEST_DB]["detections"]
        cls.repo = DetectionRepository(cls.collection)
        cls.service = DetectionService(
            detectors=cls.detectors, storage=cls.storage, repository=cls.repo
        )
        app.dependency_overrides[get_storage] = lambda: cls.storage
        app.dependency_overrides[get_detector_service] = lambda: cls.detectors
        app.dependency_overrides[get_detection_service] = lambda: cls.service
        app.dependency_overrides[get_detection_repository] = lambda: cls.repo
        app.dependency_overrides[get_optional_repository] = lambda: cls.repo
        cls.client = TestClient(app)
        cls.image_bytes = Path(TEST_IMAGE).read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.collection.delete_many({})
        cls.client_mongo.close()
        cls.tmp.cleanup()
        from backend.main import app as _app

        _app.dependency_overrides.clear()

    def test_real_end_to_end(self):
        # 1. Real inference through the API (no mocks).
        response = self.client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("real.png", self.image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        for key in ("prediction_id", "category", "anomaly_score", "processing_time_ms", "result"):
            self.assertIn(key, body)
        self.assertNotIn("is_anomaly", body)
        self.assertIsInstance(body["anomaly_score"], float)
        self.assertGreater(body["processing_time_ms"], 0)
        prediction_id = body["prediction_id"]

        # 2. Artifacts exist on disk and open as images.
        prediction_dir = self.storage.prediction_dir(prediction_id)
        for name in ("input.png", "heatmap.png", "overlay.png", "result.json"):
            self.assertTrue((prediction_dir / name).is_file(), name)
        for name in ("input.png", "heatmap.png", "overlay.png"):
            with self.Image.open(prediction_dir / name) as image:
                image.load()
                self.assertEqual(image.size, (256, 256))

        # 3. MongoDB record has the required fields.
        document = self.repo.get_detection(prediction_id)
        for key in (
            "prediction_id", "category", "anomaly_score",
            "processing_time_ms", "created_at", "result", "artifacts",
        ):
            self.assertIn(key, document)

        # 4. History + detail expose the record (no _id/paths).
        history = self.client.get("/api/detections").json()
        self.assertGreaterEqual(history["total"], 1)
        detail = self.client.get(f"/api/detections/{prediction_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn("_id", detail.json())

        # 5. Artifact downloads serve PNG bytes.
        for kind in ("input", "heatmap", "overlay"):
            artifact = self.client.get(f"/api/detections/{prediction_id}/{kind}")
            self.assertEqual(artifact.status_code, 200)
            self.assertEqual(artifact.headers["content-type"], "image/png")

        # 6. Delete removes record, files, and API visibility.
        deleted = self.client.delete(f"/api/detections/{prediction_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertIsNone(self.repo.get_detection(prediction_id))
        self.assertFalse(prediction_dir.exists())
        self.assertEqual(
            self.client.get(f"/api/detections/{prediction_id}").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
