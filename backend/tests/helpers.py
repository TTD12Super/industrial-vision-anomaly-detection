"""Shared fixtures for backend API tests."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.database.repository import DetectionRepository
from backend.dependencies import (
    get_app_settings,
    get_database_status,
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

from .fakes import FakeCollection


def make_png_bytes(color=(200, 30, 30), size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


class MockDetector:
    """MOCK inference for API tests only.

    Returns fixed arrays without running DINOv2/kNN. Proves API behavior,
    validation, storage, and schemas — NOT real AI inference.
    """

    def predict(self, image):
        heatmap = np.full((256, 256, 3), 128, dtype=np.uint8)
        overlay = np.full((256, 256, 3), 64, dtype=np.uint8)
        return {"anomaly_score": 0.42, "heatmap": heatmap, "overlay": overlay}


class MockDetectorService(DetectorService):
    """MOCK detector service: skips bank files and returns MockDetector."""

    def __init__(self) -> None:
        super().__init__(bank_dir=Path(tempfile.gettempdir()))

    def get_detector(self, category: str):
        from model.inference.config import validate_category

        validate_category(category)
        return MockDetector()


class BackendTestCase(unittest.TestCase):
    """TestCase with an isolated app (tmp storage, fake history, overridable settings)."""

    def make_client(
        self,
        detector_service=None,
        max_upload_size_mb: int = 10,
        db_status: str = "connected",
    ) -> tuple[TestClient, FileStorage, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        storage = FileStorage(root=Path(tmp.name) / "predictions")
        settings = Settings(
            bank_dir=Path(tmp.name) / "banks",
            predictions_dir=Path(tmp.name) / "predictions",
            max_upload_size_mb=max_upload_size_mb,
        )
        detectors = detector_service or MockDetectorService()
        # REAL repository code backed by an in-memory fake collection:
        # no MongoDB server is needed for these tests.
        self.repo = DetectionRepository(FakeCollection())
        service = DetectionService(detectors=detectors, storage=storage, repository=self.repo)
        app.dependency_overrides[get_app_settings] = lambda: settings
        app.dependency_overrides[get_storage] = lambda: storage
        app.dependency_overrides[get_detector_service] = lambda: detectors
        app.dependency_overrides[get_detection_service] = lambda: service
        app.dependency_overrides[get_detection_repository] = lambda: self.repo
        app.dependency_overrides[get_optional_repository] = lambda: self.repo
        app.dependency_overrides[get_database_status] = lambda: db_status
        self.addCleanup(app.dependency_overrides.clear)
        self.addCleanup(tmp.cleanup)
        return TestClient(app), storage, tmp
