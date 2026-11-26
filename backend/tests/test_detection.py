"""Detection API + storage tests.

Inference here is MOCKED (see helpers.MockDetector): these tests verify API
behavior, validation, schemas, and file storage — not real AI inference.
The missing-bank test uses the real DetectorService path resolution without
loading any model.
"""

import numpy as np

from backend.services.detection_service import DetectionService
from backend.services.detector_service import DetectorService

from .helpers import BackendTestCase, make_png_bytes


class TestDetectValidation(BackendTestCase):
    def test_invalid_category_returns_400(self):
        client, _, _ = self.make_client()
        response = client.post(
            "/api/detect",
            data={"category": "phone"},
            files={"file": ("img.png", make_png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported category", response.json()["detail"])

    def test_missing_file_returns_422(self):
        client, _, _ = self.make_client()
        response = client.post("/api/detect", data={"category": "bottle"})
        self.assertEqual(response.status_code, 422)

    def test_oversize_file_returns_413(self):
        client, _, _ = self.make_client(max_upload_size_mb=0)
        response = client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("img.png", make_png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 413)

    def test_unsupported_extension_returns_400(self):
        client, _, _ = self.make_client()
        response = client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("img.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_undecodable_image_returns_400(self):
        client, _, _ = self.make_client()
        response = client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("img.png", b"\x00" * 100, "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_bank_returns_503(self):
        # Real DetectorService against an empty bank dir: no model is loaded,
        # the request must fail with a clear "not available" error.
        import tempfile
        from pathlib import Path

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        real_detectors = DetectorService(bank_dir=Path(tmp.name) / "banks")
        client, _, _ = self.make_client(detector_service=real_detectors)
        response = client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("img.png", make_png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("not available", response.json()["detail"])


class TestDetectSuccessMock(BackendTestCase):
    """MOCK inference: verifies the full API flow with a fake detector."""

    def test_detect_and_artifacts(self):
        client, storage, _ = self.make_client()
        response = client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("img.png", make_png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category"], "bottle")
        self.assertAlmostEqual(body["anomaly_score"], 0.42)
        self.assertNotIn("is_anomaly", body)
        prediction_id = body["prediction_id"]
        for key in ("input_url", "heatmap_url", "overlay_url"):
            self.assertTrue(body["result"][key].startswith(f"/api/detections/{prediction_id}/"))
            artifact = client.get(body["result"][key])
            self.assertEqual(artifact.status_code, 200)
            self.assertEqual(artifact.headers["content-type"], "image/png")
        # Metadata persisted alongside artifacts.
        self.assertTrue((storage.prediction_dir(prediction_id) / "result.json").is_file())

    def test_unknown_prediction_returns_404(self):
        client, _, _ = self.make_client()
        self.assertEqual(client.get("/api/detections/" + "f" * 32 + "/heatmap").status_code, 404)
        self.assertEqual(
            client.get("/api/detections/does-not-exist/heatmap").status_code, 404
        )

    def test_unknown_kind_returns_404(self):
        client, _, _ = self.make_client()
        response = client.post(
            "/api/detect",
            data={"category": "bottle"},
            files={"file": ("img.png", make_png_bytes(), "image/png")},
        )
        prediction_id = response.json()["prediction_id"]
        self.assertEqual(
            client.get(f"/api/detections/{prediction_id}/nope").status_code, 404
        )


class TestFileStorage(BackendTestCase):
    def test_save_and_retrieve(self):
        _, storage, _ = self.make_client()
        from backend.storage.file_storage import new_prediction_id, validate_prediction_id

        prediction_id = new_prediction_id()
        array = np.zeros((8, 8, 3), dtype=np.uint8)
        storage.save_rgb_array(prediction_id, "heatmap", array)
        self.assertTrue(storage.get_file(prediction_id, "heatmap").is_file())
        with self.assertRaises(FileNotFoundError):
            storage.get_file(prediction_id, "overlay")
        with self.assertRaises(FileNotFoundError):
            validate_prediction_id("../escape")
        with self.assertRaises(ValueError):
            storage.save_rgb_array(prediction_id, "input", array)


class TestDetectorServicePaths(BackendTestCase):
    def test_bank_path_uses_phase1_logic(self):
        import tempfile
        from pathlib import Path

        service = DetectorService(bank_dir=Path(tempfile.gettempdir()) / "banks")
        self.assertEqual(service.bank_path("bottle").name, "feature_bank.pt")
        self.assertFalse(service.is_bank_available("bottle"))
        with self.assertRaises(ValueError):
            service.bank_path("phone")
