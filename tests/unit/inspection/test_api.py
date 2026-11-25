from io import BytesIO
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from inspection.api import InspectionService, create_app
from inspection.backends import StaticClassifier, StaticDetector, StaticLocalizer
from inspection.interfaces import Detection
from inspection.pipeline import InspectionPipeline
from inspection.policy import DecisionPolicy
from inspection.reporting import AsyncReportManager, MockReporter
from inspection.runtime import ProductRegistry


class ApiTests(unittest.TestCase):
    def test_upload_status_and_allowlisted_artifact_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs" / "products"
            config_dir.mkdir(parents=True)
            (config_dir / "widget.yaml").write_text(
                """schema_version: 1
product_key: widget
dataset:
  root: datasets/products/widget
  image_manifest: manifests/images.jsonl
  object_manifest: manifests/objects.jsonl
  split_policy: grouped
models:
  detector: {backend: yolo_obb}
  classifier: {backend: dinov2_logreg}
  localizer: {backend: anomalyclip}
  reporter: {backend: mock}
policy: {quality_labels: [GOOD, DEFECT]}
outputs: {json: true}
""",
                encoding="utf-8",
            )

            def factory(config: object) -> InspectionPipeline:
                return InspectionPipeline(
                    "widget",
                    "test-bundle",
                    StaticDetector([Detection(np.asarray([[4, 4], [28, 4], [28, 28], [4, 28]], dtype=np.float32), 0.99)]),
                    StaticClassifier(0.9),
                    DecisionPolicy(0.1, 0.05),
                    StaticLocalizer(np.ones((4, 4), dtype=np.float32)),
                    root / "runs",
                    AsyncReportManager(MockReporter()),
                )

            service = InspectionService(ProductRegistry(config_dir), factory)
            client = TestClient(create_app(config_dir, service))
            response = client.post(
                "/api/v1/inspections",
                data={"product_key": "widget"},
                files={"file": ("widget.png", self._png(), "image/png")},
            )
            self.assertEqual(response.status_code, 202)
            inspection_id = response.json()["inspection_id"]
            result = {"status": "processing"}
            for _ in range(40):
                result = client.get(f"/api/v1/inspections/{inspection_id}").json()
                if result["status"] != "processing":
                    break
                time.sleep(0.01)
            self.assertIn(result["status"], {"completed", "completed_with_pending_reports"})
            self.assertEqual(result["objects"][0]["classifier_label"], "DEFECT")
            artifacts = client.get(f"/api/v1/inspections/{inspection_id}/artifacts")
            self.assertEqual(artifacts.status_code, 200)
            self.assertTrue(artifacts.json())
            artifact_id = artifacts.json()[0]["artifact_id"]
            self.assertEqual(client.get(f"/api/v1/inspections/{inspection_id}/artifacts/{artifact_id}").status_code, 200)
            self.assertEqual(client.get(f"/api/v1/inspections/{inspection_id}/artifacts/../../etc/passwd").status_code, 404)

    @staticmethod
    def _png() -> bytes:
        stream = BytesIO()
        Image.new("RGB", (32, 32), "white").save(stream, format="PNG")
        return stream.getvalue()
