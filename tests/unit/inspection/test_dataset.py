import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from inspection.config import load_product_config
from inspection.dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = root / "configs" / "products" / "widget.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(
            """schema_version: 1
product_key: widget_one
dataset:
  root: datasets/products/widget_one
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
        return config

    def test_valid_contract_passes_and_split_group_leak_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            dataset = root / "datasets/products/widget_one"
            image_path = dataset / "images" / "one.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20), "white").save(image_path)
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            images = [
                {"image_id": "one", "relative_path": "images/one.png", "split": "train", "source_dataset": "test", "capture_group_id": "a", "domain_id": "d", "sha256": sha, "width": 20, "height": 20},
                {"image_id": "two", "relative_path": "images/one.png", "split": "test", "source_dataset": "test", "capture_group_id": "a", "domain_id": "d", "sha256": sha, "width": 20, "height": 20},
            ]
            objects = [
                {"object_id": "one_obj", "image_id": "one", "product_key": "widget_one", "obb_xyxyxyxy": [[0, 0], [19, 0], [19, 19], [0, 19]], "quality_label": "GOOD"},
                {"object_id": "two_obj", "image_id": "two", "product_key": "widget_one", "obb_xyxyxyxy": [[0, 0], [19, 0], [19, 19], [0, 19]], "quality_label": "DEFECT"},
            ]
            manifests = dataset / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in images), encoding="utf-8")
            (manifests / "objects.jsonl").write_text("".join(json.dumps(row) + "\n" for row in objects), encoding="utf-8")

            report = validate_dataset(load_product_config(config_path))

            self.assertFalse(report.valid)
            self.assertIn("capture_group_id a appears in multiple splits: ['test', 'train']", report.errors)
