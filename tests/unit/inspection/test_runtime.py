import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inspection.backends import BackendUnavailable
from inspection.config import load_product_config
from inspection.runtime import build_pipeline, product_readiness


class RuntimeReadinessTests(unittest.TestCase):
    def _config(self, root: Path, with_recognizer: bool = False) -> Path:
        config_path = root / "configs" / "products" / "widget.yaml"
        config_path.parent.mkdir(parents=True)
        (root / "artifacts").mkdir()
        (root / "artifacts/detector.pt").write_bytes(b"detector")
        (root / "artifacts/detector.json").write_text(json.dumps({"confidence": 0.2}), encoding="utf-8")
        classifier_dir = root / "artifacts/classifier/classifier"
        classifier_dir.mkdir(parents=True)
        (classifier_dir / "classifier.pt").write_bytes(b"classifier")
        (classifier_dir / "class_mapping.json").write_text(
            json.dumps({"category_to_index": {"GOOD": 0, "DEFECT": 1}}), encoding="utf-8"
        )
        (root / "artifacts/encoder").mkdir()
        (root / "scripts").mkdir()
        (root / "scripts/run.py").write_text("# test runner\n", encoding="utf-8")
        (root / "artifacts/repository").mkdir()
        (root / "artifacts/checkpoint.pt").write_bytes(b"checkpoint")
        (root / "artifacts/clip").mkdir()
        (root / "artifacts/localizer.json").write_text(json.dumps({"mask_threshold": 0.1}), encoding="utf-8")
        (root / "artifacts/policy.json").write_text(
            json.dumps({"defect_threshold": 0.1, "review_low_threshold": 0.05}), encoding="utf-8"
        )
        content = """schema_version: 1
product_key: widget
dataset:
  root: datasets/products/widget
  image_manifest: manifests/images.jsonl
  object_manifest: manifests/objects.jsonl
  split_policy: grouped
models:
  detector:
    backend: yolo_obb
    weights: artifacts/detector.pt
    calibration_path: artifacts/detector.json
  classifier:
    backend: dinov2_logreg
    artifact_dir: artifacts/classifier
    encoder_artifact_root: artifacts/encoder
  localizer:
    backend: anomalyclip
    runner: scripts/run.py
    repository: artifacts/repository
    checkpoint: artifacts/checkpoint.pt
    clip_cache: artifacts/clip
    calibration_path: artifacts/localizer.json
    supervised_refiner_enabled: false
    relation_fusion_enabled: false
  reporter:
    backend: mock
policy:
  quality_labels: [GOOD, DEFECT]
  calibration_path: artifacts/policy.json
outputs: {json: true}
"""
        if with_recognizer:
            content = content.replace(
                "  reporter:\n",
                "  product_recognizer:\n    backend: gallery\n    gallery_path: artifacts/product_gallery.npz\n    calibration_path: artifacts/product_recognizer.json\n    encoder_artifact_root: artifacts/product_encoder\n  reporter:\n",
            )
        config_path.write_text(content, encoding="utf-8")
        return config_path

    def test_complete_artifacts_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_product_config(self._config(Path(directory)))

            readiness = product_readiness(config)

            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["missing_components"], [])

    def test_missing_detector_calibration_is_not_ready_and_blocks_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            (root / "artifacts/detector.json").unlink()
            config = load_product_config(config_path)

            readiness = product_readiness(config)

            self.assertFalse(readiness["ready"])
            self.assertFalse(readiness["components"]["detector_calibration"])
            with self.assertRaisesRegex(BackendUnavailable, "detector_calibration"):
                build_pipeline(config)

    def test_missing_policy_calibration_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            (root / "artifacts/policy.json").unlink()
            config = load_product_config(config_path)

            readiness = product_readiness(config)

            self.assertFalse(readiness["ready"])
            self.assertFalse(readiness["components"]["classifier_calibration"])
            with self.assertRaisesRegex(BackendUnavailable, "Calibrated policy is missing"):
                build_pipeline(config)

    def test_missing_product_recognizer_artifact_and_calibration_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_product_config(self._config(Path(directory), with_recognizer=True))

            readiness = product_readiness(config)

            self.assertEqual(readiness["routing_mode"], "product_recognizer")
            self.assertFalse(readiness["ready"])
            self.assertFalse(readiness["components"]["product_recognizer_artifact"])
            self.assertFalse(readiness["components"]["product_recognizer_calibration"])

    def test_missing_product_recognizer_encoder_is_not_ready_and_blocks_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_product_config(self._config(root, with_recognizer=True))

            with patch("inspection.runtime.gallery_artifact_ready", return_value=True), patch(
                "inspection.runtime.calibration_artifact_ready", return_value=True
            ):
                readiness = product_readiness(config)
                self.assertFalse(readiness["ready"])
                self.assertFalse(readiness["components"]["product_recognizer_encoder"])
                self.assertIn("product_recognizer_encoder", readiness["missing_components"])
                with self.assertRaisesRegex(BackendUnavailable, "product_recognizer_encoder"):
                    build_pipeline(config)

    def test_product_recognizer_encoder_file_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_product_config(self._config(root, with_recognizer=True))
            (root / "artifacts/product_encoder").write_bytes(b"not-a-directory")

            with patch("inspection.runtime.gallery_artifact_ready", return_value=True), patch(
                "inspection.runtime.calibration_artifact_ready", return_value=True
            ):
                readiness = product_readiness(config)

            self.assertFalse(readiness["components"]["product_recognizer_encoder"])

    def test_reporter_async_setting_is_forwarded_to_runtime_manager(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            content = config_path.read_text(encoding="utf-8").replace(
                "  reporter:\n", "  reporter:\n    async_enabled: false\n"
            )
            config_path.write_text(content, encoding="utf-8")
            config = load_product_config(config_path)

            with patch("inspection.runtime.build_detector"), patch("inspection.runtime.build_classifier"), patch(
                "inspection.runtime.build_localizer"
            ):
                pipeline = build_pipeline(config)

            self.assertFalse(pipeline.reporter.async_enabled)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
