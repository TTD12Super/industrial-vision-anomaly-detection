from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from inspection.backends import PatchCorePaDiMLocalizer, StaticClassifier, StaticDetector, StaticLocalizer, StaticProductRecognizer
from inspection.interfaces import Detection, RecognitionResult
from inspection.pipeline import InspectionPipeline
from inspection.policy import DEFECT, GOOD, KNOWN_PRODUCT, NOT_ASSESSED, REVIEW_PRODUCT, REVIEW_REQUIRED, UNKNOWN_PRODUCT, DecisionPolicy
from inspection.reporting import AsyncReportManager, MockReporter


class PipelineTests(unittest.TestCase):
    def test_patchcore_padim_accepts_unbatched_and_batched_maps(self) -> None:
        unbatched = np.ones((4, 4), dtype=np.float32)
        batched = unbatched[None, ...]
        np.testing.assert_array_equal(PatchCorePaDiMLocalizer._single_map(unbatched), unbatched)
        np.testing.assert_array_equal(PatchCorePaDiMLocalizer._single_map(batched), unbatched)

    def test_mixed_objects_keep_binary_label_separate_from_operational_status(self) -> None:
        detections = [
            Detection(np.asarray([[5, 5], [35, 5], [35, 35], [5, 35]], dtype=np.float32), 0.99),
            Detection(np.asarray([[45, 5], [75, 5], [75, 35], [45, 35]], dtype=np.float32), 0.98),
            Detection(np.asarray([[5, 45], [35, 45], [35, 75], [5, 75]], dtype=np.float32), 0.97),
        ]
        localizer = StaticLocalizer(np.eye(4, dtype=np.float32), np.eye(4, dtype=bool))
        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(
                "screw",
                "test-bundle",
                StaticDetector(detections),
                StaticClassifier([0.01, 0.06, 0.90]),
                DecisionPolicy(0.10, 0.05),
                localizer,
                Path(directory) / "runs",
                AsyncReportManager(MockReporter()),
            )
            image = np.full((90, 90, 3), 180, dtype=np.uint8)
            output = pipeline.inspect_bytes(self._png(image), ".png", "inspection_01")

            self.assertEqual([item["operational_status"] for item in output["objects"]], [GOOD, REVIEW_REQUIRED, DEFECT])
            self.assertEqual(output["objects"][1]["classifier_label"], GOOD)
            self.assertEqual(output["routing_mode"], "closed_world")
            self.assertEqual(len(output["source_sha256"]), 64)
            self.assertIn("total", output["latency_ms"])
            self.assertEqual(output["objects"][0]["product_recognition_status"], "NOT_CONFIGURED")
            self.assertEqual(len(output["objects"][0]["crop_sha256"]), 64)
            self.assertIn("total", output["objects"][0]["latency_ms"])
            self.assertIn("heatmap_overlay", output["objects"][1])
            self.assertIn("heatmap_overlay", output["objects"][2])
            self.assertIn("vlm_report", output["objects"][2])
            root = Path(directory) / "runs" / "inspection_01"
            self.assertTrue((root / "overview/objects_labeled.png").is_file())
            self.assertTrue((root / "summary.pdf").is_file())
            self.assertTrue((root / "objects/screw_03/binary_mask.png").is_file())

    def test_known_product_routes_to_quality_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(
                "screw",
                "test-bundle",
                StaticDetector([self._detection()]),
                StaticClassifier(0.90),
                DecisionPolicy(0.10, 0.05),
                StaticLocalizer(np.ones((4, 4), dtype=np.float32)),
                Path(directory) / "runs",
                product_recognizer=StaticProductRecognizer(
                    RecognitionResult("screw", 0.99, "static-recognizer", KNOWN_PRODUCT, "known", 0.75)
                ),
            )

            output = pipeline.inspect_bytes(self._png(np.full((50, 50, 3), 180, dtype=np.uint8)), ".png", "known_01")
            result = output["objects"][0]

            self.assertEqual(output["routing_mode"], "product_recognizer")
            self.assertEqual(result["product_recognition_status"], KNOWN_PRODUCT)
            self.assertEqual(result["routed_product_key"], "screw")
            self.assertEqual(result["recognition_margin"], 0.75)
            self.assertEqual(result["classifier_label"], DEFECT)

    def test_review_product_has_no_product_key_and_exposes_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(
                "screw",
                "test-bundle",
                StaticDetector([self._detection()]),
                StaticClassifier(0.90),
                DecisionPolicy(0.10, 0.05),
                StaticLocalizer(np.ones((4, 4), dtype=np.float32)),
                Path(directory) / "runs",
                product_recognizer=StaticProductRecognizer(
                    RecognitionResult(None, 0.90, "static-recognizer", REVIEW_PRODUCT, "ambiguous_margin", 0.02)
                ),
            )

            result = pipeline.inspect_bytes(
                self._png(np.full((50, 50, 3), 180, dtype=np.uint8)), ".png", "review_01"
            )["objects"][0]

            self.assertEqual(result["product_recognition_status"], REVIEW_PRODUCT)
            self.assertIsNone(result["recognized_product_key"])
            self.assertIsNone(result["routed_product_key"])
            self.assertEqual(result["recognition_margin"], 0.02)
            self.assertEqual(result["operational_status"], REVIEW_REQUIRED)
            self.assertIsNone(result["classifier_label"])

    def test_unknown_product_is_saved_but_blocks_quality_and_localization(self) -> None:
        class FailingClassifier:
            def classify(self, crop: np.ndarray) -> object:
                raise AssertionError("quality classifier must not run for UNKNOWN_PRODUCT")

        class FailingLocalizer:
            def localize(self, crop: np.ndarray) -> object:
                raise AssertionError("localizer must not run for UNKNOWN_PRODUCT")

        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(
                "screw",
                "test-bundle",
                StaticDetector([self._detection()]),
                FailingClassifier(),
                DecisionPolicy(0.10, 0.05),
                FailingLocalizer(),
                Path(directory) / "runs",
                product_recognizer=StaticProductRecognizer(
                    RecognitionResult(None, 0.10, "static-recognizer", UNKNOWN_PRODUCT, "below_threshold")
                ),
            )

            output = pipeline.inspect_bytes(self._png(np.full((50, 50, 3), 180, dtype=np.uint8)), ".png", "unknown_01")
            result = output["objects"][0]
            root = Path(directory) / "runs" / "unknown_01"

            self.assertEqual(result["product_recognition_status"], UNKNOWN_PRODUCT)
            self.assertEqual(result["quality_status"], NOT_ASSESSED)
            self.assertEqual(result["operational_status"], NOT_ASSESSED)
            self.assertEqual(result["localization_status"], "not_requested")
            self.assertTrue((root / "objects/screw_01/crop_original.png").is_file())
            self.assertTrue((root / "objects/screw_01/result.json").is_file())

    @staticmethod
    def _detection() -> Detection:
        return Detection(np.asarray([[5, 5], [35, 5], [35, 35], [5, 35]], dtype=np.float32), 0.99)

    @staticmethod
    def _png(image: np.ndarray) -> bytes:
        from io import BytesIO

        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        return stream.getvalue()
