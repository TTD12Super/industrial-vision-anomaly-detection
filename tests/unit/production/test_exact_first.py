"""Focused tests for Exact-first production orchestration."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch

from mvtec_system.anomaly.detectors import AnomalyResult, PaDiMDetector, PatchCoreDetector
from mvtec_system.anomaly.multi_layer import MultiLayerConfig
from mvtec_system.core.dinov2 import DINOv2Config, DINOv2Features
from mvtec_system.production.pipeline import (
    _anomaly_report,
    _calibrate_anomaly_thresholds,
    _calibrate_detector_pixel_thresholds,
    _calibrate_detector_thresholds,
    _classification_report,
    _ERROR_ANALYSIS_FIELDS,
    _error_analysis_rows,
    _infer_with_detector,
    _localization_metric_rows,
    _markdown_report,
    _load_or_fit_detector,
    _make_refined_anomaly_map,
    _validate_category_train_paths,
    infer_exact_first,
    ProductionConfig,
    run_production,
)


def _exact(decision: str) -> dict:
    return {
        "category": "bottle",
        "status": "known_product_category",
        "rejected": False,
        "decision": decision,
        "is_defect": decision == "DEFECT",
        "anomaly_score": 0.8 if decision == "DEFECT" else 0.1,
        "threshold": 0.5,
    }


class ExactFirstTests(TestCase):
    def test_threshold_calibration_requires_validation_rows(self) -> None:
        encoder = Mock()
        config = ProductionConfig()
        with tempfile.TemporaryDirectory() as directory:
            for rows, expected in (
                (
                    [{"category": "bottle", "split": "train", "image_path": "train.png"}],
                    "bottle",
                ),
                (
                    [
                        {"category": "bottle", "split": "validation", "image_path": "val.png"},
                        {"category": "cable", "split": "train", "image_path": "train.png"},
                    ],
                    "cable",
                ),
            ):
                for calibrate, categories in (
                    (_calibrate_anomaly_thresholds, ("bottle", "cable")),
                    (_calibrate_detector_thresholds, ("bottle", "cable")),
                    (_calibrate_detector_pixel_thresholds, ("bottle", "cable")),
                ):
                    model = {category: Mock() for category in categories}
                    with self.subTest(calibrator=calibrate.__name__, missing=expected), patch(
                        "mvtec_system.production.pipeline._score_with_bank"
                    ) as score_with_bank, patch(
                        "mvtec_system.production.pipeline._score_detector_paths"
                    ) as score_detector:
                        with self.assertRaisesRegex(
                            ValueError, f"No validation images for threshold calibration: .*{expected}"
                        ):
                            calibrate(rows, Path(directory), model, encoder, config)
                        score_with_bank.assert_not_called()
                        score_detector.assert_not_called()

    def test_categories_without_train_images_fail_instead_of_using_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "No training images for categories: \\['bottle'\\]"):
            _validate_category_train_paths({"bottle": [], "cable": [Path("train.png")]})

    def test_stale_detector_artifact_is_refit_for_requested_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory)
            artifact_file = artifact_path / "coreset.pt"
            artifact_file.write_bytes(b"unchanged")
            identity = {"cache_reusable": True, "category": "bottle"}
            saved_identity = {
                **identity,
                "artifact_file_sha256": hashlib.sha256(b"unchanged").hexdigest(),
                "artifact_file_size": len(b"unchanged"),
            }
            (artifact_path / "artifact_metadata.json").write_text(json.dumps(saved_identity))
            stored_config = MultiLayerConfig(layers=(5,))
            requested = MultiLayerConfig(layers=(5, 8, 11))
            fit_configs = []
            detector = Mock()
            detector.config = requested
            detector.load.side_effect = lambda _: setattr(detector, "config", stored_config)
            detector.fit.side_effect = lambda *_: fit_configs.append(detector.config)
            detector.patch_grid = (2, 2)
            detector.padded_size = (28, 28)
            detector.image_size = (28, 28)
            encoder = Mock()
            encoder.patch_grid = (2, 2)
            encoder.padded_size = (28, 28)
            encoder.config.image_size = (28, 28)

            _load_or_fit_detector(
                detector,
                "bottle",
                [Path("normal.png")],
                encoder,
                4,
                artifact_path,
                "coreset.pt",
                requested,
                identity,
                False,
            )

        detector.load.assert_called_once_with(artifact_path)
        detector.fit.assert_called_once_with("bottle", [Path("normal.png")], encoder, 4)
        detector.save.assert_called_once_with(artifact_path)
        self.assertEqual(fit_configs, [requested])
        self.assertEqual(detector.config, requested)

    def test_changed_detector_payload_is_refit_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory)
            artifact_file = artifact_path / "coreset.pt"
            artifact_file.write_bytes(b"changed")
            identity = {"cache_reusable": True, "category": "bottle"}
            saved_identity = {
                **identity,
                "artifact_file_sha256": hashlib.sha256(b"original").hexdigest(),
                "artifact_file_size": len(b"original"),
            }
            (artifact_path / "artifact_metadata.json").write_text(json.dumps(saved_identity))
            detector = Mock()
            detector.config = MultiLayerConfig()
            encoder = Mock()

            _load_or_fit_detector(
                detector,
                "bottle",
                [Path("normal.png")],
                encoder,
                4,
                artifact_path,
                "coreset.pt",
                MultiLayerConfig(),
                identity,
                False,
            )

        detector.load.assert_not_called()
        detector.fit.assert_called_once()
        detector.save.assert_called_once_with(artifact_path)

    def test_real_patchcore_artifact_round_trips_through_cache_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory)
            config = MultiLayerConfig()
            saved = PatchCoreDetector(config)
            saved.coreset = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
            saved.category = "bottle"
            saved.patch_grid = (1, 2)
            saved.padded_size = (14, 28)
            saved.image_size = (14, 28)
            saved.save(artifact_path)
            artifact_file = artifact_path / "coreset.pt"
            identity = {"cache_reusable": True, "category": "bottle"}
            identity.update(
                artifact_file_sha256=hashlib.sha256(artifact_file.read_bytes()).hexdigest(),
                artifact_file_size=artifact_file.stat().st_size,
            )
            (artifact_path / "artifact_metadata.json").write_text(json.dumps(identity))
            encoder = Mock()
            encoder.patch_grid = (1, 2)
            encoder.padded_size = (14, 28)
            encoder.config.image_size = (14, 28)
            loaded = PatchCoreDetector(config)

            result = _load_or_fit_detector(
                loaded,
                "bottle",
                [],
                encoder,
                4,
                artifact_path,
                "coreset.pt",
                config,
                {key: value for key, value in identity.items() if not key.startswith("artifact_file_")},
                False,
            )

        self.assertIs(result, loaded)
        self.assertEqual(loaded.category, "bottle")
        torch.testing.assert_close(loaded.coreset, saved.coreset)

    def test_missing_saved_detector_config_is_refit_for_both_detectors(self) -> None:
        config = MultiLayerConfig()
        for detector_type, artifact_name in (
            (PatchCoreDetector, "coreset.pt"),
            (PaDiMDetector, "statistics.pt"),
        ):
            with self.subTest(detector=detector_type.__name__), tempfile.TemporaryDirectory() as directory:
                artifact_path = Path(directory)
                artifact_file = artifact_path / artifact_name
                torch.save({}, artifact_file)
                identity = {"cache_reusable": True, "category": "bottle"}
                saved_identity = {
                    **identity,
                    "artifact_file_sha256": hashlib.sha256(artifact_file.read_bytes()).hexdigest(),
                    "artifact_file_size": artifact_file.stat().st_size,
                }
                (artifact_path / "artifact_metadata.json").write_text(json.dumps(saved_identity))
                detector = detector_type(config)
                detector.fit = Mock()
                detector.save = Mock()
                encoder = Mock()

                _load_or_fit_detector(
                    detector,
                    "bottle",
                    [Path("normal.png")],
                    encoder,
                    4,
                    artifact_path,
                    artifact_name,
                    config,
                    identity,
                    False,
                )

                detector.fit.assert_called_once()
                detector.save.assert_called_once_with(artifact_path)

    def test_heatmap_open5_refines_map_without_changing_patch_scores(self) -> None:
        patch_scores = np.array([0.1, 0.9, 0.2, 0.3], dtype=np.float32)
        source_map = np.zeros((15, 15), dtype=np.float32)
        source_map[4:11, 4:11] = 0.5
        source_map[1, 1] = 1.0
        encoder = Mock()
        encoder.patch_grid = (2, 2)
        encoder.padded_size = (15, 15)
        encoder.config.image_size = (15, 15)
        config = ProductionConfig(heatmap_refiner="heatmap_open5")

        with patch(
            "mvtec_system.production.pipeline.make_anomaly_maps",
            return_value=source_map[None, ...],
        ):
            scores, anomaly_map = _make_refined_anomaly_map(
                patch_scores, torch.zeros((4, 2)), encoder, config
            )

        np.testing.assert_array_equal(scores, patch_scores)
        self.assertEqual(float(anomaly_map[1, 1]), 0.0)
        self.assertGreater(float(anomaly_map[7, 7]), 0.0)

    def test_independent_exact_knn_rejects_unsupported_heatmap_refiner(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported for independent exact_knn"):
            run_production(
                ProductionConfig(
                    detector="exact_knn",
                    orchestration="independent",
                    heatmap_refiner="heatmap_open5",
                )
            )

    def test_good_short_circuits_localization(self) -> None:
        patchcore = Mock()
        padim = Mock()

        result = infer_exact_first(lambda: _exact("GOOD"), patchcore, padim)

        self.assertEqual(result["decision"], "GOOD")
        self.assertEqual(result["decision_source"], "exact_knn")
        self.assertEqual(result["localization_status"], {"patchcore": "not_run", "padim": "not_run"})
        patchcore.assert_not_called()
        padim.assert_not_called()

    def test_defect_runs_both_localization_branches(self) -> None:
        result = infer_exact_first(
            lambda: _exact("DEFECT"),
            lambda: np.ones((4, 4), dtype=np.float32),
            lambda: np.zeros((4, 4), dtype=bool),
        )

        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["localization_status"]["patchcore"], "available")
        self.assertEqual(result["localization_status"]["padim"], "empty")
        self.assertIsNotNone(result["heatmap"])
        self.assertIsNotNone(result["binary_mask"])

    def test_patchcore_failure_does_not_block_padim(self) -> None:
        padim = Mock(return_value=np.ones((4, 4), dtype=bool))

        def fail_patchcore() -> np.ndarray:
            raise RuntimeError("patchcore unavailable")

        result = infer_exact_first(lambda: _exact("DEFECT"), fail_patchcore, padim)

        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["localization_status"]["patchcore"], "failed")
        self.assertEqual(result["localization_status"]["padim"], "available")
        padim.assert_called_once()

    def test_padim_failure_does_not_remove_patchcore_heatmap(self) -> None:
        patchcore = Mock(return_value=np.ones((4, 4), dtype=np.float32))

        def fail_padim() -> np.ndarray:
            raise RuntimeError("padim unavailable")

        result = infer_exact_first(lambda: _exact("DEFECT"), patchcore, fail_padim)

        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["localization_status"]["patchcore"], "available")
        self.assertEqual(result["localization_status"]["padim"], "failed")
        self.assertIsNotNone(result["heatmap"])
        self.assertIsNone(result["binary_mask"])

    def test_exact_failure_is_unavailable_and_never_good(self) -> None:
        patchcore = Mock()
        padim = Mock()

        def fail_exact() -> dict:
            raise RuntimeError("exact bank missing")

        result = infer_exact_first(fail_exact, patchcore, padim)

        self.assertIsNone(result["decision"])
        self.assertEqual(result["decision_status"], "unavailable")
        self.assertEqual(result["localization_status"], {"patchcore": "not_run", "padim": "not_run"})
        patchcore.assert_not_called()
        padim.assert_not_called()

    def test_unknown_category_does_not_fallback_to_a_bank(self) -> None:
        patchcore = Mock()
        padim = Mock()
        exact = {
            "category": None,
            "status": "unknown_product_category",
            "rejected": True,
            "reason": "confidence below threshold",
        }

        result = infer_exact_first(lambda: exact, patchcore, padim)

        self.assertIsNone(result["decision"])
        self.assertEqual(result["status"], "unknown_product_category")
        self.assertEqual(result["decision_status"], "unavailable")
        self.assertIsNone(result["anomaly_decision"])
        self.assertIsNone(result["anomaly_decision_is_defect"])
        self.assertIsNone(result["anomaly_decision_source"])
        self.assertEqual(result["anomaly_decision_status"], "unavailable")
        patchcore.assert_not_called()
        padim.assert_not_called()

    def test_exact_first_anomaly_schema_is_consistent_across_outcomes(self) -> None:
        good = infer_exact_first(lambda: _exact("GOOD"), Mock(), Mock())
        defect = infer_exact_first(
            lambda: _exact("DEFECT"),
            lambda: np.ones((2, 2), dtype=np.float32),
            lambda: np.zeros((2, 2), dtype=bool),
        )

        def fail_exact() -> dict:
            raise RuntimeError("exact bank unavailable")

        unavailable = infer_exact_first(fail_exact, Mock(), Mock())

        self.assertEqual(
            (good["anomaly_decision"], good["anomaly_decision_is_defect"]),
            ("GOOD", False),
        )
        self.assertEqual(good["anomaly_decision_source"], "exact_knn")
        self.assertEqual(good["anomaly_decision_status"], "available")
        self.assertEqual(good["decision"], "GOOD")

        self.assertEqual(
            (defect["anomaly_decision"], defect["anomaly_decision_is_defect"]),
            ("DEFECT", True),
        )
        self.assertEqual(defect["anomaly_decision_source"], "exact_knn")
        self.assertEqual(defect["anomaly_decision_status"], "available")
        self.assertEqual(defect["decision"], "DEFECT")
        self.assertEqual(defect["decision_source"], "exact_knn")
        self.assertEqual(defect["decision_status"], "available")

        self.assertIsNone(unavailable["anomaly_decision"])
        self.assertIsNone(unavailable["anomaly_decision_is_defect"])
        self.assertIsNone(unavailable["anomaly_decision_source"])
        self.assertEqual(unavailable["anomaly_decision_status"], "unavailable")
        self.assertIsNone(unavailable["decision"])
        self.assertEqual(unavailable["decision_status"], "unavailable")

    def test_missing_exact_artifact_is_explicitly_unavailable(self) -> None:
        result = infer_exact_first(
            lambda: {
                "category": "bottle",
                "status": "known_product_category",
                "rejected": False,
                "bank_status": "missing",
                "reason": "feature bank is missing",
            },
            Mock(),
            Mock(),
        )

        self.assertIsNone(result["decision"])
        self.assertEqual(result["decision_status"], "unavailable")
        self.assertEqual(result["primary_error"], "feature bank is missing")

    def test_metrics_use_prediction_not_ground_truth_as_prediction(self) -> None:
        report = _anomaly_report(
            [
                {
                    "anomaly_evaluated": True,
                    "anomaly_score": 0.9,
                    "ground_truth_is_defect": False,
                    "decision_is_defect": True,
                    "heatmap": np.ones((2, 2), dtype=np.float32),
                    "binary_mask": np.zeros((2, 2), dtype=bool),
                    "mask": np.zeros((2, 2), dtype=bool),
                }
            ]
        )

        self.assertEqual(report["image"]["thresholded"]["accuracy"], 0.0)
        self.assertEqual(report["image"]["thresholded"]["f1"], 0.0)

    def test_detector_pixel_metrics_do_not_mix_heatmap_and_mask_sources(self) -> None:
        truth = np.zeros((2, 2), dtype=bool)
        truth[0, 0] = True
        report = _anomaly_report(
            [
                {
                    "anomaly_evaluated": True,
                    "anomaly_score": 0.9,
                    "ground_truth_is_defect": True,
                    "decision_is_defect": True,
                    "patchcore_heatmap": np.ones((2, 2), dtype=np.float32),
                    "padim_heatmap": np.ones((2, 2), dtype=np.float32),
                    "padim_binary_mask": truth.copy(),
                    "mask": truth,
                }
            ]
        )

        self.assertNotIn("thresholded", report["detectors"]["patchcore"])
        self.assertIn("thresholded", report["detectors"]["padim"])

    def test_localization_metrics_cover_each_true_category_and_keep_sources_separate(self) -> None:
        empty = np.zeros((2, 2), dtype=bool)
        defect = empty.copy()
        defect[0, 0] = True
        report = _anomaly_report(
            [
                {
                    "true_category": "bottle",
                    "ground_truth_is_defect": False,
                    "rejected": False,
                    "patchcore_heatmap": empty.astype(np.float32),
                    "padim_heatmap": empty.astype(np.float32),
                    "padim_binary_mask": empty,
                    "mask": empty,
                },
                {
                    "true_category": "bottle",
                    "ground_truth_is_defect": True,
                    "rejected": True,
                    "patchcore_heatmap": defect.astype(np.float32),
                    "padim_heatmap": defect.astype(np.float32),
                    "padim_binary_mask": defect,
                    "mask": defect,
                },
                {
                    "true_category": "cable",
                    "ground_truth_is_defect": False,
                    "rejected": False,
                    "mask": empty,
                },
            ]
        )

        per_category = {row["true_category"]: row for row in report["per_category"]}
        self.assertEqual(set(per_category), {"bottle", "cable"})
        bottle = per_category["bottle"]
        self.assertEqual(bottle["image_count"], 2)
        self.assertEqual(bottle["sources"]["patchcore"]["map_images"], 2)
        self.assertEqual(bottle["sources"]["patchcore"]["mask_images"], 0)
        self.assertEqual(bottle["sources"]["padim"]["mask_images"], 2)
        self.assertEqual(bottle["sources"]["padim"]["rejected_images"], 1)
        self.assertEqual(bottle["sources"]["padim"]["rejected_map_images"], 1)
        self.assertEqual(bottle["sources"]["padim"]["rejected_mask_images"], 1)
        self.assertEqual(bottle["sources"]["padim"]["rejected_map_coverage"], 1.0)
        self.assertEqual(bottle["sources"]["padim"]["rejected_mask_coverage"], 1.0)
        defect_subgroup = bottle["sources"]["padim"]["subgroups"]["defect"]
        self.assertEqual(defect_subgroup["rejected_images"], 1)
        self.assertEqual(defect_subgroup["rejected_map_images"], 1)
        self.assertEqual(defect_subgroup["rejected_mask_images"], 1)
        self.assertEqual(defect_subgroup["rejected_map_coverage"], 1.0)
        self.assertEqual(defect_subgroup["rejected_mask_coverage"], 1.0)
        self.assertEqual(defect_subgroup["thresholded"]["evaluated_pixels"], 4)
        self.assertNotIn("map_sample_count", bottle["sources"]["padim"])
        self.assertNotIn("mask_sample_count", bottle["sources"]["padim"])
        self.assertEqual(
            bottle["sources"]["padim"]["subgroups"]["good"][
                "false_positive_pixel_rate"
            ],
            0.0,
        )
        self.assertEqual(
            bottle["sources"]["padim"]["subgroups"]["good"][
                "predicted_mask_area"
            ]["mean_pixels"],
            0.0,
        )
        self.assertIsNotNone(
            bottle["sources"]["padim"]["subgroups"]["defect"]["ranking"]["auroc"]
        )
        self.assertIn("thresholded", bottle["sources"]["padim"]["subgroups"]["defect"])
        self.assertEqual(per_category["cable"]["sources"]["patchcore"]["map_images"], 0)
        self.assertEqual(per_category["cable"]["sources"]["patchcore"]["map_coverage"], 0.0)
        self.assertIsNone(
            per_category["cable"]["sources"]["patchcore"]["rejected_map_coverage"]
        )
        self.assertIsNone(
            per_category["cable"]["sources"]["patchcore"]["rejected_mask_coverage"]
        )

        csv_rows = _localization_metric_rows(report)
        self.assertEqual(len(csv_rows), 2)
        self.assertIn("patchcore_pixel_average_precision", csv_rows[0])
        self.assertIn("padim_specificity", csv_rows[0])
        self.assertIn("padim_rejected_map_images", csv_rows[0])
        self.assertIn("padim_good_false_positive_pixel_rate", csv_rows[0])
        self.assertIn("padim_defect_pixel_auroc", csv_rows[0])
        self.assertIn("padim_dice", csv_rows[0])
        self.assertEqual(csv_rows[0]["padim_good_image_count"], 1)
        self.assertEqual(csv_rows[0]["padim_good_map_images"], 1)
        self.assertEqual(csv_rows[0]["padim_good_mask_images"], 1)
        self.assertEqual(csv_rows[0]["padim_good_map_coverage"], 1.0)
        self.assertEqual(csv_rows[0]["padim_good_mask_coverage"], 1.0)
        self.assertEqual(csv_rows[0]["padim_defect_image_count"], 1)
        self.assertEqual(csv_rows[0]["padim_defect_map_images"], 1)
        self.assertEqual(csv_rows[0]["padim_defect_mask_images"], 1)
        self.assertEqual(csv_rows[0]["padim_defect_map_coverage"], 1.0)
        self.assertEqual(csv_rows[0]["padim_defect_mask_coverage"], 1.0)
        self.assertEqual(csv_rows[0]["padim_defect_rejected_images"], 1)
        self.assertEqual(csv_rows[0]["padim_defect_rejected_map_images"], 1)
        self.assertEqual(csv_rows[0]["padim_defect_rejected_mask_images"], 1)
        self.assertEqual(csv_rows[0]["padim_defect_rejected_map_coverage"], 1.0)
        self.assertEqual(csv_rows[0]["padim_defect_rejected_mask_coverage"], 1.0)
        self.assertEqual(csv_rows[0]["padim_defect_evaluated_pixels"], 4)
        self.assertIsNone(csv_rows[1]["patchcore_defect_rejected_map_coverage"])
        self.assertIsNone(csv_rows[1]["patchcore_defect_rejected_mask_coverage"])
        self.assertNotIn("patchcore_map_sample_count", csv_rows[0])
        self.assertNotIn("padim_mask_sample_count", csv_rows[0])

    def test_error_analysis_serializes_rejected_route_audit_fields(self) -> None:
        rows = _error_analysis_rows(
            [
                {
                    "image_path": "rejected.png",
                    "true_category": "bottle",
                    "predicted_category": "bottle",
                    "category": None,
                    "localization_category": "bottle",
                    "confidence": 0.2,
                    "status": "unknown_product_category",
                    "defect_type": "crack",
                    "ground_truth_is_defect": True,
                    "rejected": True,
                    "reason": "confidence below threshold",
                    "decision": "DEFECT",
                    "anomaly_decision": "DEFECT",
                    "anomaly_decision_source": "patchcore",
                    "anomaly_decision_status": "available",
                }
            ]
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsNone(row["routed_category"])
        self.assertEqual(row["localization_category"], "bottle")
        self.assertEqual(row["anomaly_decision"], "DEFECT")
        self.assertEqual(row["anomaly_decision_source"], "patchcore")
        self.assertEqual(row["anomaly_decision_status"], "available")
        self.assertEqual(row["decision"], "DEFECT")
        self.assertTrue(set(_ERROR_ANALYSIS_FIELDS).issubset(row))

    def test_markdown_reports_subgroup_coverage_and_good_mask_condition(self) -> None:
        summary = {
            "dataset_name": "tiny",
            "dataset_root": "datasets/tiny",
            "processed_root": "datasets/tiny_processed",
            "categories": ["bottle"],
            "classification_enabled": True,
            "configuration": {
                "orchestration": "independent",
                "detector": "padim",
                "decision_authority": "padim",
                "classification_confidence_threshold": 0.5,
                "k": 1,
            },
            "classification": {
                "total_test_images": 2,
                "top1_accuracy_all": 1.0,
                "top1_accuracy_accepted": 1.0,
                "unknown_rejected_predictions": 0,
                "rejection_rate": 0.0,
                "macro_precision": 1.0,
                "macro_recall": 1.0,
                "macro_f1": 1.0,
                "good_test": {"accuracy": 1.0},
                "defect_test": {"accuracy": 1.0},
            },
            "anomaly": {
                "anomaly_evaluated_images": 2,
                "anomaly_skipped_images": 0,
                "image": {"ranking": {"auroc": 1.0, "average_precision": 1.0}},
                "pixel": {"ranking": {"auroc": 1.0, "average_precision": 1.0}},
                "detectors": {
                    "padim": {
                        "rejected_images": 0,
                        "rejected_map_images": 0,
                        "rejected_mask_images": 0,
                        "rejected_map_coverage": 0.0,
                        "rejected_mask_coverage": 0.0,
                        "subgroups": {
                            "good": {
                                "image_count": 1,
                                "map_images": 1,
                                "mask_images": 0,
                                "map_coverage": 1.0,
                                "mask_coverage": 0.0,
                                "false_positive_pixel_rate": None,
                                "predicted_mask_area": {
                                    "mean_pixels": None,
                                    "mean_ratio": None,
                                },
                            },
                            "defect": {
                                "image_count": 1,
                                "map_images": 1,
                                "mask_images": 1,
                                "map_coverage": 1.0,
                                "mask_coverage": 1.0,
                                "ranking": {"auroc": 1.0, "average_precision": 1.0},
                                "thresholded": {
                                    "precision": 1.0,
                                    "recall": 1.0,
                                    "f1": 1.0,
                                    "specificity": 1.0,
                                    "dice": 1.0,
                                    "iou": 1.0,
                                },
                            },
                        },
                    }
                },
            },
            "performance": {
                "inference": {
                    "device": "cpu",
                    "mean_seconds": 0.1,
                    "median_seconds": 0.1,
                    "p95_seconds": 0.2,
                    "images_per_second": 10.0,
                }
            },
            "output_files": {"summary": "final_summary.json"},
        }

        report = _markdown_report(summary, Path("reports"))

        self.assertIn("mask coverage **0/1 (0.00%)**", report)
        self.assertIn("conditional on available GOOD masks", report)
        self.assertIn("interpret with GOOD mask coverage", report)
        self.assertIn("DEFECT: map coverage **1/1 (100.00%)**", report)
        self.assertIn("mask coverage **1/1 (100.00%)**", report)

    def test_markdown_report_describes_actual_orchestration(self) -> None:
        def summary(orchestration: str, detector: str) -> dict:
            return {
                "dataset_name": "tiny",
                "dataset_root": "datasets/tiny",
                "processed_root": "datasets/tiny_processed",
                "categories": ["bottle"],
                "classification_enabled": True,
                "configuration": {
                    "orchestration": orchestration,
                    "detector": detector,
                    "decision_authority": "exact_knn" if orchestration == "exact_first" else detector,
                    "classification_confidence_threshold": 0.5,
                    "k": 1,
                },
                "classification": {
                    "total_test_images": 2,
                    "top1_accuracy_all": 1.0,
                    "top1_accuracy_accepted": 1.0,
                    "unknown_rejected_predictions": 0,
                    "rejection_rate": 0.0,
                    "macro_precision": 1.0,
                    "macro_recall": 1.0,
                    "macro_f1": 1.0,
                    "good_test": {"accuracy": 1.0},
                    "defect_test": {"accuracy": 1.0},
                },
                "anomaly": {
                    "anomaly_evaluated_images": 2,
                    "anomaly_skipped_images": 0,
                    "image": {"ranking": {"auroc": 1.0, "average_precision": 1.0}},
                    "pixel": {"ranking": {"auroc": 1.0, "average_precision": 1.0}},
                    "detectors": {},
                },
                "performance": {
                    "inference": {
                        "device": "cpu",
                        "mean_seconds": 0.1,
                        "median_seconds": 0.1,
                        "p95_seconds": 0.2,
                        "images_per_second": 10.0,
                    }
                },
                "output_files": {"summary": "final_summary.json"},
            }

        cases = (
            (
                "exact_first",
                "exact_knn",
                "Exact kNN image decision → DEFECT: PatchCore heatmap + PaDiM binary mask",
            ),
            (
                "independent",
                "patchcore",
                "independent PatchCore heatmap and mask evaluation",
            ),
            (
                "independent",
                "padim",
                "independent PaDiM anomaly map and binary mask evaluation",
            ),
            (
                "independent",
                "exact_knn",
                "independent Exact kNN anomaly decision",
            ),
        )
        for orchestration, detector, description in cases:
            report = _markdown_report(summary(orchestration, detector), Path("reports"))
            self.assertIn(description, report)
            self.assertNotIn("classification → Exact kNN decision → optional localization", report)

    def test_unavailable_decision_does_not_break_category_report(self) -> None:
        report, _ = _classification_report(
            [
                {
                    "true_category": "bottle",
                    "category": None,
                    "rejected": False,
                    "ground_truth_is_defect": True,
                }
            ],
            ["bottle"],
        )

        self.assertEqual(report["accepted_known_predictions"], 0)
        self.assertEqual(report["top1_accuracy_all"], 0.0)

    def test_independent_localization_uses_predicted_category_for_rejected_route(self) -> None:
        encoder = Mock()
        encoder.config = DINOv2Config(image_size=(2, 2))
        encoder.patch_grid = (1, 2)
        encoder.padded_size = (2, 2)
        encoder.metadata.return_value = {}
        features = DINOv2Features(
            global_embedding=torch.ones((1, 2)),
            patch_embeddings=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        )
        encoder.forward.return_value = features

        detector = Mock()
        detector.predict.return_value = AnomalyResult(
            image_score=0.8,
            anomaly_map=np.ones((2, 2), dtype=np.float32),
            patch_scores=np.array([0.8, 0.2], dtype=np.float32),
            metadata={},
        )
        route = Mock()
        route.rejected = True
        route.category = None
        route.predicted_category = "bottle"
        route.to_dict.return_value = {
            "category": None,
            "predicted_category": "bottle",
            "status": "unknown_product_category",
            "rejected": True,
        }
        image_thresholds = {
            "threshold_type": "image_level",
            "categories": {"bottle": {"threshold": 0.5}},
        }
        pixel_thresholds = {
            "threshold_type": "pixel_level",
            "categories": {"bottle": {"threshold": 0.5}},
        }

        with patch(
            "mvtec_system.production.pipeline.image_tensor",
            return_value=torch.zeros((3, 2, 2)),
        ), patch(
            "mvtec_system.production.pipeline.route_product", return_value=route
        ):
            result = _infer_with_detector(
                {"image_path": "rejected.png"},
                encoder,
                encoder,
                None,
                None,
                {"bottle": 0},
                {},
                {"bottle": detector},
                image_thresholds,
                pixel_thresholds,
                ProductionConfig(image_size=2, detector="patchcore"),
            )

        self.assertIsNone(result["category"])
        self.assertTrue(result["rejected"])
        self.assertEqual(result["status"], "unknown_product_category")
        self.assertEqual(result["localization_category"], "bottle")
        self.assertEqual(result["anomaly_decision"], "DEFECT")
        self.assertTrue(result["anomaly_decision_is_defect"])
        self.assertEqual(result["anomaly_decision_source"], "patchcore")
        self.assertEqual(result["anomaly_decision_status"], "available")
        self.assertEqual(result["decision"], "DEFECT")
        self.assertEqual(result["decision"], result["anomaly_decision"])
        self.assertEqual(result["decision_is_defect"], result["anomaly_decision_is_defect"])
        self.assertEqual(result["decision_source"], result["anomaly_decision_source"])
        self.assertEqual(result["decision_status"], result["anomaly_decision_status"])
        self.assertIsNotNone(result["anomaly_map"])
        detector.predict.assert_called_once_with(features)


if __name__ == "__main__":
    import unittest

    unittest.main()
