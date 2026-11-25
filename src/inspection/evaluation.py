"""Per-stage and leak-aware integrated evaluation from the canonical test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .backends import build_localizer
from .anomalyclip import AnomalyClipSession
from .config import ProductConfig
from .dataset import load_dataset_records
from .dino_refinement import DinoAffinityRefiner
from .geometry import oriented_crop
from .interfaces import KNOWN_PRODUCT, REVIEW_PRODUCT, UNKNOWN_PRODUCT, Localizer
from .mask_postprocess import postprocess_mask, refine_heatmap
from .pipeline import InspectionPipeline
from .policy import DEFECT, REVIEW_REQUIRED
from .relation_map import contextual_features, fuse_maps, relation_scores, scores_to_map
from .supervised_refinement import SupervisedPatchMaskRefiner, patch_inputs
from .supervised_segmentation import SupervisedRGBMaskRefiner
from .synthetic_segmentation import SyntheticDefectSegmenter


def polygon_iou(first: list[list[float]], second: list[list[float]]) -> float:
    """IoU of two convex OBB polygons in their source-image coordinates."""

    first_polygon = np.asarray(first, dtype=np.float32)
    second_polygon = np.asarray(second, dtype=np.float32)
    first_area = abs(cv2.contourArea(first_polygon))
    second_area = abs(cv2.contourArea(second_polygon))
    intersection, _ = cv2.intersectConvexConvex(first_polygon, second_polygon)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_objects(
    truths: list[dict[str, Any]], predictions: list[dict[str, Any]], iou_threshold: float
) -> dict[int, int]:
    """Return a deterministic one-to-one assignment maximizing valid IoU matches."""

    if not truths or not predictions:
        return {}
    overlaps = np.asarray(
        [
            [polygon_iou(truth["obb_xyxyxyxy"], prediction["obb_xyxyxyxy"]) for prediction in predictions]
            for truth in truths
        ],
        dtype=np.float64,
    )
    valid = overlaps >= iou_threshold
    # A large penalty makes cardinality dominate; IoU breaks ties among valid matches.
    cost = np.where(valid, -overlaps, 1_000_000.0)
    truth_indices, prediction_indices = linear_sum_assignment(cost)
    return {
        int(truth_index): int(prediction_index)
        for truth_index, prediction_index in zip(truth_indices, prediction_indices)
        if valid[truth_index, prediction_index]
    }


def evaluate_end_to_end(
    config: ProductConfig,
    pipeline: InspectionPipeline,
    report_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Count a GT DEFECT as captured only if it is both detected and classified DEFECT."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    product_recognizer_active = (
        getattr(pipeline, "product_recognizer", None) is not None
        or getattr(pipeline, "routing_mode", "closed_world") == "product_recognizer"
    )
    images, objects = load_dataset_records(config)
    objects_by_image: dict[str, list[dict[str, Any]]] = {}
    for object_row in objects:
        objects_by_image.setdefault(object_row["image_id"], []).append(object_row)
    totals = {
        "gt_defect": 0,
        "detected_defect": 0,
        "classified_defect": 0,
        "review_aware_capture": 0,
        "gt_good": 0,
        "good_flagged": 0,
        "missed_objects": 0,
        "ground_truth_objects": 0,
        "predicted_objects": 0,
        "matched_objects": 0,
        "missed_ground_truth_objects": 0,
        "unmatched_predictions": 0,
        "unmatched_false_positives": 0,
        "matched_good": 0,
        "missed_good": 0,
        "matched_defect": 0,
        "missed_defect": 0,
    }
    recognition_counts = {"known": 0, "unknown": 0, "review": 0}
    unknown_false_accept_count = 0
    unknown_ground_truth_count = 0
    unknown_rejected_count = 0
    known_ground_truth_count = 0
    known_defect_ground_truth_count = 0
    matched_known_ground_truth = 0
    known_recognized_matches = 0
    known_recognized_defect_count = 0
    known_recognized_classified_defect_count = 0
    per_image = []
    image_latencies: list[float] = []
    object_latencies: list[float] = []
    for index, image in enumerate(row for row in images if row["split"] == "test"):
        result = pipeline.inspect_path(config.dataset_path(Path(image["relative_path"])), f"evaluation_{index:04d}")
        if isinstance(result.get("latency_ms"), dict) and isinstance(result["latency_ms"].get("total"), (int, float)):
            image_latencies.append(float(result["latency_ms"]["total"]))
        object_latencies.extend(
            float(item["latency_ms"]["total"])
            for item in result.get("objects", [])
            if isinstance(item.get("latency_ms"), dict)
            and isinstance(item["latency_ms"].get("total"), (int, float))
        )
        predictions = result["objects"]
        truths = objects_by_image[image["image_id"]]
        truth_to_prediction = match_objects(truths, predictions, iou_threshold)
        matched = set(truth_to_prediction.values())
        for truth_index, truth in enumerate(truths):
            truth_is_known = truth.get("product_key", config.product_key) == config.product_key
            if product_recognizer_active and truth_is_known:
                known_ground_truth_count += 1
                if truth["quality_label"] == "DEFECT":
                    known_defect_ground_truth_count += 1
            if product_recognizer_active and not truth_is_known:
                unknown_ground_truth_count += 1
            position = truth_to_prediction.get(truth_index)
            prediction = predictions[position] if position is not None else None
            if prediction is not None:
                totals["matched_objects"] += 1
                if product_recognizer_active:
                    recognition_status = prediction.get("product_recognition_status")
                    if recognition_status == KNOWN_PRODUCT:
                        recognition_counts["known"] += 1
                    elif recognition_status == UNKNOWN_PRODUCT:
                        recognition_counts["unknown"] += 1
                    elif recognition_status == REVIEW_PRODUCT:
                        recognition_counts["review"] += 1
                    if truth_is_known:
                        matched_known_ground_truth += 1
                        if recognition_status == KNOWN_PRODUCT:
                            known_recognized_matches += 1
                            if truth["quality_label"] == "DEFECT":
                                known_recognized_defect_count += 1
                                if prediction.get("classifier_label") == DEFECT:
                                    known_recognized_classified_defect_count += 1
                    elif recognition_status == KNOWN_PRODUCT:
                        unknown_false_accept_count += 1
                    elif recognition_status in {UNKNOWN_PRODUCT, REVIEW_PRODUCT}:
                        unknown_rejected_count += 1
            else:
                totals["missed_ground_truth_objects"] += 1
            if truth["quality_label"] == "DEFECT":
                totals["gt_defect"] += 1
                if prediction is None:
                    totals["missed_objects"] += 1
                    totals["missed_defect"] += 1
                    continue
                totals["detected_defect"] += 1
                totals["matched_defect"] += 1
                if prediction.get("classifier_label") == DEFECT:
                    totals["classified_defect"] += 1
                if prediction.get("operational_status") in {DEFECT, REVIEW_REQUIRED}:
                    totals["review_aware_capture"] += 1
            else:
                totals["gt_good"] += 1
                if prediction is None:
                    totals["missed_good"] += 1
                else:
                    totals["matched_good"] += 1
                if prediction is not None and prediction.get("operational_status") in {DEFECT, REVIEW_REQUIRED}:
                    totals["good_flagged"] += 1
        unmatched = len(predictions) - len(matched)
        totals["ground_truth_objects"] += len(truths)
        totals["predicted_objects"] += len(predictions)
        totals["unmatched_predictions"] += unmatched
        totals["unmatched_false_positives"] += unmatched
        per_image.append(
            {
                "image_id": image["image_id"],
                "inspection_id": result["inspection_id"],
                "detected": len(predictions),
                "ground_truth_objects": len(truths),
                "predicted_objects": len(predictions),
                "matched_objects": len(matched),
                "missed_ground_truth_objects": len(truths) - len(matched),
                "unmatched_predictions": unmatched,
                "unmatched_false_positives": unmatched,
                "latency_ms": result.get("latency_ms"),
            }
        )
    def latency_summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None}
        numbers = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(numbers.mean()),
            "p50": float(np.percentile(numbers, 50)),
            "p95": float(np.percentile(numbers, 95)),
        }

    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "split": "test",
        "iou_threshold": iou_threshold,
        "product_recognizer_active": product_recognizer_active,
        "integrated_metrics": {
            **totals,
            "good_false_alarm_count": totals["good_flagged"],
            "end_to_end_defect_recall": totals["classified_defect"] / totals["gt_defect"] if totals["gt_defect"] else None,
            "review_aware_capture_rate": totals["review_aware_capture"] / totals["gt_defect"] if totals["gt_defect"] else None,
            "good_false_alarm_rate": totals["good_flagged"] / totals["gt_good"] if totals["gt_good"] else None,
            "matched_known_product_count": recognition_counts["known"],
            "matched_unknown_product_count": recognition_counts["unknown"],
            "matched_review_product_count": recognition_counts["review"],
            "known_product_objects": recognition_counts["known"],
            "unknown_product_objects": recognition_counts["unknown"],
            "review_product_objects": recognition_counts["review"],
            "unknown_false_accept_count": unknown_false_accept_count,
            "unknown_false_accept_rate": (
                unknown_false_accept_count / unknown_ground_truth_count
                if product_recognizer_active and unknown_ground_truth_count
                else None
            ),
            "unknown_rejection_rate": (
                unknown_rejected_count / unknown_ground_truth_count
                if product_recognizer_active and unknown_ground_truth_count
                else None
            ),
            "product_recognition_coverage": (
                known_recognized_matches / known_ground_truth_count
                if product_recognizer_active and known_ground_truth_count
                else None
            ),
            "known_product_recognition_coverage": (
                known_recognized_matches / known_ground_truth_count
                if product_recognizer_active and known_ground_truth_count
                else None
            ),
            "matched_known_product_recognition_coverage": (
                known_recognized_matches / matched_known_ground_truth
                if product_recognizer_active and matched_known_ground_truth
                else None
            ),
            "known_only_defect_recall": (
                known_recognized_classified_defect_count / known_defect_ground_truth_count
                if product_recognizer_active and known_defect_ground_truth_count
                else None
            ),
            "matched_known_quality_defect_recall": (
                known_recognized_classified_defect_count / known_recognized_defect_count
                if product_recognizer_active and known_recognized_defect_count
                else None
            ),
        },
        "per_image": per_image,
        "latency_ms": {
            "image_total": latency_summary(image_latencies),
            "object_total": latency_summary(object_latencies),
        },
        "warning": "This report is only independent if the configured test split was never used for model selection or adaptation.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _localizer_settings(config: ProductConfig) -> dict[str, Any]:
    settings = dict(config.model("localizer"))
    for key in (
        "runner",
        "repository",
        "checkpoint",
        "clip_cache",
        "calibration_path",
        "supervised_refiner_path",
        "supervised_refiner_calibration_path",
        "dino_artifact_root",
        "relation_prototype_path",
        "relation_calibration_path",
    ):
        if settings.get(key):
            settings[key] = str(config.project_path(settings[key]))
    return settings


def _localization_samples(config: ProductConfig, localizer: Localizer, split: str) -> list[dict[str, Any]]:
    images, objects = load_dataset_records(config)
    image_by_id = {row["image_id"]: row for row in images if row["split"] == split}
    samples: list[dict[str, Any]] = []
    for object_row in objects:
        image_row = image_by_id.get(object_row["image_id"])
        if image_row is None:
            continue
        image_path = config.dataset_path(Path(image_row["relative_path"]))
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        crop = oriented_crop(image, object_row["obb_xyxyxyxy"])
        truth = np.zeros(crop.image.shape[:2], dtype=bool)
        if object_row["mask_relative_path"]:
            mask_path = config.dataset_path(Path(object_row["mask_relative_path"]))
            source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if source_mask is None:
                raise ValueError(f"Cannot read ground-truth mask: {mask_path}")
            truth = cv2.warpPerspective(
                (source_mask > 0).astype(np.uint8),
                crop.forward_matrix,
                (crop.image.shape[1], crop.image.shape[0]),
                flags=cv2.INTER_NEAREST,
            ).astype(bool)
        result = localizer.localize(crop.image)
        if result.status != "completed" or result.heatmap is None:
            raise RuntimeError(f"AnomalyCLIP localization failed for {object_row['object_id']}: {result.reason}")
        heatmap = result.heatmap.astype(np.float32)
        if heatmap.shape != crop.image.shape[:2]:
            heatmap = cv2.resize(heatmap, (crop.image.shape[1], crop.image.shape[0]), interpolation=cv2.INTER_LINEAR)
        samples.append({"object_id": object_row["object_id"], "label": object_row["quality_label"], "truth": truth, "heatmap": heatmap, "crop": crop.image})
    if not samples:
        raise ValueError(f"No localizer samples found for split {split}")
    return samples


def _aggregate_mask_metrics(samples: list[dict[str, Any]], predictions: list[np.ndarray]) -> dict[str, float]:
    true_positive = false_positive = false_negative = true_negative = 0
    good_false_positive = good_pixels = 0
    predicted_area = truth_area = 0.0
    for sample, predicted in zip(samples, predictions):
        truth = sample["truth"]
        true_positive += int(np.logical_and(predicted, truth).sum())
        false_positive += int(np.logical_and(predicted, ~truth).sum())
        false_negative += int(np.logical_and(~predicted, truth).sum())
        true_negative += int(np.logical_and(~predicted, ~truth).sum())
        predicted_area += float(predicted.mean())
        truth_area += float(truth.mean())
        label = sample.get("label", "GOOD" if not truth.any() else "DEFECT")
        if label == "GOOD":
            good_false_positive += int(predicted.sum())
            good_pixels += int(predicted.size)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative else 1.0
    iou = true_positive / (true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative else 1.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    truth_flags = np.asarray([bool(sample["truth"].any()) for sample in samples], dtype=bool)
    prediction_flags = np.asarray([bool(predicted.any()) for predicted in predictions], dtype=bool)
    image_tp = int(np.logical_and(prediction_flags, truth_flags).sum())
    image_fp = int(np.logical_and(prediction_flags, ~truth_flags).sum())
    image_fn = int(np.logical_and(~prediction_flags, truth_flags).sum())
    image_precision = image_tp / (image_tp + image_fp) if image_tp + image_fp else 0.0
    image_recall = image_tp / (image_tp + image_fn) if image_tp + image_fn else 0.0
    image_f1 = 2 * image_precision * image_recall / (image_precision + image_recall) if image_precision + image_recall else 0.0
    good_samples = ~truth_flags
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": dice,
        "pixel_dice": dice,
        "pixel_iou": iou,
        "pixel_specificity": specificity,
        "pixel_false_positive_rate": 1.0 - specificity,
        "good_false_positive_rate": good_false_positive / good_pixels if good_pixels else false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0,
        "image_precision": image_precision,
        "image_recall": image_recall,
        "image_f1": image_f1,
        "good_image_false_positive_rate": int(np.logical_and(prediction_flags, good_samples).sum()) / int(good_samples.sum()) if good_samples.any() else 0.0,
        "mean_predicted_area_ratio": predicted_area / len(samples),
        "mean_truth_area_ratio": truth_area / len(samples),
    }


def _mask_metrics(samples: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    return _aggregate_mask_metrics(samples, [sample["heatmap"] >= threshold for sample in samples])


def _postprocessed_metrics(samples: list[dict[str, Any]], threshold: float, method: str) -> dict[str, float]:
    predictions = [postprocess_mask(sample["heatmap"], threshold, method) for sample in samples]
    return _aggregate_mask_metrics(samples, predictions)


def select_postprocess_threshold(samples: list[dict[str, Any]], method: str, target_recall: float = 0.90) -> tuple[float, dict[str, float]]:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    scores = np.concatenate([refine_heatmap(sample["heatmap"], method).ravel() for sample in samples])
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    measured = [(float(candidate), _postprocessed_metrics(samples, float(candidate), method)) for candidate in candidates]
    eligible = [item for item in measured if item[1]["pixel_recall"] >= target_recall]
    return min(eligible, key=lambda item: (item[1]["good_false_positive_rate"], -item[1]["pixel_recall"], -item[1]["pixel_precision"])) if eligible else max(measured, key=lambda item: (item[1]["pixel_recall"], item[1]["pixel_precision"], -item[1]["good_false_positive_rate"]))


def select_threshold_for_fpr(samples: list[dict[str, Any]], method: str, target_fpr: float = 0.20) -> tuple[float, dict[str, float]]:
    """Select the highest recall threshold that stays below the requested GOOD FP rate."""

    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    scores = np.concatenate([refine_heatmap(sample["heatmap"], method).ravel() for sample in samples])
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    measured = [(float(candidate), _postprocessed_metrics(samples, float(candidate), method)) for candidate in candidates]
    eligible = [item for item in measured if item[1]["good_false_positive_rate"] <= target_fpr]
    return max(eligible, key=lambda item: (item[1]["pixel_recall"], item[1]["pixel_precision"], -item[1]["good_false_positive_rate"])) if eligible else min(measured, key=lambda item: item[1]["good_false_positive_rate"])


def evaluate_localizer_methods(
    config: ProductConfig,
    report_path: str | Path,
    target_recall: float = 0.90,
    active_refiner: bool = False,
) -> dict[str, Any]:
    """Compare heatmap threshold/cleanup policies with validation-fitted thresholds."""

    localizer = build_localizer(_localizer_settings(config) if active_refiner else _relation_settings(config))
    validation = _localization_samples(config, localizer, "validation")
    test = _localization_samples(config, localizer, "test")
    methods = ("raw", "heatmap_open3", "heatmap_open5", "heatmap_open7", "heatmap_open9", "border2", "border5", "open3", "open5", "open3_close5", "components")
    results: dict[str, Any] = {}
    for method in methods:
        threshold, validation_metrics = select_postprocess_threshold(validation, method, target_recall)
        results[method] = {
            "mask_threshold": threshold,
            "validation": {**validation_metrics, "pixel_average_precision": _average_precision(validation, method)},
            "test": {**_postprocessed_metrics(test, threshold, method), "pixel_average_precision": _average_precision(test, method)},
        }
    best_method = min(results, key=lambda name: (results[name]["validation"]["good_false_positive_rate"], -results[name]["validation"]["pixel_recall"]))
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "localizer_mode": "active_supervised_refiner" if active_refiner else "baseline_anomalyclip",
        "target_validation_recall": target_recall,
        "selected_by_validation": best_method,
        "methods": results,
        "warning": "Test results are supervised-development diagnostics and must not be used for final model selection.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_dino_refinement(config: ProductConfig, report_path: str | Path, target_fpr: float = 0.20) -> dict[str, Any]:
    """Compare DINO graph refinement with raw AnomalyCLIP under an FP constraint."""

    localizer = build_localizer(_relation_settings(config))
    validation = _localization_samples(config, localizer, "validation")
    test = _localization_samples(config, localizer, "test")
    raw_validation = [{**sample, "heatmap": sample["heatmap"].copy()} for sample in validation]
    raw_test = [{**sample, "heatmap": sample["heatmap"].copy()} for sample in test]
    settings = dict(config.model("localizer"))
    if settings.get("dino_artifact_root"):
        settings["artifact_root"] = str(config.project_path(settings["dino_artifact_root"]))
    settings["top_k"] = settings.get("dino_top_k", 128)
    settings["temperature"] = settings.get("dino_temperature", 0.1)
    settings["eta"] = settings.get("dino_eta", 1.0)
    refiner = DinoAffinityRefiner(settings)
    for samples in (validation, test):
        for sample in samples:
            sample["heatmap"] = refiner.refine(sample["heatmap"], sample["crop"])
    # The DINO runs above replace only the working map; raw maps remain the comparison baseline.
    selected = {}
    for name, method in (("dino_raw", "raw"), ("dino_open3_close5", "open3_close5")):
        threshold, metrics = select_threshold_for_fpr(validation, method, target_fpr)
        selected[name] = {"mask_threshold": threshold, "validation": metrics, "test": _postprocessed_metrics(test, threshold, method)}
    raw_threshold, raw_metrics = select_threshold_for_fpr(raw_validation, "raw", target_fpr)
    selected["raw"] = {"mask_threshold": raw_threshold, "validation": raw_metrics, "test": _postprocessed_metrics(raw_test, raw_threshold, "raw")}
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "target_good_false_positive_rate": target_fpr,
        "selected_by_validation": max(selected, key=lambda name: (selected[name]["validation"]["pixel_recall"], selected[name]["validation"]["pixel_precision"])),
        "methods": selected,
        "warning": "Test results are supervised-development diagnostics and not an independent production estimate.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _average_precision(samples: list[dict[str, Any]], method: str = "raw") -> float:
    scores = np.concatenate([refine_heatmap(sample["heatmap"], method).ravel() for sample in samples])
    truth = np.concatenate([sample["truth"].ravel() for sample in samples]).astype(bool)
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked_truth = truth[order].astype(np.int64)
    cumulative = np.cumsum(ranked_truth)
    precision = cumulative / np.arange(1, len(ranked_truth) + 1)
    return float((precision * ranked_truth).sum() / positives)


def select_mask_threshold(samples: list[dict[str, Any]], target_recall: float = 0.90) -> tuple[float, dict[str, float]]:
    """Select the lowest-FPR threshold among candidates meeting pixel recall."""

    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    scores = np.concatenate([sample["heatmap"].ravel() for sample in samples])
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    measured = [(float(candidate), _mask_metrics(samples, float(candidate))) for candidate in candidates]
    eligible = [item for item in measured if item[1]["pixel_recall"] >= target_recall]
    selected = min(eligible, key=lambda item: (item[1]["good_false_positive_rate"], -item[1]["pixel_recall"], -item[1]["pixel_precision"])) if eligible else max(measured, key=lambda item: (item[1]["pixel_recall"], item[1]["pixel_precision"], -item[1]["good_false_positive_rate"]))
    return selected


def calibrate_localizer(config: ProductConfig, report_path: str | Path, target_recall: float = 0.90) -> dict[str, Any]:
    """Calibrate the AnomalyCLIP binary-mask threshold on validation only."""

    localizer = build_localizer(_relation_settings(config))
    samples = _localization_samples(config, localizer, "validation")
    method = str(config.model("localizer").get("mask_postprocess", "raw"))
    threshold, metrics = select_postprocess_threshold(samples, method, target_recall)
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "split": "validation",
        "mask_threshold": threshold,
        "mask_postprocess": method,
        "target_pixel_recall": target_recall,
        "metrics": {**metrics, "pixel_average_precision": _average_precision(samples, method)},
        "sample_count": len(samples),
        "warning": "Threshold is validation-calibrated; test metrics must remain untouched until evaluation.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_localizer(config: ProductConfig, report_path: str | Path, threshold: float | None = None) -> dict[str, Any]:
    """Evaluate raw heatmap and fixed binary-mask quality on a chosen split."""

    settings = _localizer_settings(config)
    if threshold is not None:
        settings["mask_threshold"] = threshold
    elif settings.get("calibration_path") and Path(settings["calibration_path"]).is_file():
        calibration = json.loads(Path(settings["calibration_path"]).read_text(encoding="utf-8"))
        settings["mask_threshold"] = float(calibration["mask_threshold"])
    localizer = build_localizer(settings)
    samples = _localization_samples(config, localizer, "test")
    selected_threshold = float(threshold if threshold is not None else settings.get("mask_threshold", 0.5))
    method = str(settings.get("mask_postprocess", "raw"))
    metrics = _postprocessed_metrics(samples, selected_threshold, method)
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "split": "test",
        "mask_threshold": selected_threshold,
        "mask_postprocess": method,
        "metrics": {**metrics, "pixel_average_precision": _average_precision(samples, method)},
        "sample_count": len(samples),
        "warning": "This test split is supervised-development data and is not an independent production estimate.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _relation_settings(config: ProductConfig) -> dict[str, Any]:
    settings = _localizer_settings(config)
    settings["relation_fusion_enabled"] = False
    settings["supervised_refiner_enabled"] = False
    return settings


def _relation_crop_rows(config: ProductConfig, split: str, quality_label: str | None = None) -> list[tuple[dict[str, Any], np.ndarray]]:
    images, objects = load_dataset_records(config)
    image_by_id = {row["image_id"]: row for row in images if row["split"] == split}
    rows: list[tuple[dict[str, Any], np.ndarray]] = []
    for object_row in objects:
        if quality_label is not None and object_row["quality_label"] != quality_label:
            continue
        image_row = image_by_id.get(object_row["image_id"])
        if image_row is None:
            continue
        source = cv2.imread(str(config.dataset_path(Path(image_row["relative_path"]))))
        if source is None:
            raise ValueError(f"Cannot read image {image_row['relative_path']}")
        image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        rows.append((object_row, oriented_crop(image, object_row["obb_xyxyxyxy"]).image))
    return rows


def _relation_ground_truth(config: ProductConfig, object_row: dict[str, Any], crop: np.ndarray) -> np.ndarray:
    truth = np.zeros(crop.shape[:2], dtype=bool)
    if object_row["mask_relative_path"]:
        image_rows, _ = load_dataset_records(config)
        image_row = next(row for row in image_rows if row["image_id"] == object_row["image_id"])
        source = cv2.imread(str(config.dataset_path(Path(image_row["relative_path"]))))
        original = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        oriented = oriented_crop(original, object_row["obb_xyxyxyxy"])
        mask = cv2.imread(str(config.dataset_path(Path(object_row["mask_relative_path"]))), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read ground-truth mask for {object_row['object_id']}")
        truth = cv2.warpPerspective((mask > 0).astype(np.uint8), oriented.forward_matrix, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
    return truth


def _build_relation_prototype(config: ProductConfig, session: AnomalyClipSession) -> np.ndarray:
    relations = []
    for object_row, crop in _relation_crop_rows(config, "train", "GOOD"):
        _, _, attention, features = session.infer(crop)
        relations.append(contextual_features(attention, features))
    if not relations:
        raise ValueError("No GOOD training crops are available for the relation prototype")
    prototype = np.mean(np.stack(relations), axis=0).astype(np.float32)
    prototype_path = config.project_path(config.model("localizer")["relation_prototype_path"])
    prototype_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prototype_path, prototype=prototype)
    return prototype


def _relation_samples(config: ProductConfig, session: AnomalyClipSession, prototype: np.ndarray, split: str) -> dict[str, list[dict[str, Any]]]:
    samples = {"baseline": [], "relation": [], "fusion": []}
    for object_row, crop in _relation_crop_rows(config, split):
        baseline, _, attention, features = session.infer(crop)
        relation = scores_to_map(relation_scores(attention, features, prototype), crop.shape[:2])
        baseline = cv2.resize(baseline, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        truth = _relation_ground_truth(config, object_row, crop)
        samples["baseline"].append({"heatmap": baseline, "truth": truth, "object_id": object_row["object_id"]})
        samples["relation"].append({"heatmap": relation, "truth": truth, "object_id": object_row["object_id"]})
        samples["fusion"].append({"heatmap": fuse_maps(baseline, relation), "truth": truth, "object_id": object_row["object_id"]})
    if not samples["baseline"]:
        raise ValueError(f"No relation samples found for split {split}")
    return samples


def calibrate_relation(config: ProductConfig, report_path: str | Path, target_recall: float = 0.90) -> dict[str, Any]:
    """Build GOOD relation prototypes and calibrate all three localization maps on validation."""

    session = AnomalyClipSession(_relation_settings(config))
    prototype = _build_relation_prototype(config, session)
    samples = _relation_samples(config, session, prototype, "validation")
    selected = {}
    for name, values in samples.items():
        threshold, metrics = select_mask_threshold(values, target_recall)
        selected[name] = {"mask_threshold": threshold, "metrics": {**metrics, "pixel_average_precision": _average_precision(values)}}
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "split": "validation",
        "target_pixel_recall": target_recall,
        "selected_map": "fusion",
        "mask_threshold": selected["fusion"]["mask_threshold"],
        "maps": selected,
        "prototype_path": str(config.model("localizer")["relation_prototype_path"]),
        "sample_count": len(samples["fusion"]),
        "warning": "Relation fusion remains experimental until test metrics improve over baseline.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_relation(config: ProductConfig, report_path: str | Path) -> dict[str, Any]:
    """Evaluate baseline, relation-only and equal-weight fusion using validation-fitted thresholds."""

    relation_config = config.model("localizer")
    calibration_path = config.project_path(relation_config["relation_calibration_path"])
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Relation calibration is missing: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    session = AnomalyClipSession(_relation_settings(config))
    prototype_path = config.project_path(relation_config["relation_prototype_path"])
    with np.load(prototype_path) as data:
        prototype = np.asarray(data["prototype"], dtype=np.float32)
    samples = _relation_samples(config, session, prototype, "test")
    metrics = {}
    for name, values in samples.items():
        metrics[name] = {**_mask_metrics(values, float(calibration["maps"][name]["mask_threshold"])), "pixel_average_precision": _average_precision(values)}
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "split": "test",
        "metrics": metrics,
        "prototype_path": str(relation_config["relation_prototype_path"]),
        "warning": "This test split is supervised-development data and is not an independent production estimate.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _supervised_refiner_settings(config: ProductConfig) -> dict[str, Any]:
    settings = _relation_settings(config)
    if settings.get("dino_artifact_root"):
        settings["artifact_root"] = str(config.project_path(settings["dino_artifact_root"]))
    settings["top_k"] = settings.get("dino_top_k", 128)
    settings["temperature"] = settings.get("dino_temperature", 0.1)
    settings["eta"] = settings.get("dino_eta", 1.0)
    return settings


def _fit_supervised_refiner(config: ProductConfig, session: AnomalyClipSession, dino: DinoAffinityRefiner) -> SupervisedPatchMaskRefiner:
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for object_row, crop in _relation_crop_rows(config, "train"):
        baseline, _, _, _ = session.infer(crop)
        dino_features, side = dino.patch_features(crop)
        feature_mode = str(config.model("localizer").get("supervised_refiner_feature_mode", "default"))
        feature_rows.append(patch_inputs(dino_features, baseline, side, feature_mode))
        truth = _relation_ground_truth(config, object_row, crop)
        labels = cv2.resize(truth.astype(np.uint8), (side, side), interpolation=cv2.INTER_AREA).reshape(-1) >= 0.25
        label_rows.append(labels.astype(np.int64))
    model = str(config.model("localizer").get("supervised_refiner_model", "logistic"))
    feature_mode = str(config.model("localizer").get("supervised_refiner_feature_mode", "default"))
    if model == "spatial_decoder":
        return SupervisedPatchMaskRefiner.fit_spatial_decoder(
            feature_rows,
            label_rows,
            feature_mode=feature_mode,
            epochs=int(config.model("localizer").get("supervised_refiner_epochs", 40)),
            learning_rate=float(config.model("localizer").get("supervised_refiner_learning_rate", 1e-3)),
        )
    return SupervisedPatchMaskRefiner.fit(
        np.concatenate(feature_rows),
        np.concatenate(label_rows),
        model=model,
        feature_mode=feature_mode,
    )


def _supervised_refiner_samples(config: ProductConfig, session: AnomalyClipSession, dino: DinoAffinityRefiner, refiner: SupervisedPatchMaskRefiner, split: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for object_row, crop in _relation_crop_rows(config, split):
        baseline, _, _, _ = session.infer(crop)
        dino_features, side = dino.patch_features(crop)
        refined = refiner.predict_map(
            patch_inputs(dino_features, baseline, side, refiner.feature_mode),
            side,
            crop.shape[:2],
        )
        samples.append({"heatmap": refined, "truth": _relation_ground_truth(config, object_row, crop), "label": object_row["quality_label"], "object_id": object_row["object_id"]})
    if not samples:
        raise ValueError(f"No supervised refiner samples found for split {split}")
    return samples


def train_supervised_refiner(config: ProductConfig, report_path: str | Path, target_fpr: float = 0.20) -> dict[str, Any]:
    """Fit a patch-level mask refiner on train masks and evaluate with validation-fitted threshold."""

    session = AnomalyClipSession(_supervised_refiner_settings(config))
    dino = DinoAffinityRefiner(_supervised_refiner_settings(config))
    refiner = _fit_supervised_refiner(config, session, dino)
    localizer_settings = config.model("localizer")
    artifact_path = config.project_path(localizer_settings["supervised_refiner_path"])
    refiner.save(artifact_path)
    validation = _supervised_refiner_samples(config, session, dino, refiner, "validation")
    test = _supervised_refiner_samples(config, session, dino, refiner, "test")
    threshold, validation_metrics = select_threshold_for_fpr(validation, "raw", target_fpr)
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "target_good_false_positive_rate": target_fpr,
        "mask_threshold": threshold,
        "mask_postprocess": "raw",
        "model": str(localizer_settings.get("supervised_refiner_model", "logistic")),
        "validation": validation_metrics,
        "test": _postprocessed_metrics(test, threshold, "raw"),
        "artifact_path": str(localizer_settings["supervised_refiner_path"]),
        "warning": "Test results are supervised-development diagnostics; do not claim production readiness.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _rgb_refiner_samples(config: ProductConfig, session: AnomalyClipSession, refiner: SupervisedRGBMaskRefiner | None, split: str) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    image_size = int(config.model("localizer").get("supervised_rgb_image_size", 224))
    for object_row, crop in _relation_crop_rows(config, split):
        baseline, _, _, _ = session.infer(crop)
        truth = _relation_ground_truth(config, object_row, crop)
        if refiner is None:
            inputs.append(SupervisedRGBMaskRefiner.prepare_input(crop, baseline, image_size))
            targets.append(SupervisedRGBMaskRefiner.prepare_target(truth, image_size))
        else:
            refined = refiner.predict_map(crop, baseline)
            samples.append({"heatmap": refined, "truth": truth, "label": object_row["quality_label"], "object_id": object_row["object_id"]})
    return inputs, targets, samples


def train_supervised_rgb_refiner(config: ProductConfig, report_path: str | Path, target_fpr: float = 0.20) -> dict[str, Any]:
    """Train and evaluate an RGB-plus-AnomalyCLIP mask decoder on train masks only."""

    session = AnomalyClipSession(_relation_settings(config))
    localizer_settings = config.model("localizer")
    image_size = int(localizer_settings.get("supervised_rgb_image_size", 224))
    train_inputs, train_targets, _ = _rgb_refiner_samples(config, session, None, "train")
    refiner = SupervisedRGBMaskRefiner.fit(
        np.stack(train_inputs),
        np.stack(train_targets),
        image_size=image_size,
        epochs=int(localizer_settings.get("supervised_rgb_epochs", 25)),
        learning_rate=float(localizer_settings.get("supervised_rgb_learning_rate", 1e-3)),
        batch_size=int(localizer_settings.get("supervised_rgb_batch_size", 8)),
    )
    artifact_path = config.project_path(localizer_settings["supervised_rgb_refiner_path"])
    refiner.save(artifact_path)
    _, _, validation = _rgb_refiner_samples(config, session, refiner, "validation")
    _, _, test = _rgb_refiner_samples(config, session, refiner, "test")
    threshold, validation_metrics = select_threshold_for_fpr(validation, "raw", target_fpr)
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "target_good_false_positive_rate": target_fpr,
        "mask_threshold": threshold,
        "mask_postprocess": "raw",
        "model": "rgb_unet_plus_anomalyclip",
        "validation": {**validation_metrics, "pixel_average_precision": _average_precision(validation)},
        "test": {**_postprocessed_metrics(test, threshold, "raw"), "pixel_average_precision": _average_precision(test)},
        "artifact_path": str(localizer_settings["supervised_rgb_refiner_path"]),
        "warning": "Test results are supervised-development diagnostics; do not claim production readiness.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train_synthetic_segmenter(config: ProductConfig, report_path: str | Path, target_fpr: float = 0.20) -> dict[str, Any]:
    """Train a one-class synthetic-defect segmenter using GOOD train crops only."""

    localizer_settings = config.model("localizer")
    image_size = int(localizer_settings.get("synthetic_segmenter_image_size", 224))
    train_crops = [crop for _, crop in _relation_crop_rows(config, "train", "GOOD")]
    refiner = SyntheticDefectSegmenter.fit(
        train_crops,
        image_size=image_size,
        epochs=int(localizer_settings.get("synthetic_segmenter_epochs", 20)),
        learning_rate=float(localizer_settings.get("synthetic_segmenter_learning_rate", 1e-3)),
        batch_size=int(localizer_settings.get("synthetic_segmenter_batch_size", 8)),
    )
    artifact_path = config.project_path(localizer_settings["synthetic_segmenter_path"])
    refiner.save(artifact_path)

    def evaluate_split(split: str) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for object_row, crop in _relation_crop_rows(config, split):
            samples.append({
                "heatmap": refiner.predict_map(crop),
                "truth": _relation_ground_truth(config, object_row, crop),
                "label": object_row["quality_label"],
                "object_id": object_row["object_id"],
            })
        return samples

    validation = evaluate_split("validation")
    test = evaluate_split("test")
    threshold, validation_metrics = select_threshold_for_fpr(validation, "raw", target_fpr)
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "training_regime": "GOOD-only train crops with synthetic anomaly masks",
        "target_good_false_positive_rate": target_fpr,
        "mask_threshold": threshold,
        "mask_postprocess": "raw",
        "model": "synthetic_defect_unet",
        "validation": {**validation_metrics, "pixel_average_precision": _average_precision(validation)},
        "test": {**_postprocessed_metrics(test, threshold, "raw"), "pixel_average_precision": _average_precision(test)},
        "artifact_path": str(localizer_settings["synthetic_segmenter_path"]),
        "warning": "Test results are supervised-development diagnostics; do not claim production readiness.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
