"""Validation-only confidence calibration for the operational YOLO-OBB detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ProductConfig
from .dataset import load_dataset_records
from .evaluation import polygon_iou


def select_confidence(candidates: list[dict[str, float]], target_recall: float) -> dict[str, float]:
    """Choose the strictest confidence meeting validation detection-recall target."""

    eligible = [item for item in candidates if item["recall"] >= target_recall]
    if not eligible:
        raise ValueError(f"No detector confidence reached validation recall target {target_recall}")
    return max(eligible, key=lambda item: (item["confidence"], item["precision"]))


def _metrics(predictions: list[list[list[float]]], truth: list[list[list[float]]], iou_threshold: float) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for image_predictions, image_truth in zip(predictions, truth):
        unmatched = set(range(len(image_truth)))
        for prediction in image_predictions:
            overlap, index = max(
                ((polygon_iou(prediction, image_truth[candidate]), candidate) for candidate in unmatched),
                default=(0.0, -1),
            )
            if overlap >= iou_threshold:
                true_positive += 1
                unmatched.remove(index)
            else:
                false_positive += 1
        false_negative += len(unmatched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {"precision": precision, "recall": recall, "true_positive": float(true_positive), "false_positive": float(false_positive), "false_negative": float(false_negative)}


def calibrate_detector_confidence(config: ProductConfig) -> dict[str, Any]:
    """Measure candidate confidence values and persist only a valid calibration."""

    output = measure_detector_confidence(config)
    selected = output.get("selected_metrics")
    if selected is None:
        raise ValueError(
            f"No detector confidence reached validation recall target {output['target_recall']}"
        )
    destination = config.project_path(config.model("detector")["calibration_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def measure_detector_confidence(config: ProductConfig) -> dict[str, Any]:
    """Measure all confidence candidates without mutating the runtime calibration.

    This diagnostic path is intentionally separate from ``calibrate``: an
    unattainable target must produce evidence for data/model remediation, not a
    silently weakened production threshold.
    """

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("ultralytics is required for detector calibration") from error
    images, objects = load_dataset_records(config)
    object_by_image: dict[str, list[dict[str, Any]]] = {}
    for object_row in objects:
        object_by_image.setdefault(object_row["image_id"], []).append(object_row)
    validation = [row for row in images if row["split"] == "validation"]
    if not validation:
        raise ValueError("Validation images are required for detector calibration")
    settings = config.model("detector")
    weights = config.project_path(settings["weights"])
    model = YOLO(str(weights))
    confidences = [float(value) for value in settings.get("confidence_candidates", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])]
    truths = [[item["obb_xyxyxyxy"] for item in object_by_image[row["image_id"]]] for row in validation]
    candidates: list[dict[str, float]] = []
    for confidence in confidences:
        predictions: list[list[list[float]]] = []
        for row in validation:
            result = model.predict(
                str(config.dataset_path(Path(row["relative_path"]))),
                conf=confidence,
                imgsz=int(settings.get("image_size", 1024)),
                verbose=False,
            )[0]
            obb = result.obb
            points = [] if obb is None else obb.xyxyxyxy.cpu().numpy().tolist()
            predictions.append(points)
        candidates.append({"confidence": confidence, **_metrics(predictions, truths, float(settings.get("calibration_iou", 0.5)))})
    target_recall = float(config.policy.get("target_detector_recall", 0.99))
    try:
        selected = select_confidence(candidates, target_recall)
    except ValueError:
        selected = None
    output = {
        "schema_version": "1.0",
        "source_split": "validation",
        "iou_threshold": float(settings.get("calibration_iou", 0.5)),
        "target_recall": target_recall,
        "target_reached": selected is not None,
        "confidence": selected["confidence"] if selected is not None else None,
        "selected_metrics": selected,
        "candidates": candidates,
        "checkpoint": str(weights),
    }
    return output
