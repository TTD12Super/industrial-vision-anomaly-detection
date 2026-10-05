"""Validation metrics for the product-category classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from .classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
)


def _binary_values(values: Any, name: str) -> np.ndarray:
    """Convert binary labels or predictions to a one-dimensional array."""

    array = np.asarray(values).reshape(-1)
    if array.size == 0 or not np.isfinite(array.astype(float)).all():
        raise ValueError(f"{name} must contain finite values.")
    if not np.isin(array, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only 0/1 values.")
    return array.astype(np.int64)


def _continuous_values(values: Any, name: str) -> np.ndarray:
    """Convert continuous scores to a finite one-dimensional array."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain at least one finite value.")
    return array


def _threshold_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Return binary metrics using a fixed prediction already made by a threshold."""

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(
            true_negative / specificity_denominator if specificity_denominator else 0.0
        ),
        "dice": float(
            2 * true_positive / dice_denominator if dice_denominator else 0.0
        ),
        "iou": float(true_positive / iou_denominator if iou_denominator else 0.0),
        "confusion_matrix": matrix.tolist(),
    }


def _ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return continuous-score metrics, requiring both binary classes."""

    if np.unique(labels).size < 2:
        raise ValueError("Ranking metrics require both GOOD and DEFECT labels.")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_anomaly_detection(
    image_scores: Any,
    image_labels: Any,
    pixel_scores: Any,
    pixel_labels: Any,
    image_predictions: Any | None = None,
    pixel_predictions: Any | None = None,
) -> dict[str, Any]:
    """Evaluate image and pixel anomaly scores and optional fixed decisions.

    Ranking metrics use continuous scores. Thresholded metrics are calculated
    only from predictions produced after calibration, so this function never
    selects a threshold from evaluation labels.
    """

    image_values = _continuous_values(image_scores, "image_scores")
    image_truth = _binary_values(image_labels, "image_labels")
    if image_values.shape != image_truth.shape:
        raise ValueError("image_scores and image_labels must have the same length.")
    pixel_values = _continuous_values(pixel_scores, "pixel_scores")
    pixel_truth = _binary_values(pixel_labels, "pixel_labels")
    if pixel_values.shape != pixel_truth.shape:
        raise ValueError("pixel_scores and pixel_labels must have the same length.")

    image_report: dict[str, Any] = {
        "sample_count": int(image_values.size),
        "ranking": _ranking_metrics(image_values, image_truth),
    }
    if image_predictions is not None:
        image_pred = _binary_values(image_predictions, "image_predictions")
        if image_pred.shape != image_truth.shape:
            raise ValueError("image_predictions and image_labels must have the same length.")
        image_report["thresholded"] = _threshold_metrics(image_truth, image_pred)

    pixel_report: dict[str, Any] = {
        "sample_count": int(pixel_values.size),
        "ranking": _ranking_metrics(pixel_values, pixel_truth),
    }
    if pixel_predictions is not None:
        pixel_pred = _binary_values(pixel_predictions, "pixel_predictions")
        if pixel_pred.shape != pixel_truth.shape:
            raise ValueError("pixel_predictions and pixel_labels must have the same length.")
        pixel_report["thresholded"] = _threshold_metrics(pixel_truth, pixel_pred)

    return {"image": image_report, "pixel": pixel_report}


def evaluate_product_classifier(
    classifier: dict[str, Any],
    artifact: dict[str, Any],
    class_mapping: dict[str, int],
) -> dict[str, Any]:
    """Calculate classification metrics for one held-out embedding split."""

    labels = artifact.get("labels")
    embeddings = artifact.get("embeddings")
    if not isinstance(labels, torch.Tensor) or not isinstance(embeddings, torch.Tensor):
        raise ValueError("Evaluation artifact must contain tensor embeddings and labels.")
    if labels.ndim != 1 or embeddings.ndim != 2 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Evaluation embedding and label shapes do not match.")
    logits, probabilities = predict_linear_classifier(classifier, embeddings)
    predictions = logits.argmax(dim=1)
    class_ids = sorted(class_mapping.values())
    class_names = [
        category for category, index in sorted(class_mapping.items(), key=lambda item: item[1])
    ]
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    precision = precision_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    top_indices = probabilities.topk(min(3, probabilities.shape[1]), dim=1).indices
    top_k_accuracy = top_indices.eq(labels.cpu().unsqueeze(1)).any(dim=1).float().mean().item()

    per_class = {}
    for index, category in zip(range(len(class_names)), class_names):
        per_class[category] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int((y_true == class_ids[index]).sum()),
        }
    return {
        "split": artifact.get("split", "unknown"),
        "sample_count": int(labels.shape[0]),
        "class_mapping": class_mapping,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "top_3_accuracy": float(top_k_accuracy),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        "per_class": per_class,
    }


def save_classification_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write metrics as a readable JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def select_linear_classifier(
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    configs: list[LinearClassifierConfig],
) -> tuple[dict[str, Any], LinearClassifierConfig, dict[str, Any], list[dict[str, Any]]]:
    """Select a classifier using validation macro-F1, never the official test set."""

    if not configs:
        raise ValueError("At least one classifier configuration is required.")
    candidates = []
    best = None
    for config in configs:
        classifier = fit_linear_classifier(train_artifact, class_mapping, config)
        report = evaluate_product_classifier(classifier, validation_artifact, class_mapping)
        metrics = report["metrics"]
        score = (
            metrics["macro_f1"],
            metrics["accuracy"],
            metrics["top_3_accuracy"],
        )
        candidate = {"config": config.__dict__, "metrics": metrics}
        candidates.append(candidate)
        if best is None or score > best["score"]:
            best = {
                "classifier": classifier,
                "config": config,
                "report": report,
                "score": score,
            }
    best["report"]["model_selection"] = {
        "selection_metric": "macro_f1",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_config": best["config"].__dict__,
    }
    return best["classifier"], best["config"], best["report"], candidates
