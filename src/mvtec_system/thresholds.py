"""Calibration of image and pixel anomaly decision thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


GOOD_LABEL = "GOOD"
DEFECT_LABEL = "DEFECT"


def _score_tensor(scores: Any) -> torch.Tensor:
    """Convert scores to one finite one-dimensional tensor."""

    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
    if values.numel() == 0:
        raise ValueError("scores must contain at least one value.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores must contain only finite values.")
    return values


def calibrate_quantile_threshold(
    scores: Any,
    target_acceptance_rate: float = 0.99,
) -> dict[str, Any]:
    """Calibrate a high-score threshold from known-normal scores only."""

    if not 0.0 < target_acceptance_rate <= 1.0:
        raise ValueError("target_acceptance_rate must be in the interval (0, 1].")
    values = _score_tensor(scores)
    threshold = torch.quantile(
        values,
        target_acceptance_rate,
        interpolation="linear",
    ).item()
    observed_acceptance = (values <= threshold).float().mean().item()
    return {
        "threshold": threshold,
        "target_acceptance_rate": target_acceptance_rate,
        "observed_acceptance_rate": observed_acceptance,
        "sample_count": int(values.numel()),
        "quantile": target_acceptance_rate,
        "quantile_interpolation": "linear",
        "decision_rule": "score >= threshold -> DEFECT; otherwise GOOD",
    }


def calibrate_anomaly_thresholds(
    image_scores: Any,
    pixel_scores: Any,
    target_acceptance_rate: float = 0.99,
) -> dict[str, Any]:
    """Calibrate image and pixel thresholds from known-normal validation data."""

    image = calibrate_quantile_threshold(image_scores, target_acceptance_rate)
    pixel = calibrate_quantile_threshold(pixel_scores, target_acceptance_rate)
    return {
        "artifact_version": "v1",
        "calibration_split": "known_normal_validation",
        "target_acceptance_rate": target_acceptance_rate,
        "image": image,
        "pixel": pixel,
        "protocol": {
            "strategy": "normal_only_quantile",
            "description": "Upper quantile of continuous scores on held-out train/good validation images.",
            "test_labels_used": False,
            "defect_recall_claim": False,
            "threshold_rule": "score >= threshold -> DEFECT; otherwise GOOD",
        },
    }


def image_scores_from_patch_scores(patch_scores: Any) -> torch.Tensor:
    """Convert each image's patch scores to its maximum image score."""

    scores = torch.as_tensor(patch_scores, dtype=torch.float32)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("patch_scores must have shape [images, patches] and be non-empty.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("patch_scores must contain only finite values.")
    return scores.max(dim=1).values.cpu()


def calibrate_category_image_thresholds(
    category_scores: dict[str, Any],
    target_acceptance_rate: float = 0.99,
) -> dict[str, Any]:
    """Calibrate one image threshold per category from normal validation scores."""

    if not isinstance(category_scores, dict) or not category_scores:
        raise ValueError("category_scores must be a non-empty mapping.")
    categories = {}
    for category, scores in sorted(category_scores.items()):
        if not isinstance(category, str) or not category:
            raise ValueError("Category names must be non-empty strings.")
        categories[category] = calibrate_quantile_threshold(scores, target_acceptance_rate)
    return {
        "artifact_version": "v1",
        "threshold_type": "image_level",
        "calibration_split": "known_normal_validation",
        "target_acceptance_rate": target_acceptance_rate,
        "categories": categories,
        "protocol": {
            "strategy": "normal_only_quantile",
            "score_aggregation": "maximum_patch_score",
            "test_labels_used": False,
            "threshold_rule": "score >= threshold -> DEFECT; otherwise GOOD",
        },
    }


def classify_image_score(
    category: str,
    score: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply a persisted category threshold to one continuous image score."""

    categories = thresholds.get("categories")
    if not isinstance(categories, dict) or category not in categories:
        raise ValueError(f"No image threshold is available for category: {category}")
    score_value = _score_tensor([score]).item()
    category_threshold = categories[category]
    if not isinstance(category_threshold, dict) or "threshold" not in category_threshold:
        raise ValueError(f"Invalid threshold metadata for category: {category}")
    threshold = float(category_threshold["threshold"])
    if not bool(torch.isfinite(torch.tensor(threshold))):
        raise ValueError("Threshold metadata must contain a finite threshold.")
    is_defect = score_value >= threshold
    return {
        "category": category,
        "score": score_value,
        "threshold": threshold,
        "label": DEFECT_LABEL if is_defect else GOOD_LABEL,
        "is_defect": is_defect,
    }


def save_anomaly_thresholds(path: str | Path, thresholds: dict[str, Any]) -> None:
    """Write calibrated anomaly thresholds as a JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
