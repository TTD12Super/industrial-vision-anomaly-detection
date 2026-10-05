"""Calibration of image and pixel anomaly decision thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


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
        "decision_rule": "score > threshold -> DEFECT; otherwise GOOD",
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
            "threshold_rule": "score > threshold -> DEFECT; otherwise GOOD",
        },
    }


def save_anomaly_thresholds(path: str | Path, thresholds: dict[str, Any]) -> None:
    """Write calibrated anomaly thresholds as a JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
