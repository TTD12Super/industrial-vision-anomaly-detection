"""Validation-only open-set calibration for product routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .classifier import predict_linear_classifier


UNKNOWN_PRODUCT_CATEGORY = "unknown_product_category"


@torch.inference_mode()
def nearest_global_distances(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return exact Euclidean distance to the nearest reference embedding."""

    if query_embeddings.ndim != 2 or reference_embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional tensors.")
    if query_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("Query and reference dimensions do not match.")
    if reference_embeddings.shape[0] == 0 or chunk_size < 1:
        raise ValueError("Reference embeddings must be non-empty and chunk_size must be positive.")
    query_embeddings = query_embeddings.float().cpu()
    reference_embeddings = reference_embeddings.float().cpu()
    distances = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        chunk = query_embeddings[start : start + chunk_size]
        distances.append(torch.cdist(chunk, reference_embeddings, p=2).min(dim=1).values)
    return torch.cat(distances)


def calibrate_open_set_thresholds(
    classifier: dict[str, Any],
    reference_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    target_acceptance_rate: float = 0.95,
) -> dict[str, Any]:
    """Choose confidence and distance thresholds from known validation images."""

    if not 0.0 < target_acceptance_rate <= 1.0:
        raise ValueError("target_acceptance_rate must be in the interval (0, 1].")
    reference_embeddings = reference_artifact.get("embeddings")
    validation_embeddings = validation_artifact.get("embeddings")
    if not isinstance(reference_embeddings, torch.Tensor) or not isinstance(validation_embeddings, torch.Tensor):
        raise ValueError("Calibration artifacts must contain tensor embeddings.")
    _, probabilities = predict_linear_classifier(classifier, validation_embeddings)
    confidences = probabilities.max(dim=1).values
    distances = nearest_global_distances(validation_embeddings, reference_embeddings)
    confidence_threshold = torch.quantile(
        confidences, 1.0 - target_acceptance_rate, interpolation="linear"
    ).item()
    distance_threshold = torch.quantile(
        distances, target_acceptance_rate, interpolation="linear"
    ).item()
    confidence_known = confidences >= confidence_threshold
    distance_known = distances <= distance_threshold
    accepted = confidence_known & distance_known
    return {
        "artifact_version": "v1",
        "unknown_category": UNKNOWN_PRODUCT_CATEGORY,
        "class_mapping": class_mapping,
        "reference_split": reference_artifact.get("split", "train"),
        "calibration_split": validation_artifact.get("split", "validation"),
        "target_acceptance_rate": target_acceptance_rate,
        "confidence_threshold": confidence_threshold,
        "nearest_global_distance_threshold": distance_threshold,
        "known_validation_acceptance_rate": accepted.float().mean().item(),
        "confidence_only_acceptance_rate": confidence_known.float().mean().item(),
        "distance_only_acceptance_rate": distance_known.float().mean().item(),
        "calibration_protocol": {
            "confidence": "lower quantile of known validation max softmax confidence",
            "distance": "upper quantile of known validation minimum Euclidean distance to train embeddings",
            "quantile_interpolation": "linear",
            "reject_rule": "confidence < threshold OR nearest_global_distance > threshold",
            "unknown_examples_used": False,
        },
    }


def save_calibration(path: str | Path, calibration: dict[str, Any]) -> None:
    """Write calibrated routing thresholds as a JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
