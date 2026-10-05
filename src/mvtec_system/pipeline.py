"""Unified one-image inference for classification and anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .dinov2 import SharedDINOv2, image_tensor
from .knn import nearest_neighbor_scores
from .maps import make_anomaly_maps
from .router import route_product
from .thresholds import (
    binary_anomaly_masks,
    classify_image_score,
    image_scores_from_patch_scores,
)


def _load_inference_bank(
    bank_directory: str | Path,
    category: str,
    embedding_dimension: int,
) -> torch.Tensor:
    """Load one already-validated category bank for inference."""

    path = Path(bank_directory) / "feature_bank.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Feature bank is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Feature bank payload must be a dictionary.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("category") != category:
        raise ValueError("Feature bank category metadata does not match the route.")
    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        raise ValueError("Feature bank payload is missing tensor features.")
    if features.ndim != 2 or features.shape[1] != embedding_dimension:
        raise ValueError("Feature bank dimension does not match patch features.")
    if features.shape[0] == 0 or not bool(torch.isfinite(features).all()):
        raise ValueError("Feature bank must be non-empty and finite.")
    return features.float().cpu()


def infer_image(
    image_path: str | Path,
    encoder: SharedDINOv2,
    classifier: dict[str, Any],
    reference_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    calibration: dict[str, Any],
    bank_root: str | Path,
    image_thresholds: dict[str, Any] | None = None,
    pixel_thresholds: dict[str, Any] | None = None,
    k: int = 1,
    query_chunk_size: int = 512,
    bank_chunk_size: int = 16384,
) -> dict[str, Any]:
    """Run classification, routing, kNN, map creation, and optional decisions."""

    if k < 1:
        raise ValueError("k must be positive.")
    encoder_metadata = encoder.metadata()
    images = image_tensor(image_path, encoder.config).unsqueeze(0)
    features = encoder.forward(images)
    route = route_product(
        features.global_embedding,
        classifier,
        reference_artifact,
        class_mapping,
        calibration,
        bank_root,
        encoder_metadata,
    )
    result = route.to_dict()
    result.update(
        {
            "image_path": str(image_path),
            "anomaly_score": None,
            "threshold": None,
            "label": None,
            "patch_scores": None,
            "anomaly_map": None,
            "binary_mask": None,
        }
    )
    if route.rejected or route.bank_status != "compatible" or route.bank_path is None:
        return result

    patch_embeddings = features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1])
    bank = _load_inference_bank(
        route.bank_path,
        route.category,
        patch_embeddings.shape[1],
    )
    patch_scores = nearest_neighbor_scores(
        patch_embeddings,
        bank,
        k=k,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
    ).reshape(1, -1)
    image_score = float(image_scores_from_patch_scores(patch_scores)[0].item())
    anomaly_map = make_anomaly_maps(
        patch_scores,
        encoder.patch_grid,
        encoder.padded_size,
        encoder.config.image_size,
    )[0]
    result["anomaly_score"] = image_score
    result["patch_scores"] = patch_scores[0].numpy()
    result["anomaly_map"] = anomaly_map

    if image_thresholds is not None:
        decision = classify_image_score(route.category, image_score, image_thresholds)
        result["threshold"] = decision["threshold"]
        result["label"] = decision["label"]
        result["is_defect"] = decision["is_defect"]
    if pixel_thresholds is not None:
        result["binary_mask"] = binary_anomaly_masks(
            route.category,
            anomaly_map[None, ...],
            pixel_thresholds,
        )[0].numpy()
    return result
