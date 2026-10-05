"""Route product images to a compatible anomaly feature bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .calibration import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    UNKNOWN_PRODUCT_CATEGORY,
    nearest_global_distances,
)
from .classifier import predict_linear_classifier


KNOWN_PRODUCT_CATEGORY = "known_product_category"


@dataclass(frozen=True)
class BankResolution:
    """Result of checking one category's existing local bank."""

    status: str
    path: str | None
    reason: str
    requires_rebuild: bool


@dataclass(frozen=True)
class RouteResult:
    """Classification and routing decision for one product image."""

    status: str
    category: str | None
    predicted_category: str
    classification_enabled: bool
    confidence: float
    nearest_global_distance: float
    global_distance_threshold: float
    global_distance_ood: bool
    rejected: bool
    bank_status: str
    bank_path: str | None
    reason: str
    requires_rebuild: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classification_confidence_threshold(
    calibration: dict[str, Any],
    override: float | None,
) -> float:
    """Read the mandatory product-confidence threshold from configuration."""

    value = (
        DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD
        if override is None
        else override
    )
    if override is None and "classification_confidence_threshold" in calibration:
        value = calibration["classification_confidence_threshold"]
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("classification confidence threshold must be between 0 and 1")
    return value


def routing_rejection_reasons(
    confidence: float,
    nearest_global_distance: float,
    calibration: dict[str, Any],
    classification_confidence_threshold: float | None = None,
) -> list[str]:
    """Apply the single configured product-confidence rejection rule."""

    del nearest_global_distance
    confidence_threshold = _classification_confidence_threshold(
        calibration, classification_confidence_threshold
    )
    reasons = []
    if confidence < confidence_threshold:
        reasons.append(
            "product confidence below threshold "
            f"({confidence:.6f} < {confidence_threshold:.6f})"
        )
    return reasons


def routing_decision(
    predicted_category: str,
    confidence: float,
    nearest_global_distance: float,
    calibration: dict[str, Any],
    classification_confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Return the confidence gate and non-rejecting global-distance flag."""

    reasons = routing_rejection_reasons(
        confidence,
        nearest_global_distance,
        calibration,
        classification_confidence_threshold,
    )
    rejected = bool(reasons)
    global_distance_threshold = float(
        calibration.get(
            "global_distance_defect_threshold",
            calibration.get("nearest_global_distance_threshold", float("inf")),
        )
    )
    global_distance_ood = nearest_global_distance > global_distance_threshold
    return {
        "status": UNKNOWN_PRODUCT_CATEGORY if rejected else KNOWN_PRODUCT_CATEGORY,
        "category": None if rejected else predicted_category,
        "rejected": rejected,
        "global_distance_threshold": global_distance_threshold,
        "global_distance_ood": global_distance_ood,
        "reason": "; ".join(reasons) if rejected else "product category accepted",
    }


def _inverse_mapping(class_mapping: dict[str, int]) -> dict[int, str]:
    inverse = {index: category for category, index in class_mapping.items()}
    if sorted(inverse) != list(range(len(inverse))):
        raise ValueError("Class mapping indices must be contiguous from zero.")
    return inverse


def _expected_bank_values(encoder_metadata: dict[str, Any]) -> dict[str, Any]:
    """Translate shared encoder metadata to the existing bank metadata schema."""

    return {
        "model_name": encoder_metadata["model_name"],
        "model_revision": encoder_metadata["model_revision"],
        "feature_key": encoder_metadata["patch_feature_key"],
        "image_size": encoder_metadata["image_size"],
        "padding": encoder_metadata["padding"],
        "mean": encoder_metadata["mean"],
        "std": encoder_metadata["std"],
        "l2_normalization": encoder_metadata["l2_normalize_patches"],
        "embedding_dimension": encoder_metadata["embedding_dimension"],
        "patch_grid": encoder_metadata["patch_grid"],
    }


def resolve_category_bank(
    bank_root: str | Path,
    category: str,
    encoder_metadata: dict[str, Any] | None = None,
) -> BankResolution:
    """Find a bank and verify its metadata without loading its tensor."""

    root = Path(bank_root).resolve()
    bank_dir = (root / category).resolve()
    if not bank_dir.is_relative_to(root):
        raise ValueError("Category would leave the feature-bank root.")
    bank_path = str(bank_dir)
    feature_bank = bank_dir / "feature_bank.pt"
    metadata_path = bank_dir / "bank_metadata.json"
    if not feature_bank.is_file() or not metadata_path.is_file():
        return BankResolution("missing", bank_path, "feature bank or metadata is missing", True)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return BankResolution("incompatible", bank_path, "bank metadata is unreadable", True)
    if metadata.get("category") != category:
        return BankResolution("incompatible", bank_path, "bank category does not match route", True)
    if encoder_metadata is not None:
        expected = _expected_bank_values(encoder_metadata)
        mismatches = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        if mismatches:
            return BankResolution(
                "incompatible",
                bank_path,
                "bank metadata mismatch: " + ", ".join(mismatches),
                True,
            )
    return BankResolution("compatible", bank_path, "compatible bank found", False)


def route_product(
    global_embedding: torch.Tensor,
    classifier: dict[str, Any] | None,
    reference_artifact: dict[str, Any] | None,
    class_mapping: dict[str, int],
    calibration: dict[str, Any],
    bank_root: str | Path,
    encoder_metadata: dict[str, Any] | None = None,
    classification_confidence_threshold: float | None = None,
    resolve_bank: bool = True,
) -> RouteResult:
    """Route an image, bypassing classification when only one category exists."""

    if global_embedding.ndim == 1:
        global_embedding = global_embedding.unsqueeze(0)
    if global_embedding.ndim != 2 or global_embedding.shape[0] != 1:
        raise ValueError("Router expects one global embedding.")
    inverse_mapping = _inverse_mapping(class_mapping)
    classification_enabled = len(inverse_mapping) > 1
    if classification_enabled:
        if classifier is None:
            raise ValueError("A classifier artifact is required for multiple categories.")
        logits, probabilities = predict_linear_classifier(classifier, global_embedding)
        predicted_index = int(logits.argmax(dim=1).item())
        predicted_category = inverse_mapping[predicted_index]
        confidence = float(probabilities.max(dim=1).values.item())
    else:
        predicted_category = next(iter(inverse_mapping.values()))
        confidence = 1.0

    reference_embeddings = (
        reference_artifact.get("embeddings")
        if reference_artifact is not None
        else None
    )
    if reference_embeddings is None:
        distance = 0.0
    elif isinstance(reference_embeddings, torch.Tensor):
        distance = float(nearest_global_distances(global_embedding, reference_embeddings)[0].item())
    else:
        raise ValueError("Reference artifact must contain tensor embeddings.")
    decision = routing_decision(
        predicted_category,
        confidence,
        distance,
        calibration,
        classification_confidence_threshold,
    )
    if not classification_enabled:
        decision["status"] = KNOWN_PRODUCT_CATEGORY
        decision["category"] = predicted_category
        decision["rejected"] = False
        decision["reason"] = "classification disabled: dataset has one category"
    if decision["rejected"]:
        return RouteResult(
            status=decision["status"],
            category=None,
            predicted_category=predicted_category,
            classification_enabled=classification_enabled,
            confidence=confidence,
            nearest_global_distance=distance,
            global_distance_threshold=decision["global_distance_threshold"],
            global_distance_ood=decision["global_distance_ood"],
            rejected=True,
            bank_status="skipped",
            bank_path=None,
            reason=decision["reason"],
            requires_rebuild=False,
        )

    bank = (
        resolve_category_bank(bank_root, predicted_category, encoder_metadata)
        if resolve_bank
        else BankResolution(
            "not_requested",
            None,
            "classification accepted; bank resolution skipped",
            False,
        )
    )
    route_reason = bank.reason
    if not classification_enabled:
        route_reason = decision["reason"]
        if bank.status != "not_requested":
            route_reason += "; " + bank.reason
    return RouteResult(
        status=decision["status"],
        category=predicted_category,
        predicted_category=predicted_category,
        classification_enabled=classification_enabled,
        confidence=confidence,
        nearest_global_distance=distance,
        global_distance_threshold=decision["global_distance_threshold"],
        global_distance_ood=decision["global_distance_ood"],
        rejected=False,
        bank_status=bank.status,
        bank_path=bank.path,
        reason=route_reason,
        requires_rebuild=bank.requires_rebuild,
    )
