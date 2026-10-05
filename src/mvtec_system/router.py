"""Route product images to a compatible anomaly feature bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .calibration import UNKNOWN_PRODUCT_CATEGORY, nearest_global_distances
from .classifier import predict_linear_classifier


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

    category: str
    predicted_category: str
    confidence: float
    nearest_global_distance: float
    rejected: bool
    bank_status: str
    bank_path: str | None
    reason: str
    requires_rebuild: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    classifier: dict[str, Any],
    reference_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    calibration: dict[str, Any],
    bank_root: str | Path,
    encoder_metadata: dict[str, Any] | None = None,
) -> RouteResult:
    """Apply open-set rejection before resolving a category bank."""

    if global_embedding.ndim == 1:
        global_embedding = global_embedding.unsqueeze(0)
    if global_embedding.ndim != 2 or global_embedding.shape[0] != 1:
        raise ValueError("Router expects one global embedding.")
    reference_embeddings = reference_artifact.get("embeddings")
    if not isinstance(reference_embeddings, torch.Tensor):
        raise ValueError("Reference artifact must contain tensor embeddings.")
    inverse_mapping = _inverse_mapping(class_mapping)
    logits, probabilities = predict_linear_classifier(classifier, global_embedding)
    predicted_index = int(logits.argmax(dim=1).item())
    predicted_category = inverse_mapping[predicted_index]
    confidence = float(probabilities.max(dim=1).values.item())
    distance = float(nearest_global_distances(global_embedding, reference_embeddings)[0].item())
    confidence_threshold = float(calibration["confidence_threshold"])
    distance_threshold = float(calibration["nearest_global_distance_threshold"])
    rejection_reasons = []
    if confidence < confidence_threshold:
        rejection_reasons.append("confidence below threshold")
    if distance > distance_threshold:
        rejection_reasons.append("nearest global distance above threshold")
    if rejection_reasons:
        return RouteResult(
            category=UNKNOWN_PRODUCT_CATEGORY,
            predicted_category=predicted_category,
            confidence=confidence,
            nearest_global_distance=distance,
            rejected=True,
            bank_status="skipped",
            bank_path=None,
            reason="; ".join(rejection_reasons),
            requires_rebuild=False,
        )

    bank = resolve_category_bank(bank_root, predicted_category, encoder_metadata)
    return RouteResult(
        category=predicted_category,
        predicted_category=predicted_category,
        confidence=confidence,
        nearest_global_distance=distance,
        rejected=False,
        bank_status=bank.status,
        bank_path=bank.path,
        reason=bank.reason,
        requires_rebuild=bank.requires_rebuild,
    )
