"""Reusable components for the continued MVTec AD system."""

from .dinov2 import DINOv2Config, DINOv2Features, SharedDINOv2, image_tensor, select_device
from .classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)
from .embeddings import (
    build_global_embedding_artifact,
    category_mapping,
    load_compatible_embedding_artifacts,
    make_embedding_metadata,
    read_classification_manifest,
    save_embedding_artifacts,
)

__all__ = [
    "DINOv2Config",
    "DINOv2Features",
    "SharedDINOv2",
    "image_tensor",
    "select_device",
    "LinearClassifierConfig",
    "fit_linear_classifier",
    "predict_linear_classifier",
    "save_linear_classifier",
    "build_global_embedding_artifact",
    "category_mapping",
    "load_compatible_embedding_artifacts",
    "make_embedding_metadata",
    "read_classification_manifest",
    "save_embedding_artifacts",
]
