"""Reusable components for the continued MVTec AD system."""

from .dinov2 import DINOv2Config, DINOv2Features, SharedDINOv2, image_tensor, select_device
from .classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)
from .calibration import (
    UNKNOWN_PRODUCT_CATEGORY,
    calibrate_open_set_thresholds,
    nearest_global_distances,
    save_calibration,
)
from .router import BankResolution, RouteResult, resolve_category_bank, route_product
from .feature_bank import (
    FeatureBankResult,
    build_feature_bank,
    feature_bank_path,
    load_feature_bank,
    load_or_build_feature_bank,
    make_bank_metadata,
    save_feature_bank,
)
from .knn import distances_for_chunk, nearest_neighbor_scores
from .maps import make_anomaly_maps, plot_anomaly_examples
from .embeddings import (
    build_global_embedding_artifact,
    category_mapping,
    load_compatible_embedding_artifacts,
    make_embedding_metadata,
    read_classification_manifest,
    save_embedding_artifacts,
)
from .evaluation import (
    evaluate_product_classifier,
    save_classification_report,
    select_linear_classifier,
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
    "UNKNOWN_PRODUCT_CATEGORY",
    "calibrate_open_set_thresholds",
    "nearest_global_distances",
    "save_calibration",
    "BankResolution",
    "RouteResult",
    "resolve_category_bank",
    "route_product",
    "FeatureBankResult",
    "build_feature_bank",
    "feature_bank_path",
    "load_feature_bank",
    "load_or_build_feature_bank",
    "make_bank_metadata",
    "save_feature_bank",
    "distances_for_chunk",
    "nearest_neighbor_scores",
    "make_anomaly_maps",
    "plot_anomaly_examples",
    "build_global_embedding_artifact",
    "category_mapping",
    "load_compatible_embedding_artifacts",
    "make_embedding_metadata",
    "read_classification_manifest",
    "save_embedding_artifacts",
    "evaluate_product_classifier",
    "save_classification_report",
    "select_linear_classifier",
]
