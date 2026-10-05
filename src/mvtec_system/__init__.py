"""Reusable components for the continued MVTec AD system."""

from .anomaly import pipeline as pipeline
from .classification import router as router
from .core.dinov2 import DINOv2Config, DINOv2Features, SharedDINOv2, image_tensor, select_device
from .classification.classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)
from .classification.calibration import (
    DEFAULT_GLOBAL_DISTANCE_DEFECT_PERCENTILE,
    UNKNOWN_PRODUCT_CATEGORY,
    calibrate_open_set_thresholds,
    nearest_global_distances,
    save_calibration,
)
from .core.artifacts import ArtifactLayout, make_artifact_layout
from .classification.router import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    KNOWN_PRODUCT_CATEGORY,
    BankResolution,
    RouteResult,
    resolve_category_bank,
    route_product,
    routing_decision,
    routing_rejection_reasons,
)
from .classification.product_test_evaluation import (
    discover_mvtec_test_images,
    evaluate_product_classification_test_split,
    product_category_from_mvtec_path,
)
from .anomaly.feature_bank import (
    FeatureBankResult,
    build_feature_bank,
    feature_bank_path,
    load_feature_bank,
    load_or_build_feature_bank,
    make_bank_metadata,
    save_feature_bank,
)
from .anomaly.knn import distances_for_chunk, nearest_neighbor_scores
from .anomaly.maps import make_anomaly_maps, plot_anomaly_examples
from .anomaly.pipeline import infer_image
from .evaluation.performance import available_backends, file_size_mb, measure_callable, memory_usage_mb
from .evaluation.reproduction import (
    REFERENCE_MACRO,
    reproduce_saved_baseline,
    save_reproduction_report,
)
from .anomaly.thresholds import (
    DEFECT_LABEL,
    GOOD_LABEL,
    calibrate_anomaly_thresholds,
    calibrate_category_image_thresholds,
    calibrate_category_pixel_thresholds,
    calibrate_quantile_threshold,
    binary_anomaly_masks,
    classify_image_score,
    image_scores_from_patch_scores,
    save_anomaly_thresholds,
)
from .classification.embeddings import (
    build_global_embedding_artifact,
    category_mapping,
    load_compatible_embedding_artifacts,
    make_embedding_metadata,
    read_classification_manifest,
    save_embedding_artifacts,
)
from .evaluation.evaluation import (
    compare_routing_modes,
    evaluate_anomaly_detection,
    evaluate_product_classifier,
    save_classification_report,
    select_linear_classifier,
)
from .data.preprocessing import (
    create_classification_manifest,
    dataset_manifest,
    discover_dataset_root,
    image_files,
    prepare_mvtec_dataset,
    preprocess_dataset,
    validate_processed_dataset,
    validate_raw_dataset,
)
from .evaluation.visualization import plot_dataset_distribution
from .production.pipeline import ProductionConfig, run_production

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
    "DEFAULT_GLOBAL_DISTANCE_DEFECT_PERCENTILE",
    "ArtifactLayout",
    "make_artifact_layout",
    "calibrate_open_set_thresholds",
    "nearest_global_distances",
    "save_calibration",
    "BankResolution",
    "RouteResult",
    "DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD",
    "KNOWN_PRODUCT_CATEGORY",
    "resolve_category_bank",
    "route_product",
    "routing_decision",
    "routing_rejection_reasons",
    "discover_mvtec_test_images",
    "evaluate_product_classification_test_split",
    "product_category_from_mvtec_path",
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
    "infer_image",
    "available_backends",
    "file_size_mb",
    "measure_callable",
    "memory_usage_mb",
    "REFERENCE_MACRO",
    "reproduce_saved_baseline",
    "save_reproduction_report",
    "DEFECT_LABEL",
    "GOOD_LABEL",
    "calibrate_anomaly_thresholds",
    "calibrate_category_image_thresholds",
    "calibrate_category_pixel_thresholds",
    "calibrate_quantile_threshold",
    "binary_anomaly_masks",
    "classify_image_score",
    "image_scores_from_patch_scores",
    "save_anomaly_thresholds",
    "build_global_embedding_artifact",
    "category_mapping",
    "load_compatible_embedding_artifacts",
    "make_embedding_metadata",
    "read_classification_manifest",
    "save_embedding_artifacts",
    "evaluate_product_classifier",
    "evaluate_anomaly_detection",
    "compare_routing_modes",
    "save_classification_report",
    "select_linear_classifier",
    "dataset_manifest",
    "discover_dataset_root",
    "create_classification_manifest",
    "image_files",
    "prepare_mvtec_dataset",
    "preprocess_dataset",
    "validate_processed_dataset",
    "validate_raw_dataset",
    "plot_dataset_distribution",
    "ProductionConfig",
    "run_production",
]
