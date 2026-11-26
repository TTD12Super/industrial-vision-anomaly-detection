"""DINOv2 + exact-kNN anomaly-detection inference.

Reference implementation: ``notebooks/02_first_model.ipynb``.

This package uses lazy loading (PEP 562): importing the package or a
dependency-free submodule (``config``, ``preprocessing``) never requires
torch/Pillow/numpy. Heavy modules import those libraries on first use.
"""

from __future__ import annotations

_LAZY_EXPORTS = {
    "AnomalyDetector": "detector",
    "BANK_CHUNK_SIZE": "config",
    "CATEGORIES": "config",
    "DINOv2FeatureExtractor": "feature_extractor",
    "FEATURE_KEY": "config",
    "IMAGE_SIZE": "config",
    "IMAGENET_MEAN": "config",
    "IMAGENET_STD": "config",
    "INFERENCE_BATCH_SIZE": "config",
    "K_NEIGHBORS": "config",
    "L2_NORMALIZE_FEATURES": "config",
    "MODEL_NAME": "config",
    "MODEL_REPOSITORY": "config",
    "MODEL_REVISION": "config",
    "PROCESSED_SIZE": "config",
    "QUERY_CHUNK_SIZE": "config",
    "anomaly_map_to_heatmap": "anomaly_map",
    "bank_metadata": "feature_bank",
    "build_feature_bank": "feature_bank",
    "compute_padding": "preprocessing",
    "compute_patch_grid": "preprocessing",
    "create_anomaly_map": "anomaly_map",
    "create_anomaly_maps": "anomaly_map",
    "default_bank_path": "config",
    "distances_for_chunk": "knn",
    "get_bank_device": "config",
    "get_device": "config",
    "image_tensor": "preprocessing",
    "list_train_images": "feature_bank",
    "load_feature_bank": "feature_bank",
    "load_rgb_image": "preprocessing",
    "nearest_neighbor_scores": "knn",
    "normalize_features": "feature_extractor",
    "overlay_heatmap_on_image": "anomaly_map",
    "pad_for_patch_embedding": "preprocessing",
    "prepare_image": "preprocessing",
    "read_processed_rgb": "preprocessing",
    "save_feature_bank": "feature_bank",
    "validate_category": "config",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")


def __dir__():
    return sorted(__all__)
