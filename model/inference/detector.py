"""End-to-end anomaly detector for one MVTec category.

Pipeline (same order as the notebook, one category at a time):
1. preprocessing (RGB -> 256x256 -> normalized tensor)
2. frozen DINOv2 patch-feature extraction (+ L2 normalization)
3. load the category's ``train/good`` feature bank
4. exact-kNN patch scores
5. image anomaly score = maximum patch score (continuous, no threshold)
6. anomaly map (19x19 grid -> bilinear -> crop 256x256)
7. display-only heatmap / overlay (never affect scores)

The result dict intentionally contains **no** ``is_anomaly`` / Normal-Anomaly
label: the baseline has no calibrated threshold (notebook: "continuous
scores only; no test-label calibration").
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from .anomaly_map import (
    anomaly_map_to_heatmap,
    create_anomaly_map,
    overlay_heatmap_on_image,
)
from .config import (
    FEATURE_KEY,
    IMAGE_AGGREGATION,
    K_NEIGHBORS,
    MODEL_NAME,
    MODEL_REVISION,
    PROCESSED_SIZE,
    default_bank_path,
    get_bank_device,
    validate_category,
)
from .feature_bank import load_feature_bank
from .feature_extractor import DINOv2FeatureExtractor
from .knn import nearest_neighbor_scores
from .preprocessing import ImageLike, load_rgb_image

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Single-category detector backed by a prebuilt normal feature bank."""

    def __init__(
        self,
        category: str,
        bank_path: str | Path | None = None,
        bank_dir: str | Path | None = None,
        extractor: DINOv2FeatureExtractor | None = None,
        device: torch.device | str | None = None,
        torch_hub_dir: str | None = None,
    ) -> None:
        """Args:
        category: MVTec category (one of the 15 supported).
        bank_path: explicit ``feature_bank.pt`` file. When omitted,
            resolved as ``<bank_dir>/<category>/feature_bank.pt``.
        bank_dir: root holding per-category bank folders.
        extractor: reuse an existing extractor, otherwise one is created.
        device: compute device for the backbone (default: auto).
        torch_hub_dir: optional torch hub cache directory for DINOv2 weights.
        """
        self.category = validate_category(category)
        if bank_path is None:
            if bank_dir is None:
                raise ValueError(
                    "Provide bank_path or bank_dir so the category feature bank "
                    "can be located."
                )
            bank_path = default_bank_path(category, bank_dir)
        self.bank_path = Path(bank_path)

        self.extractor = extractor or DINOv2FeatureExtractor(
            device=device, torch_hub_dir=torch_hub_dir
        )
        expected_metadata_keys = {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "feature_key": FEATURE_KEY,
            "image_size": list(PROCESSED_SIZE),
            "patch_grid": [self.extractor.patch_grid[0], self.extractor.patch_grid[1]],
        }
        bank, metadata = load_feature_bank(self.bank_path)
        for key, expected in expected_metadata_keys.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Feature bank {self.bank_path} has {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}; refusing to mix incompatible settings."
                )
        if metadata.get("embedding_dimension") != self.extractor.embedding_dimension:
            # Embedding dim is discovered on first forward; run one cheap check
            # only when the extractor already knows its dimension.
            if self.extractor.embedding_dimension is not None:
                raise ValueError(
                    f"Feature bank embedding dim {metadata.get('embedding_dimension')} "
                    "does not match the extractor; refusing to mix banks."
                )
        self.metadata = metadata
        self.bank = bank.to(get_bank_device(self.extractor.device))
        logger.info(
            "AnomalyDetector ready (category=%s, bank=%s, device=%s).",
            self.category,
            tuple(self.bank.shape),
            self.extractor.device,
        )

    def predict(self, image: ImageLike, with_visualization: bool = True) -> dict[str, Any]:
        """Run inference on one image.

        Returns a dict with ``category``, ``anomaly_score`` (float, max patch
        distance), ``patch_scores`` (361-vector), ``anomaly_map`` (256x256),
        plus ``heatmap``/``overlay`` uint8 RGB arrays when requested, and
        provenance metadata (patch grid, model, timing, bank shape).
        """
        started = time.perf_counter()
        rgb = load_rgb_image(image)
        patch_features = self.extractor.extract(image)  # (P, D)
        if self.extractor.embedding_dimension is None:
            raise RuntimeError("Extractor did not report an embedding dimension.")
        patch_scores = nearest_neighbor_scores(
            patch_features, self.bank, k=K_NEIGHBORS
        )
        anomaly_score = float(patch_scores.max().item())  # IMAGE_AGGREGATION == "max"
        anomaly_map = create_anomaly_map(
            patch_scores,
            self.extractor.patch_grid,
            self.extractor.padded_size,
            PROCESSED_SIZE,
        )
        result: dict[str, Any] = {
            "category": self.category,
            "anomaly_score": anomaly_score,
            "aggregation": IMAGE_AGGREGATION,
            "patch_scores": patch_scores.numpy(),
            "anomaly_map": anomaly_map,
            "patch_grid": self.extractor.patch_grid,
            "padded_size": self.extractor.padded_size,
            "embedding_dimension": self.extractor.embedding_dimension,
            "bank_shape": tuple(self.bank.shape),
            "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
            "inference_seconds": time.perf_counter() - started,
        }
        if with_visualization:
            heatmap = anomaly_map_to_heatmap(anomaly_map)
            result["heatmap"] = heatmap
            try:
                import numpy as np

                result["overlay"] = overlay_heatmap_on_image(
                    np.array(rgb, copy=True), anomaly_map
                )
            except ImportError as exc:
                raise ImportError("numpy is required for overlays.") from exc
        logger.info(
            "Predicted category=%s score=%.4f in %.2fs.",
            self.category,
            anomaly_score,
            result["inference_seconds"],
        )
        return result
