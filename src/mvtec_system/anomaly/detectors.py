"""Detector registry with Exact KNN, PatchCore-style, and PaDiM-style models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.dinov2 import DINOv2Features, SharedDINOv2, image_tensor
from .knn import nearest_neighbor_scores
from .maps import make_anomaly_maps
from .multi_layer import (
    MultiLayerConfig,
    extract_fused_descriptors,
    fit_padim_statistics,
    fused_patch_features,
    local_neighborhood_aggregate,
    padim_scores,
    select_coreset,
)


@dataclass(frozen=True)
class AnomalyResult:
    """Common detector output consumed by routing and reporting."""

    image_score: float
    anomaly_map: np.ndarray
    patch_scores: np.ndarray
    metadata: dict[str, Any]


class AnomalyDetector:
    """Small common contract for all selectable detectors."""

    name = "base"

    def fit(self, category: str, normal_images: list[Path], encoder: SharedDINOv2, batch_size: int = 8):
        raise NotImplementedError

    def predict(self, features: DINOv2Features) -> AnomalyResult:
        raise NotImplementedError

    def predict_batch(self, features: DINOv2Features) -> list[AnomalyResult]:
        """Predict a batch; concrete detectors override this for vectorization."""

        batch_size = int(features.patch_embeddings.shape[0])
        return [
            self.predict(
                DINOv2Features(
                    features.global_embedding[index : index + 1],
                    features.patch_embeddings[index : index + 1],
                    {
                        layer: values[index : index + 1]
                        for layer, values in (features.layer_patch_embeddings or {}).items()
                    }
                    or None,
                )
            )
            for index in range(batch_size)
        ]

    def save(self, artifact_dir: str | Path) -> None:
        raise NotImplementedError

    def load(self, artifact_dir: str | Path) -> "AnomalyDetector":
        raise NotImplementedError

    def validate_artifacts(self) -> None:
        raise NotImplementedError


class ExactKNNDetector(AnomalyDetector):
    """Adapter for the existing final-layer exact kNN baseline."""

    name = "exact_knn"

    def __init__(self, k: int = 1):
        self.k = k
        self.bank: torch.Tensor | None = None
        self.patch_grid: tuple[int, int] | None = None
        self.padded_size: tuple[int, int] | None = None
        self.image_size: tuple[int, int] | None = None
        self.category: str | None = None
        self.metadata: dict[str, Any] = {}

    def fit(self, category: str, normal_images: list[Path], encoder: SharedDINOv2, batch_size: int = 8):
        if encoder.config.feature_layers:
            raise ValueError("Exact KNN uses the final-layer encoder configuration")
        if not normal_images or batch_size < 1:
            raise ValueError("normal_images must be non-empty and batch_size must be positive")
        chunks = []
        for start in range(0, len(normal_images), batch_size):
            images = torch.stack(
                [
                    image_tensor(path, encoder.config)
                    for path in normal_images[start : start + batch_size]
                ]
            )
            features = encoder.forward(images).patch_embeddings
            chunks.append(features.reshape(-1, features.shape[-1]).detach().cpu())
        self.bank = torch.cat(chunks, dim=0).float()
        self.category = category
        self.patch_grid = encoder.patch_grid
        self.padded_size = encoder.padded_size
        self.image_size = encoder.config.image_size
        self.metadata = {
            "category": category,
            "detector": self.name,
            "train_paths": [str(path) for path in normal_images],
            "embedding_dimension": int(self.bank.shape[1]),
        }
        return self

    def predict(self, features: DINOv2Features) -> AnomalyResult:
        self.validate_artifacts()
        patches = features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1])
        scores = nearest_neighbor_scores(patches, self.bank, k=self.k).reshape(1, -1)
        anomaly_map = make_anomaly_maps(scores, self.patch_grid, self.padded_size, self.image_size)[0]
        return AnomalyResult(float(scores.max().item()), anomaly_map, scores[0].detach().cpu().numpy(), {"detector": self.name, "k": self.k})

    def save(self, artifact_dir: str | Path) -> None:
        self.validate_artifacts()
        output = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": self.bank,
                "metadata": self.metadata,
                "patch_grid": self.patch_grid,
                "padded_size": self.padded_size,
                "image_size": self.image_size,
            },
            output / "feature_bank.pt",
        )

    def load(self, artifact_dir: str | Path) -> "ExactKNNDetector":
        payload = torch.load(Path(artifact_dir) / "feature_bank.pt", map_location="cpu", weights_only=True)
        self.bank = payload["features"].float()
        self.metadata = payload["metadata"]
        self.category = self.metadata.get("category")
        self.patch_grid = tuple(payload["patch_grid"])
        self.padded_size = tuple(payload["padded_size"])
        self.image_size = tuple(payload["image_size"])
        return self

    def validate_artifacts(self) -> None:
        if self.bank is None or self.bank.ndim != 2 or self.bank.shape[0] == 0:
            raise ValueError("Exact KNN bank is missing or invalid")
        if self.patch_grid is None or self.padded_size is None or self.image_size is None:
            raise ValueError("Exact KNN geometry is missing")


class PatchCoreDetector(AnomalyDetector):
    """Multi-layer, locally aggregated, representative-memory detector."""

    name = "patchcore"

    def __init__(self, config: MultiLayerConfig | None = None, k: int = 1):
        self.config = config or MultiLayerConfig()
        self.k = k
        self.coreset: torch.Tensor | None = None
        self.patch_grid: tuple[int, int] | None = None
        self.padded_size: tuple[int, int] | None = None
        self.image_size: tuple[int, int] | None = None
        self.category: str | None = None

    def fit(self, category: str, normal_images: list[Path], encoder: SharedDINOv2, batch_size: int = 8):
        descriptors = extract_fused_descriptors(
            normal_images,
            encoder,
            self.config.layers,
            batch_size,
            self.config.neighborhood_size,
        )
        self.coreset = select_coreset(
            descriptors.reshape(-1, descriptors.shape[-1]),
            self.config.coreset_ratio,
            self.config.coreset_max_points,
            self.config.seed,
        )
        self.category = category
        self.patch_grid = encoder.patch_grid
        self.padded_size = encoder.padded_size
        self.image_size = encoder.config.image_size
        return self

    def predict(self, features: DINOv2Features) -> AnomalyResult:
        self.validate_artifacts()
        descriptors = fused_patch_features(features, self.config.layers)
        descriptors = local_neighborhood_aggregate(
            descriptors, self.patch_grid, self.config.neighborhood_size
        )
        scores = nearest_neighbor_scores(
            descriptors.reshape(-1, descriptors.shape[-1]), self.coreset, k=self.k
        ).reshape(1, -1)
        anomaly_map = make_anomaly_maps(scores, self.patch_grid, self.padded_size, self.image_size)[0]
        metadata = {"detector": self.name, "k": self.k, **self.config.to_dict()}
        return AnomalyResult(float(scores.max().item()), anomaly_map, scores[0].detach().cpu().numpy(), metadata)

    def predict_batch(self, features: DINOv2Features) -> list[AnomalyResult]:
        self.validate_artifacts()
        descriptors = fused_patch_features(features, self.config.layers)
        descriptors = local_neighborhood_aggregate(
            descriptors, self.patch_grid, self.config.neighborhood_size
        )
        scores = nearest_neighbor_scores(
            descriptors.reshape(-1, descriptors.shape[-1]), self.coreset, k=self.k
        ).reshape(descriptors.shape[0], -1)
        anomaly_maps = make_anomaly_maps(
            scores, self.patch_grid, self.padded_size, self.image_size
        )
        metadata = {"detector": self.name, "k": self.k, **self.config.to_dict()}
        return [
            AnomalyResult(float(scores[index].max().item()), anomaly_maps[index], scores[index].detach().cpu().numpy(), metadata)
            for index in range(scores.shape[0])
        ]

    def save(self, artifact_dir: str | Path) -> None:
        self.validate_artifacts()
        output = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "coreset": self.coreset,
                "category": self.category,
                "config": self.config.to_dict(),
                "patch_grid": self.patch_grid,
                "padded_size": self.padded_size,
                "image_size": self.image_size,
            },
            output / "coreset.pt",
        )

    def load(self, artifact_dir: str | Path) -> "PatchCoreDetector":
        payload = torch.load(Path(artifact_dir) / "coreset.pt", map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError("PatchCore artifact is missing a valid detector configuration")
        self.coreset = payload["coreset"].float()
        self.config = MultiLayerConfig(**payload["config"])
        self.category = payload["category"]
        self.patch_grid = tuple(payload["patch_grid"])
        self.padded_size = tuple(payload["padded_size"])
        self.image_size = tuple(payload["image_size"])
        return self

    def validate_artifacts(self) -> None:
        if self.coreset is None or self.coreset.ndim != 2 or self.coreset.shape[0] == 0:
            raise ValueError("PatchCore coreset is missing or invalid")
        if self.patch_grid is None or self.padded_size is None or self.image_size is None:
            raise ValueError("PatchCore geometry is missing")


class PaDiMDetector(AnomalyDetector):
    """Multi-layer spatial Gaussian detector using Mahalanobis distance."""

    name = "padim"

    def __init__(self, config: MultiLayerConfig | None = None):
        self.config = config or MultiLayerConfig()
        self.means: torch.Tensor | None = None
        self.inverse_covariance: torch.Tensor | None = None
        self.projection: torch.Tensor | None = None
        self.patch_grid: tuple[int, int] | None = None
        self.padded_size: tuple[int, int] | None = None
        self.image_size: tuple[int, int] | None = None
        self.category: str | None = None

    def fit(self, category: str, normal_images: list[Path], encoder: SharedDINOv2, batch_size: int = 8):
        descriptors = extract_fused_descriptors(normal_images, encoder, self.config.layers, batch_size)
        self.means, self.inverse_covariance, self.projection = fit_padim_statistics(
            descriptors,
            self.config.reduced_dim,
            self.config.covariance_regularization,
            self.config.seed,
        )
        self.category = category
        self.patch_grid = encoder.patch_grid
        self.padded_size = encoder.padded_size
        self.image_size = encoder.config.image_size
        return self

    def predict(self, features: DINOv2Features) -> AnomalyResult:
        self.validate_artifacts()
        descriptors = fused_patch_features(features, self.config.layers)
        scores = padim_scores(
            descriptors, self.means, self.inverse_covariance, self.projection
        )
        anomaly_map = make_anomaly_maps(
            scores, self.patch_grid, self.padded_size, self.image_size
        )[0]
        metadata = {"detector": self.name, **self.config.to_dict()}
        return AnomalyResult(float(scores.max().item()), anomaly_map, scores[0].detach().cpu().numpy(), metadata)

    def predict_batch(self, features: DINOv2Features) -> list[AnomalyResult]:
        self.validate_artifacts()
        descriptors = fused_patch_features(features, self.config.layers)
        scores = padim_scores(
            descriptors, self.means, self.inverse_covariance, self.projection
        )
        anomaly_maps = make_anomaly_maps(
            scores, self.patch_grid, self.padded_size, self.image_size
        )
        metadata = {"detector": self.name, **self.config.to_dict()}
        return [
            AnomalyResult(float(scores[index].max().item()), anomaly_maps[index], scores[index].detach().cpu().numpy(), metadata)
            for index in range(scores.shape[0])
        ]

    def save(self, artifact_dir: str | Path) -> None:
        self.validate_artifacts()
        output = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "means": self.means,
                "inverse_covariance": self.inverse_covariance,
                "projection": self.projection,
                "category": self.category,
                "config": self.config.to_dict(),
                "patch_grid": self.patch_grid,
                "padded_size": self.padded_size,
                "image_size": self.image_size,
            },
            output / "statistics.pt",
        )

    def load(self, artifact_dir: str | Path) -> "PaDiMDetector":
        payload = torch.load(Path(artifact_dir) / "statistics.pt", map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError("PaDiM artifact is missing a valid detector configuration")
        self.means = payload["means"].float()
        self.inverse_covariance = payload["inverse_covariance"].float()
        self.projection = payload["projection"].float()
        self.config = MultiLayerConfig(**payload["config"])
        self.category = payload["category"]
        self.patch_grid = tuple(payload["patch_grid"])
        self.padded_size = tuple(payload["padded_size"])
        self.image_size = tuple(payload["image_size"])
        return self

    def validate_artifacts(self) -> None:
        if any(value is None for value in (self.means, self.inverse_covariance, self.projection)):
            raise ValueError("PaDiM statistics are incomplete")
        if self.patch_grid is None or self.padded_size is None or self.image_size is None:
            raise ValueError("PaDiM geometry is missing")


DETECTOR_REGISTRY = {
    ExactKNNDetector.name: ExactKNNDetector,
    PatchCoreDetector.name: PatchCoreDetector,
    PaDiMDetector.name: PaDiMDetector,
}


def create_detector(name: str, **kwargs: Any) -> AnomalyDetector:
    """Create one registered detector from configuration, without branching callers."""

    try:
        detector_type = DETECTOR_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown detector {name!r}; choose from {sorted(DETECTOR_REGISTRY)}") from error
    return detector_type(**kwargs)


def detector_names() -> tuple[str, ...]:
    return tuple(sorted(DETECTOR_REGISTRY))
