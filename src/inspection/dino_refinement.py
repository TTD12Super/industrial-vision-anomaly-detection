"""DINO-guided, score-preserving refinement for CLIP anomaly maps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2


def propagate_scores(scores: np.ndarray, features: np.ndarray, top_k: int = 128, temperature: float = 0.1, eta: float = 1.0) -> np.ndarray:
    """Propagate anomaly scores over a sparse cosine-affinity graph."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != scores.size:
        raise ValueError("features must be [patch, dimension] with one score per patch")
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("scores and features must be finite")
    if not 0.0 <= eta <= 1.0 or temperature <= 0.0:
        raise ValueError("eta must be in [0, 1] and temperature must be positive")
    if scores.size < 2:
        return scores.copy()
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    affinity = normalized @ normalized.T / temperature
    np.fill_diagonal(affinity, -np.inf)
    neighbors = min(max(1, int(top_k)), scores.size - 1)
    indices = np.argpartition(-affinity, neighbors, axis=1)[:, :neighbors]
    selected = np.take_along_axis(affinity, indices, axis=1)
    selected -= np.max(selected, axis=1, keepdims=True)
    weights = np.exp(selected)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    propagated = np.sum(weights * scores[indices], axis=1)
    return ((1.0 - eta) * scores + eta * propagated).astype(np.float32)


class DinoAffinityRefiner:
    """Use frozen DINOv2 only as a structural graph, never as the anomaly scorer."""

    def __init__(self, settings: dict[str, object]) -> None:
        artifact_root = Path(str(settings.get("artifact_root", "artifacts/production/mvtec_ad")))
        device = settings.get("device")
        configured_layers = settings.get("dino_feature_layers", ())
        self.feature_layers = tuple(int(layer) for layer in configured_layers)
        image_size = int(settings.get("dino_image_size", 256))
        if image_size < 14:
            raise ValueError("dino_image_size must be at least one DINO patch")
        self.image_size = image_size
        self.encoder = SharedDINOv2(
            DINOv2Config(image_size=(image_size, image_size), feature_layers=self.feature_layers),
            artifact_root=artifact_root,
            device=str(device) if device else None,
        )
        self.top_k = int(settings.get("top_k", 128))
        self.temperature = float(settings.get("temperature", 0.1))
        self.eta = float(settings.get("eta", 1.0))

    def refine(self, heatmap: np.ndarray, crop: np.ndarray) -> np.ndarray:
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("crop must be RGB [height, width, 3]")
        features, side = self.patch_features(crop)
        token_scores = cv2.resize(np.asarray(heatmap, dtype=np.float32), (side, side), interpolation=cv2.INTER_AREA).reshape(-1)
        refined = propagate_scores(token_scores, features, self.top_k, self.temperature, self.eta).reshape(side, side)
        return cv2.resize(refined, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def patch_features(self, crop: np.ndarray) -> tuple[np.ndarray, int]:
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("crop must be RGB [height, width, 3]")
        resized = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
        pixels = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.encoder.config.mean).view(3, 1, 1)
        std = torch.tensor(self.encoder.config.std).view(3, 1, 1)
        encoded = self.encoder.forward(((pixels - mean) / std).unsqueeze(0))
        if self.feature_layers:
            if encoded.layer_patch_embeddings is None:
                raise ValueError("DINO intermediate patch features are unavailable")
            features = np.concatenate(
                [encoded.layer_patch_embeddings[layer][0].cpu().numpy() for layer in self.feature_layers],
                axis=1,
            )
        else:
            features = encoded.patch_embeddings[0].cpu().numpy()
        side = int(round(np.sqrt(features.shape[0])))
        if side * side != features.shape[0]:
            raise ValueError(f"DINO patch count is not square: {features.shape[0]}")
        return features.astype(np.float32), side
