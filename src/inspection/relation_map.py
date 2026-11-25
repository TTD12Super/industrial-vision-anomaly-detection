"""Optional contextual patch features for an AnomalyCLIP-compatible backend."""

from __future__ import annotations

import numpy as np
import cv2


INCOMPATIBLE_BACKEND = "INCOMPATIBLE_BACKEND"


def contextual_features(attention: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Apply the documented convention M = A @ F where A[i,j] attends j to i."""

    attention = np.asarray(attention, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError("attention must be a finite square [patch, patch] matrix")
    if features.ndim != 2 or features.shape[0] != attention.shape[0]:
        raise ValueError("features must be [patch, dimension] with matching patch order")
    if not np.isfinite(attention).all() or not np.isfinite(features).all():
        raise ValueError("attention and features must be finite")
    return attention @ features


def relation_scores(attention: np.ndarray, features: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Return position-indexed V2 L2 deviation scores from a GOOD prototype."""

    relation = contextual_features(attention, features)
    prototype = np.asarray(prototype, dtype=np.float32)
    if prototype.shape != relation.shape or not np.isfinite(prototype).all():
        raise ValueError("prototype must be finite and match the relation feature shape")
    return np.linalg.norm(relation - prototype, axis=1)


def scores_to_map(scores: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Reshape a square patch score vector and upsample it to image dimensions."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    side = int(round(float(np.sqrt(scores.size))))
    if side * side != scores.size:
        raise ValueError("patch scores must form a square token grid")
    height, width = output_size
    if height < 1 or width < 1:
        raise ValueError("output_size must be positive")
    return cv2.resize(scores.reshape(side, side), (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("map must be a finite two-dimensional array")
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def fuse_maps(baseline: np.ndarray, relation: np.ndarray, beta: float = 1.0) -> np.ndarray:
    if beta <= 0:
        raise ValueError("beta must be positive")
    if baseline.shape != relation.shape:
        raise ValueError("baseline and relation maps must have the same shape")
    return beta * normalize_map(baseline) + normalize_map(relation)
