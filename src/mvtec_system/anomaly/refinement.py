"""Small, deterministic heatmap refinements validated on the screw branch."""

from __future__ import annotations

import cv2
import numpy as np


def propagate_scores(
    scores: np.ndarray,
    features: np.ndarray,
    top_k: int = 128,
    temperature: float = 0.1,
    eta: float = 1.0,
) -> np.ndarray:
    """Propagate detector scores over a sparse cosine-affinity graph."""

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    descriptors = np.asarray(features, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[0] != values.size:
        raise ValueError("features must be [patch, dimension] with one score per patch")
    if not np.isfinite(values).all() or not np.isfinite(descriptors).all():
        raise ValueError("scores and features must be finite")
    if top_k < 1 or temperature <= 0 or not 0 <= eta <= 1:
        raise ValueError("top_k must be positive, temperature positive, eta in [0, 1]")
    if values.size < 2 or eta == 0:
        return values.copy()
    normalized = descriptors / np.maximum(
        np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8
    )
    affinity = normalized @ normalized.T / temperature
    np.fill_diagonal(affinity, -np.inf)
    neighbors = min(int(top_k), values.size - 1)
    indices = np.argpartition(-affinity, neighbors, axis=1)[:, :neighbors]
    selected = np.take_along_axis(affinity, indices, axis=1)
    selected -= selected.max(axis=1, keepdims=True)
    weights = np.exp(selected)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    propagated = np.sum(weights * values[indices], axis=1)
    return ((1.0 - eta) * values + eta * propagated).astype(np.float32)


def refine_heatmap(heatmap: np.ndarray, method: str) -> np.ndarray:
    """Apply a deterministic continuous-map cleanup before thresholding."""

    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("heatmap must be a finite, non-empty 2-D array")
    size = {"heatmap_open3": 3, "heatmap_open5": 5, "heatmap_open7": 7, "heatmap_open9": 9}.get(method)
    if size is None:
        if method != "raw":
            raise ValueError(f"Unknown heatmap refinement method: {method}")
        return values
    return cv2.morphologyEx(values, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))


def postprocess_mask(heatmap: np.ndarray, threshold: float, method: str) -> np.ndarray:
    """Threshold one map and apply one explicit cleanup policy."""

    values = refine_heatmap(heatmap, method if method.startswith("heatmap_") else "raw")
    mask = (values >= float(threshold)).astype(np.uint8)
    if method.startswith("heatmap_") or method == "raw":
        return mask.astype(bool)
    if method == "open3":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    elif method == "open5":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    elif method == "open3_close5":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    elif method == "components":
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        minimum_area = max(9, int(mask.shape[0] * mask.shape[1] * 0.0005))
        keep = np.zeros_like(mask)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= minimum_area:
                keep[labels == index] = 1
        mask = keep
    else:
        raise ValueError(f"Unknown mask postprocess method: {method}")
    return mask.astype(bool)
