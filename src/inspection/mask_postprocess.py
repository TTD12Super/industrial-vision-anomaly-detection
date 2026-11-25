"""Small deterministic cleanup policies for validated heatmap masks."""

from __future__ import annotations

import cv2
import numpy as np


def refine_heatmap(heatmap: np.ndarray, method: str) -> np.ndarray:
    """Apply an optional continuous-map cleanup before thresholding/ranking."""

    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("heatmap must be a finite 2-D array")
    size = {"heatmap_open3": 3, "heatmap_open5": 5, "heatmap_open7": 7, "heatmap_open9": 9}.get(method)
    if size is None:
        return values
    return cv2.morphologyEx(values, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))


def postprocess_mask(heatmap: np.ndarray, threshold: float, method: str) -> np.ndarray:
    """Turn a heatmap into a binary mask using one explicit cleanup policy."""

    heatmap = refine_heatmap(heatmap, method)
    if method.startswith("heatmap_"):
        method = "raw"
    mask = (heatmap >= threshold).astype(np.uint8)
    if method == "raw":
        return mask.astype(bool)
    if method.startswith("border"):
        fraction = float(method.removeprefix("border")) / 100.0
        border_y = int(mask.shape[0] * fraction)
        border_x = int(mask.shape[1] * fraction)
        mask[:border_y] = 0
        mask[-border_y:] = 0
        mask[:, :border_x] = 0
        mask[:, -border_x:] = 0
        method = "open3"
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
    elif method != "raw":
        raise ValueError(f"Unknown heatmap postprocess method: {method}")
    return mask.astype(bool)
