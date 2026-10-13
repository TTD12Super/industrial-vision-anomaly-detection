"""Small, deterministic post-processing helpers for localization outputs."""

from __future__ import annotations

import cv2
import numpy as np


def refine_binary_mask(
    mask: np.ndarray,
    min_component_area: int = 9,
    opening_kernel_size: int = 3,
) -> np.ndarray:
    """Remove isolated mask noise without changing the image-level decision."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("mask must be a non-empty 2D array")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")
    if opening_kernel_size < 1 or opening_kernel_size % 2 == 0:
        raise ValueError("opening_kernel_size must be a positive odd number")

    refined = values.astype(np.uint8)
    if opening_kernel_size > 1:
        kernel = np.ones((opening_kernel_size, opening_kernel_size), dtype=np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined, connectivity=8
    )
    keep = np.zeros_like(refined)
    for component in range(1, component_count):
        if stats[component, cv2.CC_STAT_AREA] >= min_component_area:
            keep[labels == component] = 1
    return keep.astype(bool)


def gate_heatmap(heatmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep the raw heatmap only inside a refined localization mask."""

    values = np.asarray(heatmap, dtype=np.float32)
    gate = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or gate.ndim != 2 or values.shape != gate.shape:
        raise ValueError("heatmap and mask must be 2D arrays with the same shape")
    if not np.isfinite(values).all():
        raise ValueError("heatmap must contain only finite values")
    return np.where(gate, values, 0.0).astype(np.float32)
