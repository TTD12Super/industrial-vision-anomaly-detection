"""Oriented crops and reversible image-coordinate projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


def order_corners(corners: Iterable[Iterable[float]]) -> np.ndarray:
    """Return exactly four clockwise corners beginning at the top-left corner."""

    points = np.asarray(list(corners), dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("OBB corners must be four finite [x, y] points")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    signed_area = float(np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1)) - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1)))
    if signed_area > 0:
        ordered = ordered[::-1]
    start = min(range(4), key=lambda index: (float(ordered[index, 1]), float(ordered[index, 0])))
    return np.roll(ordered, -start, axis=0)


@dataclass(frozen=True)
class OrientedCrop:
    image: np.ndarray
    corners: np.ndarray
    forward_matrix: np.ndarray
    inverse_matrix: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        return height, width


def oriented_crop(image: np.ndarray, corners: Iterable[Iterable[float]]) -> OrientedCrop:
    """Warp an oriented object into an upright RGB crop and retain its inverse map."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [height, width, 3]")
    ordered = order_corners(corners)
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    width = max(1, int(round(max(top, bottom))))
    height = max(1, int(round(max(left, right))))
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    forward = cv2.getPerspectiveTransform(ordered, destination)
    inverse = cv2.getPerspectiveTransform(destination, ordered)
    crop = cv2.warpPerspective(image, forward, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return OrientedCrop(crop, ordered, forward, inverse)


def project_crop_map(crop_map: np.ndarray, crop: OrientedCrop, image_shape: tuple[int, int]) -> np.ndarray:
    """Project a heatmap or mask from crop coordinates back to its source image."""

    if crop_map.ndim != 2:
        raise ValueError("crop_map must be two-dimensional")
    image_height, image_width = image_shape
    if image_height < 1 or image_width < 1:
        raise ValueError("image_shape must be positive")
    interpolation = cv2.INTER_NEAREST if crop_map.dtype == np.bool_ else cv2.INTER_LINEAR
    source = crop_map.astype(np.uint8) if crop_map.dtype == np.bool_ else crop_map.astype(np.float32)
    projected = cv2.warpPerspective(source, crop.inverse_matrix, (image_width, image_height), flags=interpolation)
    return projected.astype(bool) if crop_map.dtype == np.bool_ else projected
