"""Small shared helpers for the screw YOLO-OBB preparation scripts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def polygon_area(points: np.ndarray) -> float:
    """Return the absolute area of an Nx2 polygon."""

    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def order_corners(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Return four corners in a stable clockwise order."""

    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError(f"Expected four corners, got shape {pts.shape}")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    if polygon_signed_area(ordered) > 0:
        ordered = ordered[::-1]
    start = min(range(4), key=lambda i: (float(ordered[i, 1]), float(ordered[i, 0])))
    return np.roll(ordered, -start, axis=0)


def polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_to_obb(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Fit a minimum-area rotated rectangle to a segmentation polygon."""

    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3 or polygon_area(pts) <= 0:
        raise ValueError("Segmentation polygon is empty or degenerate")
    rect = cv2.minAreaRect(pts)
    return order_corners(cv2.boxPoints(rect))


def axis_box_to_corners(bbox: Iterable[float]) -> np.ndarray:
    """Convert COCO [x, y, width, height] to four pixel corners."""

    x, y, width, height = [float(v) for v in bbox[:4]]
    if width <= 0 or height <= 0:
        raise ValueError("Axis-aligned box must have positive width and height")
    return order_corners([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])


def mvtec_bbox_to_corners(bbox: Iterable[float]) -> np.ndarray:
    """Convert MVTec COCO-like [row, col, width, height, phi] to corners."""

    row, col, width, height, phi = [float(value) for value in bbox[:5]]
    if width <= 0 or height <= 0:
        raise ValueError("MVTec oriented box must have positive width and height")
    # MVTec uses a mathematical angle: +pi/2 points upward in image space.
    along = np.asarray([math.cos(phi), -math.sin(phi)], dtype=np.float32)
    across = np.asarray([math.sin(phi), math.cos(phi)], dtype=np.float32)
    center = np.asarray([col, row], dtype=np.float32)
    half_along = along * (width / 2.0)
    half_across = across * (height / 2.0)
    return order_corners(
        [
            center - half_along - half_across,
            center + half_along - half_across,
            center + half_along + half_across,
            center - half_along + half_across,
        ]
    )


def normalize_corners(points: Iterable[Iterable[float]], width: int, height: int) -> np.ndarray:
    """Normalize pixel corners to YOLO's [0, 1] coordinate range."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    pts = order_corners(points).astype(np.float64)
    pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)
    return pts


def _coerce_four_points(value: Any) -> np.ndarray | None:
    """Accept a flat 8-value list or a list of four [x, y] pairs."""

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 8:
        return order_corners(arr.reshape(4, 2))
    if arr.ndim == 2 and arr.shape == (4, 2):
        return order_corners(arr)
    return None


def annotation_to_corners(annotation: dict[str, Any], *, allow_axis_fallback: bool = False) -> np.ndarray:
    """Extract oriented corners from a COCO-like annotation.

    Segmentation is preferred.  An axis-aligned bbox is only converted when
    ``allow_axis_fallback`` is explicitly enabled, so a missing segmentation
    cannot silently become a fake oriented label.
    """

    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        polygons = []
        for segment in segmentation:
            try:
                arr = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if len(arr) >= 3:
                polygons.append(arr)
        if polygons:
            return polygon_to_obb(max(polygons, key=polygon_area))
    if isinstance(segmentation, dict):
        raise ValueError("RLE segmentation needs pycocotools; use axis fallback explicitly if acceptable")

    for key in ("obb", "oriented_bbox", "oriented_box", "corners", "points", "polygon"):
        points = _coerce_four_points(annotation.get(key))
        if points is not None:
            return points

    bbox = annotation.get("bbox")
    if bbox is not None:
        values = list(bbox)
        if len(values) == 5:
            return mvtec_bbox_to_corners(values)
        if len(values) >= 8:
            return order_corners(np.asarray(values[:8], dtype=np.float32).reshape(4, 2))
        if allow_axis_fallback:
            return axis_box_to_corners(values)
    raise ValueError("Annotation has no usable segmentation or oriented box")


def write_yolo_label(path: Path, objects: Iterable[np.ndarray], width: int, height: int) -> int:
    """Write one class-0 YOLO-OBB label per object."""

    lines = []
    for corners in objects:
        normalized = normalize_corners(corners, width, height)
        values = [0.0, *normalized.reshape(-1).tolist()]
        lines.append(" ".join(f"{value:.7f}" for value in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_screw_category(name: str) -> bool:
    normalized = name.lower().replace("_", " ").replace("-", " ")
    return "screw" in normalized and "nut" not in normalized


def mvtec_type_number(name: str) -> int | None:
    match = re.fullmatch(r"type_(\d+)", name.strip().lower())
    return int(match.group(1)) if match else None


def sanitize_filename(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe or "image.png"
