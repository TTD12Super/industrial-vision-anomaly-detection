from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from screw_yolo_obb import (  # noqa: E402
    annotation_to_corners,
    axis_box_to_corners,
    mvtec_bbox_to_corners,
    normalize_corners,
)


def test_axis_box_is_four_normalized_corners() -> None:
    corners = axis_box_to_corners([10, 20, 30, 40])
    normalized = normalize_corners(corners, 100, 100)
    assert normalized.shape == (4, 2)
    assert np.all((normalized >= 0) & (normalized <= 1))
    assert np.isclose(normalized[:, 0].min(), 0.1)
    assert np.isclose(normalized[:, 0].max(), 0.4)


def test_segmentation_is_preferred_over_axis_fallback() -> None:
    annotation = {
        "bbox": [10, 20, 30, 40],
        "segmentation": [[20, 20, 40, 20, 40, 25, 20, 25]],
    }
    corners = annotation_to_corners(annotation)
    assert np.isclose(corners[:, 1].max() - corners[:, 1].min(), 5)


def test_axis_fallback_is_explicit() -> None:
    annotation = {"bbox": [10, 20, 30, 40]}
    try:
        annotation_to_corners(annotation)
    except ValueError:
        pass
    else:
        raise AssertionError("axis fallback must not be implicit")
    assert annotation_to_corners(annotation, allow_axis_fallback=True).shape == (4, 2)


def test_mvtec_five_value_bbox_uses_row_col_and_angle() -> None:
    corners = mvtec_bbox_to_corners([100, 200, 40, 20, 0])
    assert np.isclose(corners[:, 0].min(), 180)
    assert np.isclose(corners[:, 0].max(), 220)
    assert np.isclose(corners[:, 1].min(), 90)
    assert np.isclose(corners[:, 1].max(), 110)
