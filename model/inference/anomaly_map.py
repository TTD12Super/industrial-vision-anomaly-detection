"""Anomaly-map reconstruction + display-only visualization.

Reconstruction mirrors the notebook cell "Reconstruct the continuous anomaly
maps" exactly:

1. patch scores reshaped to the observed patch grid (19x19 for 256px / p14)
2. bilinear interpolation (``align_corners=False``) to the padded canvas
3. crop back to 256x256

No per-image rescaling or smoothing touches the scores used for evaluation.

Visualization helpers (heatmap / overlay) are display-only: colormap
normalization never feeds back into ``anomaly_score`` or ``anomaly_map``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import PROCESSED_SIZE

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def create_anomaly_maps(
    patch_scores: torch.Tensor,
    patch_grid: tuple[int, int],
    padded_size: tuple[int, int],
    output_size: tuple[int, int] = PROCESSED_SIZE,
) -> "object":
    """Build continuous anomaly maps of shape ``(N, H, W)`` (numpy array).

    Args:
        patch_scores: ``(num_images, num_patches)`` tensor of kNN distances.
        patch_grid: ``(grid_height, grid_width)``, e.g. ``(19, 19)``.
        padded_size: ``(height, width)`` of the padded model canvas,
            e.g. ``(266, 266)``.
        output_size: ``(height, width)`` to crop back to (default 256x256).
    """
    import torch as _torch
    import torch.nn.functional as _F

    with _torch.inference_mode():
        grid_height, grid_width = patch_grid
        if patch_scores.ndim != 2 or patch_scores.shape[1] != grid_height * grid_width:
            raise ValueError(
                f"patch_scores has shape {tuple(patch_scores.shape)} but patch grid "
                f"{patch_grid} expects {grid_height * grid_width} patches per image."
            )
        grids = patch_scores.reshape(-1, 1, grid_height, grid_width)
        padded_maps = _F.interpolate(
            grids, size=padded_size, mode="bilinear", align_corners=False
        )
        maps = padded_maps[:, 0, : output_size[0], : output_size[1]]
        if not _torch.isfinite(maps).all().item():
            raise ValueError("Anomaly maps contain invalid values.")
        return maps.numpy()


def create_anomaly_map(
    patch_scores: torch.Tensor,
    patch_grid: tuple[int, int],
    padded_size: tuple[int, int],
    output_size: tuple[int, int] = PROCESSED_SIZE,
) -> "object":
    """Single-image variant returning an ``(H, W)`` numpy anomaly map."""
    if patch_scores.ndim == 1:
        patch_scores = patch_scores.unsqueeze(0)
    elif patch_scores.ndim != 2 or patch_scores.shape[0] != 1:
        raise ValueError(
            "Single-image patch_scores must have shape (num_patches,) or (1, num_patches), "
            f"got {tuple(patch_scores.shape)}."
        )
    return create_anomaly_maps(patch_scores, patch_grid, padded_size, output_size)[0]


def anomaly_map_to_heatmap(anomaly_map: "object") -> "object":
    """Render an anomaly map as an RGB heatmap (uint8, ``inferno`` colormap).

    Display-only: normalized by the map maximum (epsilon-guarded).
    """
    try:
        import numpy as np
        from matplotlib import colormaps
    except ImportError as exc:
        raise ImportError("numpy and matplotlib are required for heatmaps.") from exc
    heatmap_source = np.asarray(anomaly_map, dtype=np.float64)
    if heatmap_source.ndim != 2:
        raise ValueError(
            f"anomaly_map must be 2D, got shape {heatmap_source.shape}."
        )
    maximum = max(float(heatmap_source.max(initial=0.0)), float(np.finfo(np.float32).eps))
    normalized = (heatmap_source / maximum).clip(0.0, 1.0)
    rgba = colormaps["inferno"](normalized)
    return (rgba[..., :3] * 255.0).round().astype(np.uint8)


def overlay_heatmap_on_image(
    rgb_image: "object", anomaly_map: "object", alpha: float = 0.5
) -> "object":
    """Alpha-blend a heatmap over the original RGB image (uint8 RGB output).

    Display-only; the returned overlay never affects inference scores.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for overlays.") from exc
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    base = np.asarray(rgb_image, dtype=np.float64)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError(f"rgb_image must have shape (H, W, 3), got {base.shape}.")
    heatmap = np.asarray(anomaly_map_to_heatmap(anomaly_map), dtype=np.float64)
    if heatmap.shape[:2] != base.shape[:2]:
        raise ValueError(
            f"Heatmap shape {heatmap.shape[:2]} does not match image shape {base.shape[:2]}."
        )
    blended = (1.0 - alpha) * base + alpha * heatmap
    return blended.round().clip(0, 255).astype(np.uint8)
