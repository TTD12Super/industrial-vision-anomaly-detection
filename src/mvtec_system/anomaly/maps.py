"""Anomaly-map interpolation and comparison heatmaps."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def make_anomaly_maps(
    patch_scores: torch.Tensor | np.ndarray,
    patch_grid: tuple[int, int] | list[int],
    padded_size: tuple[int, int] | list[int],
    image_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Interpolate patch scores to the padded canvas, then crop to image size."""

    scores = torch.as_tensor(patch_scores, dtype=torch.float32)
    if scores.ndim != 2:
        raise ValueError("patch_scores must have shape [images, patches].")
    patch_grid = tuple(int(value) for value in patch_grid)
    padded_size = tuple(int(value) for value in padded_size)
    image_size = tuple(int(value) for value in image_size)
    if min(*patch_grid, *padded_size, *image_size) < 1:
        raise ValueError("Map dimensions must be positive.")
    patch_count = patch_grid[0] * patch_grid[1]
    if scores.shape[1] != patch_count:
        raise ValueError(f"Expected {patch_count} patch scores, got {scores.shape[1]}.")
    if image_size[0] > padded_size[0] or image_size[1] > padded_size[1]:
        raise ValueError("Image size cannot exceed padded map size.")
    if not torch.isfinite(scores).all().item():
        raise ValueError("Patch scores contain invalid values.")
    grids = scores.reshape(-1, 1, patch_grid[0], patch_grid[1])
    padded_maps = F.interpolate(
        grids,
        size=padded_size,
        mode="bilinear",
        align_corners=False,
    )
    maps = padded_maps[:, 0, : image_size[0], : image_size[1]]
    if not torch.isfinite(maps).all().item():
        raise ValueError("Anomaly maps contain invalid values.")
    return maps.numpy()


def plot_anomaly_examples(
    category: str,
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    anomaly_maps: np.ndarray,
    image_scores: Sequence[float],
    is_anomaly: Sequence[bool],
    k_neighbors: int = 1,
    examples_per_class: int = 1,
    map_title: str = "Continuous anomaly map",
) -> plt.Figure:
    """Create the same image/mask/heatmap/overlay comparison as the notebook."""

    if examples_per_class < 1:
        raise ValueError("examples_per_class must be positive.")
    if not (
        len(images)
        == len(masks)
        == len(anomaly_maps)
        == len(image_scores)
        == len(is_anomaly)
    ):
        raise ValueError("Images, masks, maps, scores, and labels must have equal lengths.")
    good_indices = [index for index, value in enumerate(is_anomaly) if not value]
    defect_indices = [index for index, value in enumerate(is_anomaly) if value]
    selected = good_indices[:examples_per_class] + defect_indices[:examples_per_class]
    if not selected:
        raise ValueError("At least one example is required.")
    figure, axes = plt.subplots(
        len(selected),
        4,
        figsize=(13, 3.6 * len(selected)),
        squeeze=False,
        layout="constrained",
    )
    maximum = max(float(np.max(anomaly_maps)), np.finfo(np.float32).eps)
    heatmap = None
    for row, index in enumerate(selected):
        axes[row, 0].imshow(images[index])
        axes[row, 0].set_title(f"{'defect' if is_anomaly[index] else 'good'}: score {image_scores[index]:.3f}")
        axes[row, 1].imshow(masks[index], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
        heatmap = axes[row, 2].imshow(
            anomaly_maps[index], cmap="inferno", vmin=0, vmax=maximum
        )
        axes[row, 2].set_title(map_title)
        axes[row, 3].imshow(images[index])
        axes[row, 3].imshow(
            anomaly_maps[index], cmap="inferno", alpha=0.5, vmin=0, vmax=maximum
        )
        axes[row, 3].set_title("Overlay")
        for axis in axes[row]:
            axis.axis("off")
    figure.colorbar(heatmap, ax=axes[:, 2:], label="Euclidean distance", shrink=0.8)
    figure.suptitle(f"{category}: frozen DINOv2 + {k_neighbors}-NN")
    return figure
