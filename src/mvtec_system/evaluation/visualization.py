"""Reusable dataset and classification report visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(figure: plt.Figure, path: str | Path) -> Path:
    """Save a chart and release its matplotlib resources."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.tight_layout()
    except RuntimeError:
        # Figures created with Matplotlib's constrained layout already have a
        # layout engine; switching them to tight_layout is not supported.
        pass
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_dataset_distribution(
    validation_report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot train, test, and ground-truth counts for every category."""

    categories = validation_report.get("categories", [])
    counts = validation_report.get("counts", {})
    if not categories or not isinstance(counts, dict):
        raise ValueError("Validation report must contain categories and counts.")
    train = [counts[category]["train"] for category in categories]
    test = [counts[category]["test"] for category in categories]
    masks = [counts[category]["ground_truth"] for category in categories]
    positions = np.arange(len(categories))
    width = 0.26
    figure, axis = plt.subplots(figsize=(max(12, len(categories) * 0.8), 6))
    axis.bar(positions - width, train, width, label="Train images", color="#2563eb")
    axis.bar(positions, test, width, label="Test images", color="#f59e0b")
    axis.bar(positions + width, masks, width, label="Ground-truth masks", color="#16a34a")
    axis.set_title("MVTec AD split sizes by category")
    axis.set_xlabel("Category")
    axis.set_ylabel("Number of files")
    axis.set_xticks(positions, categories, rotation=45, ha="right")
    axis.legend()
    return save_figure(figure, output_path)
