"""Create readable classification-evaluation charts from a JSON report.

Run from the repository root:

    DAenv/bin/python reports/classification_evaluation/visualize_classification.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REPORT = Path("artifacts/product_classifier/v1/validation_metrics.json")
DEFAULT_OUTPUT = Path("reports/classification_evaluation")


def read_report(path: Path) -> dict:
    """Read the validation report created by Task 006."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one chart and release its memory."""

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_overall_metrics(report: dict, output_dir: Path) -> Path:
    """Plot the five required aggregate metrics."""

    metrics = report["metrics"]
    names = [
        "Accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
        "Top-3 accuracy",
    ]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "top_3_accuracy"]
    values = [metrics[key] for key in keys]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, values, color="#2f6f9f")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Product classification validation metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    path = output_dir / "overall_metrics.png"
    save_figure(figure, path)
    return path


def plot_per_class_metrics(report: dict, output_dir: Path) -> Path:
    """Plot validation F1 and support for every product category."""

    per_class = report["per_class"]
    categories = list(per_class)
    f1_scores = [per_class[category]["f1"] for category in categories]
    supports = [per_class[category]["support"] for category in categories]
    positions = np.arange(len(categories))

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(positions, f1_scores, color="#4c956c")
    axis.set_yticks(positions, labels=categories)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("F1 score")
    axis.set_title("Validation F1 score by product category")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value, support in zip(bars, f1_scores, supports):
        axis.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f} (n={support})", va="center")
    path = output_dir / "per_class_f1.png"
    save_figure(figure, path)
    return path


def plot_confusion_matrix(report: dict, output_dir: Path) -> Path:
    """Plot the category confusion matrix with readable labels."""

    mapping = report["class_mapping"]
    categories = [category for category, _ in sorted(mapping.items(), key=lambda item: item[1])]
    matrix = np.asarray(report["confusion_matrix"])

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of validation images")
    axis.set_xticks(range(len(categories)), labels=categories, rotation=45, ha="right")
    axis.set_yticks(range(len(categories)), labels=categories)
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("True category")
    axis.set_title("Validation confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    path = output_dir / "confusion_matrix.png"
    save_figure(figure, path)
    return path


def plot_model_selection(report: dict, output_dir: Path) -> Path:
    """Plot validation macro-F1 for every candidate C value."""

    selection = report.get("model_selection", {})
    candidates = selection.get("candidates", [])
    if not candidates:
        return output_dir / "model_selection.png"
    c_values = [str(candidate["config"]["C"]) for candidate in candidates]
    scores = [candidate["metrics"]["macro_f1"] for candidate in candidates]

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(c_values, scores, color="#d98e04")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Logistic-regression C")
    axis.set_ylabel("Validation macro F1")
    axis.set_title("Validation model selection")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, score in zip(bars, scores):
        axis.text(bar.get_x() + bar.get_width() / 2, score + 0.015, f"{score:.3f}", ha="center")
    path = output_dir / "model_selection.png"
    save_figure(figure, path)
    return path


def create_charts(report_path: Path, output_dir: Path) -> list[Path]:
    """Create all requested charts and return their paths."""

    report = read_report(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_overall_metrics(report, output_dir),
        plot_per_class_metrics(report, output_dir),
        plot_confusion_matrix(report, output_dir),
        plot_model_selection(report, output_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in create_charts(args.report, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
