"""Recompute saved MVTec baseline metrics with the refactored evaluator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .evaluation import evaluate_anomaly_detection


REFERENCE_MACRO = {
    "image_auroc": 0.9803247058806648,
    "image_ap": 0.987077244579432,
    "pixel_auroc": 0.9612823597274186,
    "pixel_ap": 0.5162392690459406,
}


def _mask_for_row(
    processed_root: Path,
    category: str,
    defect_type: str,
    image_path: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Read one processed ground-truth mask, or zeros for a good image."""

    if defect_type == "good":
        return np.zeros(shape, dtype=np.uint8)
    stem = Path(image_path).stem
    path = processed_root / category / "ground_truth" / defect_type / f"{stem}_mask.png"
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask is missing: {path}")
    with Image.open(path) as mask:
        array = np.asarray(mask.convert("L"))
    if tuple(array.shape) != shape:
        raise ValueError(f"Mask shape {array.shape} does not match anomaly map {shape}.")
    return (array > 0).astype(np.uint8)


def _category_report(
    category_dir: Path,
    processed_root: Path,
) -> dict[str, Any]:
    """Recompute one category's ranking metrics from saved scores and maps."""

    category = category_dir.name
    with (category_dir / "image_scores.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No image score rows found for category: {category}")
    archive = np.load(category_dir / "anomaly_maps.npz")
    patch_scores = np.asarray(archive["patch_scores"], dtype=np.float64)
    anomaly_maps = np.asarray(archive["anomaly_maps"], dtype=np.float64)
    if patch_scores.ndim != 2 or anomaly_maps.ndim != 3:
        raise ValueError(f"Unexpected saved score/map shapes for category: {category}")
    if len(rows) != len(patch_scores) or len(rows) != len(anomaly_maps):
        raise ValueError(f"CSV and saved score counts differ for category: {category}")

    image_scores = np.asarray([float(row["anomaly_score"]) for row in rows])
    image_labels = np.asarray([int(row["is_anomaly"]) for row in rows])
    saved_image_scores = patch_scores.max(axis=1)
    score_consistency = float(np.max(np.abs(image_scores - saved_image_scores)))
    masks = np.stack(
        [
            _mask_for_row(
                processed_root,
                category,
                row["defect_type"],
                row["image_path"],
                tuple(anomaly_maps[index].shape),
            )
            for index, row in enumerate(rows)
        ]
    )
    report = evaluate_anomaly_detection(
        image_scores,
        image_labels,
        anomaly_maps.reshape(-1),
        masks.reshape(-1),
    )
    return {
        "category": category,
        "sample_count": len(rows),
        "score_consistency_max_abs_diff": score_consistency,
        "image_auroc": report["image"]["ranking"]["auroc"],
        "image_ap": report["image"]["ranking"]["average_precision"],
        "pixel_auroc": report["pixel"]["ranking"]["auroc"],
        "pixel_ap": report["pixel"]["ranking"]["average_precision"],
    }


def reproduce_saved_baseline(
    run_root: str | Path = "artifacts/dinov2_knn/final_run",
    processed_root: str | Path = "datasets/mvtec_ad_processed",
) -> dict[str, Any]:
    """Recompute all available category metrics and compare macro references."""

    run_root = Path(run_root)
    processed_root = Path(processed_root)
    category_dirs = sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and (path / "image_scores.csv").is_file()
        and (path / "anomaly_maps.npz").is_file()
    )
    if not category_dirs:
        raise FileNotFoundError(f"No saved category runs found under {run_root}")
    categories = [_category_report(path, processed_root) for path in category_dirs]
    metric_names = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap")
    observed_macro = {
        metric: float(np.mean([category[metric] for category in categories]))
        for metric in metric_names
    }
    deltas = {
        metric: observed_macro[metric] - REFERENCE_MACRO[metric] for metric in metric_names
    }
    return {
        "run_root": str(run_root),
        "processed_root": str(processed_root),
        "category_count": len(categories),
        "categories": categories,
        "observed_macro": observed_macro,
        "reference_macro": REFERENCE_MACRO,
        "macro_delta": deltas,
        "max_abs_macro_delta": max(abs(value) for value in deltas.values()),
        "all_image_score_reconstructions_within_tolerance": all(
            category["score_consistency_max_abs_diff"] <= 1e-6 for category in categories
        ),
    }


def save_reproduction_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write the baseline reproduction report as JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
