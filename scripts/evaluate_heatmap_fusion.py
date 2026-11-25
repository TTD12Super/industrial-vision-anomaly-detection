#!/usr/bin/env python3
"""Evaluate validation-selected fusions of the supervised refiner and raw AnomalyCLIP maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspection.anomalyclip import AnomalyClipSession  # noqa: E402
from inspection.config import load_product_config  # noqa: E402
from inspection.dataset import load_dataset_records  # noqa: E402
from inspection.dino_refinement import DinoAffinityRefiner  # noqa: E402
from inspection.evaluation import (  # noqa: E402
    _average_precision,
    _localizer_settings,
    _postprocessed_metrics,
    _relation_ground_truth,
    _relation_crop_rows,
    _supervised_refiner_settings,
    select_postprocess_threshold,
)
from inspection.geometry import oriented_crop  # noqa: E402
from inspection.supervised_refinement import SupervisedPatchMaskRefiner, patch_inputs  # noqa: E402


def robust_unit(values: np.ndarray) -> np.ndarray:
    """Map one heatmap to [0, 1] using robust per-image percentiles."""

    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def build_variants(raw: np.ndarray, refined: np.ndarray, crop: np.ndarray) -> dict[str, np.ndarray]:
    """Build deterministic score fusions without changing the upstream models."""

    raw01 = robust_unit(raw)
    refined01 = robust_unit(refined)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 3.0)
    contrast01 = robust_unit(np.abs(gray - smooth))
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient01 = robust_unit(cv2.magnitude(gradient_x, gradient_y))
    texture01 = np.maximum(contrast01, gradient01)
    return {
        "raw_anomalyclip": raw.astype(np.float32),
        "refined": refined.astype(np.float32),
        "refined_x_raw01": (refined01 * raw01).astype(np.float32),
        "geometric_mean01": np.sqrt(refined01 * raw01).astype(np.float32),
        "minimum01": np.minimum(refined01, raw01).astype(np.float32),
        "blend25_raw01": (0.75 * refined01 + 0.25 * raw01).astype(np.float32),
        "blend50_raw01": (0.50 * refined01 + 0.50 * raw01).astype(np.float32),
        "refined_x_texture01": (refined01 * texture01).astype(np.float32),
        "refined_soft_texture01": (refined01 * (0.25 + 0.75 * texture01)).astype(np.float32),
        "blend15_texture01": (0.85 * refined01 + 0.15 * texture01).astype(np.float32),
        "refined_x_contrast01": (refined01 * contrast01).astype(np.float32),
        "refined_x_gradient01": (refined01 * gradient01).astype(np.float32),
    }


def collect(config_path: Path, split: str) -> list[dict[str, object]]:
    config = load_product_config(config_path)
    settings = _supervised_refiner_settings(config)
    session = AnomalyClipSession(settings)
    dino = DinoAffinityRefiner(settings)
    localizer = config.model("localizer")
    refiner_path = config.project_path(localizer["supervised_refiner_path"])
    refiner = SupervisedPatchMaskRefiner.load(refiner_path)
    rows: list[dict[str, object]] = []
    for index, (object_row, crop) in enumerate(_relation_crop_rows(config, split), start=1):
        raw, _, _, _ = session.infer(crop)
        features, side = dino.patch_features(crop)
        refined = refiner.predict_map(
            patch_inputs(features, raw, side, refiner.feature_mode),
            side,
            crop.shape[:2],
        )
        truth = _relation_ground_truth(config, object_row, crop)
        raw = cv2.resize(raw, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        variants = build_variants(raw, refined, crop)
        for name, values in variants.items():
            rows.append({
                "object_id": object_row["object_id"],
                "label": object_row["quality_label"],
                "variant": name,
                "heatmap": values,
                "truth": truth,
            })
        if index % 10 == 0 or index == len(_relation_crop_rows(config, split)):
            print(f"{split}: processed {index}", flush=True)
    return rows


def group(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    return grouped


def run(args: argparse.Namespace) -> dict[str, object]:
    validation = group(collect(args.config, "validation"))
    test = group(collect(args.config, "test"))
    results: dict[str, object] = {}
    for name in validation:
        validation_rows = validation[name]
        test_rows = test[name]
        threshold, validation_metrics = select_postprocess_threshold(validation_rows, "raw", args.target_recall)
        results[name] = {
            "mask_threshold": float(threshold),
            "validation": {
                **validation_metrics,
                "pixel_average_precision": _average_precision(validation_rows, "raw"),
            },
            "test": {
                **_postprocessed_metrics(test_rows, threshold, "raw"),
                "pixel_average_precision": _average_precision(test_rows, "raw"),
            },
        }
    selected = max(
        results,
        key=lambda name: (
            results[name]["validation"]["pixel_average_precision"],
            results[name]["validation"]["pixel_precision"],
        ),
    )
    report = {
        "schema_version": "1.0",
        "experiment": "validation_selected_heatmap_fusion",
        "config": str(args.config),
        "target_validation_recall": args.target_recall,
        "selected_by_validation": selected,
        "variants": results,
        "warning": "MVTec official test images are supervised-development diagnostics, not an independent production estimate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/screw_full_anomaly_active_refiner.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/products/screw/screw-full-dev-011/heatmap_fusion.json"))
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in (0, 1]")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
