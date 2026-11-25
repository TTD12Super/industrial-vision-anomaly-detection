"""Experimental, validation-calibrated masked SegAD-lite evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backends import build_classifier, build_localizer
from .config import ProductConfig
from .dataset import load_dataset_records
from .geometry import oriented_crop
from .interfaces import BinaryQualityClassifier, Localizer


DEFAULT_INTERIOR_BORDER_FRACTION = 0.10


def derive_object_interior_mask(
    shape: tuple[int, int] | Sequence[int],
    border_fraction: float = DEFAULT_INTERIOR_BORDER_FRACTION,
) -> np.ndarray:
    """Return a conservative central support mask, never a ground-truth defect mask."""

    if len(shape) != 2:
        raise ValueError("object crop shape must contain height and width")
    height, width = (int(shape[0]), int(shape[1]))
    if height < 1 or width < 1:
        raise ValueError("object crop dimensions must be positive")
    if not 0.0 <= border_fraction < 0.5:
        raise ValueError("border_fraction must be in [0, 0.5)")
    border = int(np.ceil(min(height, width) * border_fraction))
    if 2 * border >= min(height, width):
        raise ValueError("border_fraction leaves no object interior")
    mask = np.zeros((height, width), dtype=bool)
    mask[border : height - border, border : width - border] = True
    return mask


def _compactness(binary_mask: np.ndarray) -> float:
    area = float(binary_mask.sum())
    if area == 0.0:
        return 0.0
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    return float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0


def masked_heatmap_stats(heatmap: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    """Summarize heatmap values only inside the deterministic support mask."""

    values = np.asarray(heatmap, dtype=np.float32)
    mask = np.asarray(interior_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("heatmap and interior mask must be matching two-dimensional arrays")
    if not np.isfinite(values).all() or not mask.any():
        raise ValueError("heatmap must be finite and interior mask must be non-empty")
    selected = values[mask]
    p90 = float(np.quantile(selected, 0.90))
    binary = (values >= p90) & mask
    return {
        "masked_mean": float(selected.mean()),
        "masked_std": float(selected.std()),
        "masked_p90": p90,
        "masked_max": float(selected.max()),
        "masked_area_fraction_at_p90": float(binary.sum() / mask.sum()),
        "masked_compactness_at_p90": _compactness(binary),
    }


def _select_threshold(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    candidates = np.unique(np.concatenate(([0.0, 1.0], np.asarray(scores, dtype=np.float64))))
    eligible = [
        float(candidate)
        for candidate in candidates
        if float(((scores >= candidate) & (labels == 1)).sum()) / int((labels == 1).sum()) >= target_recall
    ]
    if not eligible:
        raise ValueError("target defect recall is unattainable on validation data")
    return max(eligible)


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("classification metrics require matching GOOD/DEFECT labels")
    predicted = scores >= threshold
    defects = labels == 1
    goods = labels == 0
    return {
        "sample_count": int(labels.size),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "defect_recall_at_threshold": float((predicted & defects).sum() / defects.sum()),
        "good_false_positive_count": int((predicted & goods).sum()),
        "good_false_positive_rate": float((predicted & goods).sum() / goods.sum()),
    }


def fit_validation_logistic(
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    target_defect_recall: float = 0.90,
) -> dict[str, Any]:
    """Fit and threshold a calibrator using validation rows only."""

    features = np.asarray(validation_features, dtype=np.float32)
    labels = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("validation features and labels have incompatible shapes")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("validation calibration requires both GOOD and DEFECT rows")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(features, labels)
    scores = model.predict_proba(features)[:, 1]
    threshold = _select_threshold(labels, scores, target_defect_recall)
    return {
        "model": model,
        "threshold": threshold,
        "validation_metrics": classification_metrics(labels, scores, threshold),
    }


def _pixel_metrics(samples: Sequence[dict[str, Any]], threshold: float) -> dict[str, float | int | None]:
    scores: list[np.ndarray] = []
    truth_values: list[np.ndarray] = []
    true_positive = false_positive = false_negative = 0
    for sample in samples:
        support = sample["interior_mask"]
        truth = sample["truth"] & support
        predicted = (sample["heatmap"] >= threshold) & support
        values = sample["heatmap"][support]
        scores.append(values)
        truth_values.append(truth[support])
        true_positive += int(np.logical_and(predicted, truth).sum())
        false_positive += int(np.logical_and(predicted, ~truth).sum())
        false_negative += int(np.logical_and(~predicted, truth).sum())
    flat_truth = np.concatenate(truth_values).astype(np.int64)
    flat_scores = np.concatenate(scores)
    denominator = true_positive + false_positive + false_negative
    return {
        "pixel_average_precision": float(average_precision_score(flat_truth, flat_scores)) if flat_truth.any() else None,
        "pixel_precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        "pixel_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "pixel_dice": 2 * true_positive / (2 * true_positive + false_positive + false_negative) if denominator else 1.0,
        "pixel_iou": true_positive / denominator if denominator else 1.0,
        "support_pixel_count": int(flat_truth.size),
    }


def _select_pixel_threshold(samples: Sequence[dict[str, Any]], target_recall: float) -> float:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    values = np.concatenate([sample["heatmap"][sample["interior_mask"]].ravel() for sample in samples])
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 101)))
    measured = [(float(candidate), _pixel_metrics(samples, float(candidate))) for candidate in candidates]
    eligible = [item for item in measured if item[1]["pixel_recall"] >= target_recall]
    if not eligible:
        return max(measured, key=lambda item: (item[1]["pixel_recall"], item[1]["pixel_precision"]))[0]
    return min(eligible, key=lambda item: (1.0 - item[1]["pixel_precision"], -item[1]["pixel_recall"]))[0]


def _resolve_paths(config: ProductConfig, settings: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    resolved = dict(settings)
    for key in keys:
        if resolved.get(key):
            resolved[key] = str(config.project_path(resolved[key]))
    return resolved


def _samples(
    config: ProductConfig,
    split: str,
    localizer: Localizer,
    classifier: BinaryQualityClassifier | None,
    border_fraction: float,
) -> list[dict[str, Any]]:
    images, objects = load_dataset_records(config)
    image_by_id = {row["image_id"]: row for row in images if row["split"] == split}
    samples: list[dict[str, Any]] = []
    for object_row in objects:
        image_row = image_by_id.get(object_row["image_id"])
        if image_row is None:
            continue
        image_path = config.dataset_path(Path(image_row["relative_path"]))
        source = cv2.imread(str(image_path))
        if source is None:
            raise ValueError(f"Cannot read image: {image_path}")
        crop = oriented_crop(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), object_row["obb_xyxyxyxy"])
        localization = localizer.localize(crop.image)
        if localization.status != "completed" or localization.heatmap is None:
            raise RuntimeError(f"Localization failed for {object_row['object_id']}: {localization.reason}")
        heatmap = np.asarray(localization.heatmap, dtype=np.float32)
        if heatmap.shape != crop.image.shape[:2]:
            heatmap = cv2.resize(heatmap, (crop.image.shape[1], crop.image.shape[0]), interpolation=cv2.INTER_LINEAR)
        interior_mask = derive_object_interior_mask(heatmap.shape, border_fraction)
        p_defect = None if classifier is None else float(classifier.classify(crop.image).p_defect)
        if p_defect is not None and not 0.0 <= p_defect <= 1.0:
            raise ValueError("classifier p_defect must be in [0, 1]")
        stats = masked_heatmap_stats(heatmap, interior_mask)
        truth = None
        mask_relative_path = object_row.get("mask_relative_path")
        if mask_relative_path:
            mask_path = config.dataset_path(Path(mask_relative_path))
            source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if source_mask is None:
                raise ValueError(f"Cannot read ground-truth mask: {mask_path}")
            truth = cv2.warpPerspective(
                (source_mask > 0).astype(np.uint8),
                crop.forward_matrix,
                (crop.image.shape[1], crop.image.shape[0]),
                flags=cv2.INTER_NEAREST,
            ).astype(bool)
        elif object_row["quality_label"] == "GOOD":
            truth = np.zeros(heatmap.shape, dtype=bool)
        feature_values = [stats["masked_mean"], stats["masked_p90"], stats["masked_max"], stats["masked_std"]]
        if p_defect is not None:
            feature_values.append(p_defect)
        samples.append(
            {
                "label": 1 if object_row["quality_label"] == "DEFECT" else 0,
                "features": feature_values,
                "heatmap": heatmap,
                "interior_mask": interior_mask,
                "truth": truth,
                "stats": stats,
            }
        )
    if not samples:
        raise ValueError(f"No object samples found for split {split}")
    return samples


def _split_report(samples: Sequence[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    stats = [sample["stats"] for sample in samples]
    return {
        "classification": classification_metrics(np.asarray([sample["label"] for sample in samples]), scores, threshold),
        "masked_heatmap_stats": {
            key: float(np.mean([item[key] for item in stats]))
            for key in stats[0]
        },
        "sample_count": len(samples),
    }


def evaluate_segad_lite(
    config: ProductConfig,
    report_path: str | Path,
    target_defect_recall: float = 0.90,
    pixel_target_recall: float = 0.90,
    interior_border_fraction: float = DEFAULT_INTERIOR_BORDER_FRACTION,
    include_p_defect: bool = False,
    localizer: Localizer | None = None,
    classifier: BinaryQualityClassifier | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit on validation object crops and evaluate the locked historical test split."""

    if localizer is None:
        localizer = build_localizer(
            _resolve_paths(config, config.model("localizer"), ("runner", "repository", "checkpoint", "clip_cache", "calibration_path"))
        )
    if include_p_defect and classifier is None:
        classifier = build_classifier(
            _resolve_paths(config, config.model("classifier"), ("artifact_dir", "encoder_artifact_root"))
        )
    validation = _samples(config, "validation", localizer, classifier if include_p_defect else None, interior_border_fraction)
    test = _samples(config, "test", localizer, classifier if include_p_defect else None, interior_border_fraction)
    validation_features = np.asarray([sample["features"] for sample in validation], dtype=np.float32)
    test_features = np.asarray([sample["features"] for sample in test], dtype=np.float32)
    validation_labels = np.asarray([sample["label"] for sample in validation], dtype=np.int64)
    test_labels = np.asarray([sample["label"] for sample in test], dtype=np.int64)
    calibration = fit_validation_logistic(validation_features, validation_labels, target_defect_recall)
    validation_scores = calibration["model"].predict_proba(validation_features)[:, 1]
    test_scores = calibration["model"].predict_proba(test_features)[:, 1]
    validation_report = _split_report(validation, validation_scores, calibration["threshold"])
    test_report = _split_report(test, test_scores, calibration["threshold"])
    pixel_available = all(sample["truth"] is not None for sample in validation + test)
    warnings = [
        "Calibrator and all thresholds are fitted on validation only; test is historical and locked.",
        "The central-inset mask is an analysis support mask, not a ground-truth defect mask.",
    ]
    if pixel_available:
        pixel_threshold = _select_pixel_threshold(validation, pixel_target_recall)
        validation_report["pixel_metrics"] = _pixel_metrics(validation, pixel_threshold)
        test_report["pixel_metrics"] = _pixel_metrics(test, pixel_threshold)
    else:
        pixel_threshold = None
        validation_report["pixel_metrics"] = None
        test_report["pixel_metrics"] = None
        warnings.append("Pixel metrics omitted because complete validation and test ground-truth masks are unavailable.")
    report = {
        "schema_version": "1.0",
        "experiment": "masked_segad_lite",
        "product_key": config.product_key,
        "calibration": {
            "fit_split": "validation",
            "target_defect_recall": target_defect_recall,
            "threshold": calibration["threshold"],
            "feature_names": ["masked_mean", "masked_p90", "masked_max", "masked_std"] + (["p_defect"] if include_p_defect else []),
            "classifier": "standard_scaler_logistic",
            "pixel_threshold": pixel_threshold,
            "pixel_target_recall": pixel_target_recall if pixel_available else None,
        },
        "mask_strategy": {
            "name": "central_inset_rectangle",
            "border_fraction": interior_border_fraction,
            "purpose": "conservative support mask for heatmap statistics; not a defect annotation",
        },
        "validation": validation_report,
        "test": test_report,
        "provenance": {
            "config": str(config.source_path),
            "image_manifest": str(config.dataset_path(config.image_manifest)),
            "object_manifest": str(config.dataset_path(config.object_manifest)),
            "validation_split": "validation",
            "test_split": "test",
            "test_locked": True,
            "detector_used": False,
            "p_defect_included": include_p_defect,
        },
        "warnings": warnings,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_INTERIOR_BORDER_FRACTION",
    "classification_metrics",
    "derive_object_interior_mask",
    "evaluate_segad_lite",
    "fit_validation_logistic",
    "masked_heatmap_stats",
]
