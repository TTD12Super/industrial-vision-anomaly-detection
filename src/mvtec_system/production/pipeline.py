"""One-command automation for an arbitrary MVTec-style dataset."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import torch

from ..anomaly.detectors import AnomalyDetector, AnomalyResult, create_detector
from ..anomaly.feature_bank import FeatureBankResult, load_or_build_feature_bank
from ..anomaly.knn import nearest_neighbor_scores
from ..anomaly.maps import make_anomaly_maps, plot_anomaly_examples
from ..anomaly.multi_layer import MultiLayerConfig
from ..anomaly.thresholds import (
    binary_anomaly_masks,
    calibrate_category_image_thresholds,
    calibrate_category_pixel_thresholds,
    classify_image_score,
)
from ..classification.calibration import calibrate_open_set_thresholds, save_calibration
from ..classification.classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    save_linear_classifier,
)
from ..classification.embeddings import (
    build_global_embedding_artifact,
    category_mapping,
    make_embedding_metadata,
    read_classification_manifest,
    save_embedding_artifacts,
)
from ..classification.product_test_evaluation import discover_mvtec_test_images
from ..classification.router import route_product
from ..core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor
from ..data.preprocessing import (
    category_directories,
    create_classification_manifest,
    discover_dataset_root,
    image_files,
    prepare_mvtec_dataset,
    validate_raw_dataset,
)
from ..evaluation.visualization import plot_dataset_distribution, save_figure
from ..evaluation.performance import synchronize


@dataclass(frozen=True)
class ProductionConfig:
    """All run settings; dataset and category names are discovered at runtime."""

    datasets_root: Path = Path("datasets")
    dataset_root: Path | None = None
    processed_root: Path | None = None
    report_root: Path = Path("reports/production")
    artifact_root: Path = Path("artifacts/production")
    image_size: int = 256
    validation_fraction: float = 0.30
    seed: int = 42
    batch_size: int = 8
    k: int = 1
    detector: str = "exact_knn"  # Used only by offline independent comparison.
    orchestration: str = "exact_first"
    detector_layers: tuple[int, ...] = (5, 8, 11)
    neighborhood_size: int = 3
    coreset_ratio: float = 0.10
    coreset_max_points: int = 50_000
    padim_reduced_dim: int = 100
    covariance_regularization: float = 0.01
    target_acceptance_rate: float = 0.99
    classification_confidence_threshold: float = 0.50
    device: str | None = None
    artifact_namespace: str | None = None
    force: bool = False


def _error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _test_manifest_info(records: list[dict[str, Any]], processed_root: Path) -> dict[str, Any]:
    """Hash the ordered test identities so comparisons cannot rely on count alone."""

    root = processed_root.resolve()
    entries = []
    for record in records:
        path = Path(record["image_path"]).resolve()
        try:
            image_path = str(path.relative_to(root))
        except ValueError:
            image_path = str(path)
        entries.append(
            {
                "image_path": image_path,
                "true_category": record["true_category"],
                "defect_type": record["defect_type"],
                "is_defect": bool(record["is_defect"]),
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "count": len(entries),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def infer_exact_first(
    exact_runner: Callable[[], dict[str, Any]],
    patchcore_runner: Callable[[], np.ndarray],
    padim_runner: Callable[[], np.ndarray],
) -> dict[str, Any]:
    """Run Exact kNN first, then separate PatchCore and PaDiM branches on DEFECT."""

    try:
        exact = dict(exact_runner())
    except Exception as error:
        return {
            "category": None,
            "status": "decision_unavailable",
            "rejected": False,
            "decision": None,
            "decision_source": None,
            "decision_status": "unavailable",
            "primary_error": _error_message(error),
            "anomaly_score": None,
            "threshold": None,
            "label": None,
            "is_defect": None,
            "decision_is_defect": None,
            "image_score": None,
            "image_threshold": None,
            "heatmap": None,
            "binary_mask": None,
            "anomaly_map": None,
            "patchcore": {"status": "not_run", "heatmap": None, "error": None},
            "padim": {"status": "not_run", "mask": None, "error": None},
            "localization_status": {"patchcore": "not_run", "padim": "not_run"},
        }

    result = dict(exact)
    result.setdefault("decision", None)
    result.setdefault("decision_source", None)
    result.setdefault("decision_status", "unavailable")
    result.setdefault("decision_is_defect", None)
    result.setdefault("primary_error", None)
    # Exact kNN's internal map is not a production localization result.
    result["heatmap"] = None
    result["binary_mask"] = None
    result["anomaly_map"] = None
    result["patchcore"] = {"status": "not_run", "heatmap": None, "error": None}
    result["padim"] = {"status": "not_run", "mask": None, "error": None}
    result["localization_status"] = {"patchcore": "not_run", "padim": "not_run"}

    if result.get("decision") is None and result.get("is_defect") is not None:
        result["decision"] = "DEFECT" if bool(result["is_defect"]) else "GOOD"
    decision = result.get("decision")
    if decision not in {"GOOD", "DEFECT"}:
        result["decision_status"] = "unavailable"
        if result.get("bank_status") in {"missing", "incompatible"}:
            result["primary_error"] = result.get("reason") or (
                f"Exact kNN artifact is {result['bank_status']}"
            )
        result.setdefault("decision_source", None)
        result.setdefault("decision_is_defect", None)
        return result

    result["decision_source"] = "exact_knn"
    result["decision_status"] = "available"
    result["decision_is_defect"] = decision == "DEFECT"
    result["image_score"] = result.get("anomaly_score")
    result["image_threshold"] = result.get("threshold")
    result["label"] = decision
    result["is_defect"] = decision == "DEFECT"

    if decision == "GOOD":
        return result

    try:
        heatmap = np.asarray(patchcore_runner(), dtype=np.float32)
        if heatmap.ndim != 2 or heatmap.size == 0 or not np.isfinite(heatmap).all():
            raise ValueError("PatchCore heatmap must be a finite non-empty 2D array")
        result["heatmap"] = heatmap
        result["anomaly_map"] = heatmap
        result["patchcore"] = {"status": "available", "heatmap": heatmap, "error": None}
        result["localization_status"]["patchcore"] = "available"
    except Exception as error:
        message = _error_message(error)
        result["patchcore"] = {"status": "failed", "heatmap": None, "error": message}
        result["localization_status"]["patchcore"] = "failed"

    try:
        mask = np.asarray(padim_runner(), dtype=bool)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("PaDiM mask must be a non-empty 2D array")
        status = "available" if bool(mask.any()) else "empty"
        result["binary_mask"] = mask
        result["padim"] = {"status": status, "mask": mask, "error": None}
        result["localization_status"]["padim"] = status
    except Exception as error:
        message = _error_message(error)
        result["padim"] = {"status": "failed", "mask": None, "error": message}
        result["localization_status"]["padim"] = "failed"

    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    size = max(8.0, len(labels) * 0.55)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="True category",
        title=title,
    )
    axis.tick_params(axis="x", rotation=45)
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
                fontsize=7,
            )
    save_figure(figure, path)


def _score_with_bank(
    image_path: Path,
    encoder: SharedDINOv2,
    bank: torch.Tensor,
    k: int,
) -> tuple[float, np.ndarray]:
    """Run one frozen forward pass and exact kNN against one category bank."""

    with torch.inference_mode():
        features = encoder.forward(torch.stack([image_tensor(image_path, encoder.config)]))
    patches = features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1])
    patch_scores = nearest_neighbor_scores(patches, bank, k=k).reshape(1, -1)
    score = float(patch_scores.max().item())
    anomaly_map = make_anomaly_maps(
        patch_scores,
        encoder.patch_grid,
        encoder.padded_size,
        encoder.config.image_size,
    )[0]
    return score, anomaly_map


def _calibrate_anomaly_thresholds(
    rows: list[dict[str, str]],
    processed_root: Path,
    banks: dict[str, FeatureBankResult],
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_scores: dict[str, list[float]] = {}
    pixel_scores: dict[str, list[float]] = {}
    for category in sorted(banks):
        selected = [row for row in rows if row["category"] == category and row["split"] == "validation"]
        if not selected:
            selected = [row for row in rows if row["category"] == category and row["split"] == "train"][:1]
        scores: list[float] = []
        pixels: list[np.ndarray] = []
        for row in selected:
            score, anomaly_map = _score_with_bank(
                processed_root / row["image_path"], encoder, banks[category].features, config.k
            )
            scores.append(score)
            pixels.append(anomaly_map.reshape(-1))
        image_scores[category] = scores
        pixel_scores[category] = np.concatenate(pixels).tolist()
    return (
        calibrate_category_image_thresholds(image_scores, config.target_acceptance_rate),
        calibrate_category_pixel_thresholds(pixel_scores, config.target_acceptance_rate),
    )


def _detector_config(config: ProductionConfig) -> MultiLayerConfig:
    return MultiLayerConfig(
        layers=config.detector_layers,
        neighborhood_size=config.neighborhood_size,
        coreset_ratio=config.coreset_ratio,
        coreset_max_points=config.coreset_max_points,
        reduced_dim=config.padim_reduced_dim,
        covariance_regularization=config.covariance_regularization,
        seed=config.seed,
    )


def _score_with_detector(
    image_path: Path,
    encoder: SharedDINOv2,
    detector: AnomalyDetector,
) -> AnomalyResult:
    with torch.inference_mode():
        features = encoder.forward(torch.stack([image_tensor(image_path, encoder.config)]))
    return detector.predict(features)


def _calibrate_detector_thresholds(
    rows: list[dict[str, str]],
    processed_root: Path,
    detectors: dict[str, AnomalyDetector],
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_scores: dict[str, list[float]] = {}
    pixel_scores: dict[str, list[float]] = {}
    for category in sorted(detectors):
        selected = [
            row for row in rows
            if row["category"] == category and row["split"] == "validation"
        ]
        if not selected:
            selected = [
                row for row in rows
                if row["category"] == category and row["split"] == "train"
            ][:1]
        scores: list[float] = []
        pixels: list[np.ndarray] = []
        for row in selected:
            result = _score_with_detector(
                processed_root / row["image_path"], encoder, detectors[category]
            )
            scores.append(result.image_score)
            pixels.append(result.anomaly_map.reshape(-1))
        image_scores[category] = scores
        pixel_scores[category] = np.concatenate(pixels).tolist()
    return (
        calibrate_category_image_thresholds(image_scores, config.target_acceptance_rate),
        calibrate_category_pixel_thresholds(pixel_scores, config.target_acceptance_rate),
    )


def _calibrate_detector_pixel_thresholds(
    rows: list[dict[str, str]],
    processed_root: Path,
    detectors: dict[str, AnomalyDetector],
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> dict[str, Any]:
    """Calibrate only the pixel threshold needed by a localization detector."""

    pixel_scores: dict[str, list[float]] = {}
    for category in sorted(detectors):
        selected = [
            row for row in rows
            if row["category"] == category and row["split"] == "validation"
        ]
        if not selected:
            selected = [
                row for row in rows
                if row["category"] == category and row["split"] == "train"
            ][:1]
        maps = [
            _score_with_detector(
                processed_root / row["image_path"], encoder, detectors[category]
            ).anomaly_map.reshape(-1)
            for row in selected
        ]
        pixel_scores[category] = np.concatenate(maps).tolist()
    return calibrate_category_pixel_thresholds(pixel_scores, config.target_acceptance_rate)


def _infer_with_detector(
    item: dict[str, Any],
    encoder: SharedDINOv2,
    classifier: dict[str, Any] | None,
    reference_artifact: dict[str, Any] | None,
    mapping: dict[str, int],
    calibration: dict[str, Any],
    detectors: dict[str, AnomalyDetector],
    image_thresholds: dict[str, Any],
    pixel_thresholds: dict[str, Any],
    config: ProductionConfig,
) -> dict[str, Any]:
    """Run one detector for the offline independent-comparison mode."""

    with torch.inference_mode():
        features = encoder.forward(
            torch.stack([image_tensor(item["image_path"], encoder.config)])
        )
    route = route_product(
        features.global_embedding,
        classifier,
        reference_artifact,
        mapping,
        calibration,
        config.artifact_root,
        encoder.metadata(),
        config.classification_confidence_threshold,
        resolve_bank=False,
    )
    result = route.to_dict()
    result.update(
        {
            "image_path": item["image_path"],
            "anomaly_score": None,
            "threshold": None,
            "label": None,
            "patch_scores": None,
            "anomaly_map": None,
            "binary_mask": None,
        }
    )
    if route.rejected or route.category not in detectors:
        return result
    detector_result = detectors[route.category].predict(features)
    result["anomaly_score"] = detector_result.image_score
    result["patch_scores"] = detector_result.patch_scores
    result["anomaly_map"] = detector_result.anomaly_map
    decision = classify_image_score(route.category, detector_result.image_score, image_thresholds)
    result.update(
        {
            "threshold": decision["threshold"],
            "label": decision["label"],
            "is_defect": decision["is_defect"],
            "decision": decision["label"],
            "decision_source": config.detector,
            "decision_status": "available",
            "decision_is_defect": decision["is_defect"],
            "image_score": detector_result.image_score,
            "image_threshold": decision["threshold"],
            "binary_mask": binary_anomaly_masks(
                route.category,
                detector_result.anomaly_map[None, ...],
                pixel_thresholds,
            )[0].numpy(),
        }
    )
    return result


def _safe_ranking(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    if len(set(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _latency_summary(samples: list[float], device: torch.device) -> dict[str, Any]:
    """Summarize per-image inference timings from the official test loop."""

    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return {
            "device": str(device),
            "sample_count": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "images_per_second": None,
            "scope": "detector inference including shared DINOv2 forward and routing",
        }
    total = float(values.sum())
    return {
        "device": str(device),
        "sample_count": int(values.size),
        "total_seconds": total,
        "mean_seconds": float(values.mean()),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "images_per_second": float(values.size / total) if total > 0 else None,
        "scope": "detector inference including shared DINOv2 forward and routing",
    }


def _anomaly_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("anomaly_evaluated")]
    scores = [float(row["anomaly_score"]) for row in evaluated]
    labels = [int(row["ground_truth_is_defect"]) for row in evaluated]
    predictions = [int(row["decision_is_defect"]) for row in evaluated]
    report: dict[str, Any] = {
        "test_images": len(rows),
        "anomaly_evaluated_images": len(evaluated),
        "anomaly_skipped_images": len(rows) - len(evaluated),
        "image": {"ranking": _safe_ranking(scores, labels)},
    }
    if evaluated:
        true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
        false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
        false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
        true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
        report["image"]["thresholded"] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "counts": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "positive_total": sum(labels),
                "negative_total": len(labels) - sum(labels),
            },
        }
    pixel_scores: list[float] = []
    pixel_labels: list[int] = []
    pixel_predictions: list[int] = []
    pixel_rows = [
        row for row in evaluated
        if row.get("heatmap") is not None and row.get("binary_mask") is not None
    ]
    for row in pixel_rows:
        truth = np.asarray(row["mask"], dtype=np.uint8).reshape(-1)
        score_map = np.asarray(row["heatmap"], dtype=np.float32).reshape(-1)
        prediction = np.asarray(row["binary_mask"], dtype=np.uint8).reshape(-1)
        pixel_scores.extend(score_map.tolist())
        pixel_labels.extend((truth > 0).astype(np.int64).tolist())
        pixel_predictions.extend(prediction.astype(np.int64).tolist())
    report["pixel"] = {"ranking": _safe_ranking(pixel_scores, pixel_labels)}
    if pixel_predictions:
        true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(pixel_labels, pixel_predictions))
        false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(pixel_labels, pixel_predictions))
        false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(pixel_labels, pixel_predictions))
        true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(pixel_labels, pixel_predictions))
        report["pixel"]["thresholded"] = {
            "precision": float(precision_score(pixel_labels, pixel_predictions, zero_division=0)),
            "recall": float(recall_score(pixel_labels, pixel_predictions, zero_division=0)),
            "f1": float(f1_score(pixel_labels, pixel_predictions, zero_division=0)),
            "counts": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "positive_total": sum(pixel_labels),
                "negative_total": len(pixel_labels) - sum(pixel_labels),
            },
        }
    return report


def _classification_report(rows: list[dict[str, Any]], categories: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known = {category: index for index, category in enumerate(categories)}
    true = [known[row["true_category"]] for row in rows]
    accepted = [
        row for row in rows
        if not row["rejected"] and row.get("category") is not None
    ]
    predictions = [
        known[row["category"]]
        if not row["rejected"] and row.get("category") is not None
        else -1
        for row in rows
    ]
    correct = sum(
        not row["rejected"]
        and row.get("category") is not None
        and row["category"] == row["true_category"]
        for row in rows
    )
    rejected = sum(row["rejected"] for row in rows)
    accepted_correct = sum(row["category"] == row["true_category"] for row in accepted)
    per_category = []
    for category in categories:
        subset = [row for row in rows if row["true_category"] == category]
        category_rejected = sum(row["rejected"] for row in subset)
        category_correct = sum(
            not row["rejected"]
            and row.get("category") is not None
            and row["category"] == category
            for row in subset
        )
        category_incorrect = len(subset) - category_rejected - category_correct
        per_category.append(
            {
                "category": category,
                "support": len(subset),
                "correct": category_correct,
                "incorrect": category_incorrect,
                "rejected": category_rejected,
                "accuracy": category_correct / len(subset) if subset else 0.0,
                "rejection_rate": category_rejected / len(subset) if subset else 0.0,
            }
        )
    summary = {
        "total_test_images": len(rows),
        "accepted_known_predictions": len(accepted),
        "unknown_rejected_predictions": rejected,
        "rejection_rate": rejected / len(rows) if rows else 0.0,
        "top1_accuracy_all": correct / len(rows) if rows else 0.0,
        "top1_accuracy_accepted": accepted_correct / len(accepted) if accepted else 0.0,
        "macro_precision": float(precision_score(true, predictions, labels=list(range(len(categories))), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, predictions, labels=list(range(len(categories))), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, predictions, labels=list(range(len(categories))), average="macro", zero_division=0)),
        "per_category": per_category,
    }
    for split_name, split_rows in (
        ("good_test", [row for row in rows if not row["ground_truth_is_defect"]]),
        ("defect_test", [row for row in rows if row["ground_truth_is_defect"]]),
    ):
        split_accepted = [
            row for row in split_rows
            if not row["rejected"] and row.get("category") is not None
        ]
        split_correct = sum(
            not row["rejected"]
            and row.get("category") is not None
            and row["category"] == row["true_category"]
            for row in split_rows
        )
        summary[split_name] = {
            "total_images": len(split_rows),
            "accepted_known_predictions": len(split_accepted),
            "unknown_rejected_predictions": len(split_rows) - len(split_accepted),
            "accuracy": split_correct / len(split_rows) if split_rows else 0.0,
            "accepted_accuracy": (
                sum(row["category"] == row["true_category"] for row in split_accepted)
                / len(split_accepted)
                if split_accepted
                else 0.0
            ),
            "rejection_rate": (
                (len(split_rows) - len(split_accepted)) / len(split_rows)
                if split_rows
                else 0.0
            ),
        }
    return summary, per_category


def _load_mask(item: dict[str, Any], image_size: int) -> np.ndarray:
    if not item["ground_truth_is_defect"]:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    image = Path(item["image_path"])
    mask_path = image.parents[2] / "ground_truth" / item["defect_type"] / f"{image.stem}_mask.png"
    if not mask_path.is_file():
        return np.zeros((image_size, image_size), dtype=np.uint8)
    with Image.open(mask_path) as mask:
        return np.asarray(mask.convert("L").resize((image_size, image_size))) > 0


def _markdown_report(summary: dict[str, Any], report_dir: Path) -> str:
    classification = summary["classification"]
    anomaly = summary["anomaly"]
    lines = [
        f"# Production run report: `{summary['dataset_name']}`",
        "",
        "This report was generated automatically after the one-command pipeline completed.",
        "",
        "## Pipeline",
        "",
        "`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Processed root: `{summary['processed_root']}`",
        f"- Categories: **{len(summary['categories'])}**",
        f"- Classification: **{'enabled' if summary['classification_enabled'] else 'disabled (one category)'}**",
        f"- Image decision authority: **{summary['configuration']['decision_authority']}**",
        f"- Product confidence threshold: **{summary['configuration']['classification_confidence_threshold']:.2f}**",
        f"- kNN neighbors: **{summary['configuration']['k']}**",
        "",
        "## Product classification",
        "",
        f"- Official test images: **{classification['total_test_images']}**",
        f"- Top-1 accuracy over all images: **{classification['top1_accuracy_all']:.2%}**",
        f"- Top-1 accuracy over accepted images: **{classification['top1_accuracy_accepted']:.2%}**",
        f"- UNKNOWN/rejected: **{classification['unknown_rejected_predictions']} ({classification['rejection_rate']:.2%})**",
        f"- Macro precision / recall / F1: **{classification['macro_precision']:.4f} / {classification['macro_recall']:.4f} / {classification['macro_f1']:.4f}**",
        f"- Good-image accuracy: **{classification['good_test']['accuracy']:.2%}**; defect-image accuracy: **{classification['defect_test']['accuracy']:.2%}**",
        f"- Accuracy drop on defect images: **{classification['good_test']['accuracy'] - classification['defect_test']['accuracy']:.2%}**",
        "",
        "## Defect detection",
        "",
        f"- Images with anomaly inference: **{anomaly['anomaly_evaluated_images']}**",
        f"- Images without completed anomaly inference: **{anomaly['anomaly_skipped_images']}**",
        f"- Image AUROC / AP: **{anomaly['image']['ranking']['auroc']} / {anomaly['image']['ranking']['average_precision']}**",
        f"- Pixel AUROC / AP: **{anomaly['pixel']['ranking']['auroc']} / {anomaly['pixel']['ranking']['average_precision']}**",
        "",
        "## Inference latency",
        "",
        f"- Device: **{summary['performance']['inference']['device']}**",
        f"- Mean / median / p95: **{summary['performance']['inference']['mean_seconds']:.4f} / {summary['performance']['inference']['median_seconds']:.4f} / {summary['performance']['inference']['p95_seconds']:.4f} s/image**",
        f"- Throughput: **{summary['performance']['inference']['images_per_second']:.2f} images/s**",
        "",
        "## Generated files",
        "",
    ]
    for name in summary["output_files"].values():
        lines.append(f"- `{report_dir / name}`")
    return "\n".join(lines) + "\n"


def run_production(config: ProductionConfig | None = None) -> dict[str, Any]:
    """Run the complete dataset-driven pipeline and write the final report."""

    config = config or ProductionConfig()
    if config.detector not in {"exact_knn", "patchcore", "padim"}:
        raise ValueError("detector must be one of: exact_knn, patchcore, padim")
    if config.orchestration not in {"exact_first", "independent"}:
        raise ValueError("orchestration must be one of: exact_first, independent")
    raw_root = discover_dataset_root(config.datasets_root, config.dataset_root)
    processed_root = (config.processed_root or raw_root.parent / f"{raw_root.name}_processed").resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (config.report_root / raw_root.name / run_name).resolve()
    artifact_base = (config.artifact_root / raw_root.name).resolve()
    artifact_dir = (
        artifact_base / "variants" / config.artifact_namespace
        if config.artifact_namespace
        else artifact_base
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    preparation = prepare_mvtec_dataset(raw_root, processed_root, config.image_size, config.force)
    raw_validation = validate_raw_dataset(raw_root)
    _write_json(report_dir / "dataset_validation.json", preparation)
    plot_dataset_distribution(raw_validation, report_dir / "dataset_distribution.png")

    manifest_info = create_classification_manifest(
        processed_root,
        artifact_dir / "classification_split.csv",
        config.validation_fraction,
        config.seed,
    )
    rows = read_classification_manifest(manifest_info["manifest_path"], processed_root)
    mapping = category_mapping(rows)
    categories = [category for category, _ in sorted(mapping.items(), key=lambda item: item[1])]
    exact_first = config.orchestration == "exact_first"
    if exact_first:
        exact_encoder = SharedDINOv2(
            DINOv2Config(image_size=(config.image_size, config.image_size), feature_layers=()),
            artifact_root=artifact_base,
            device=config.device,
        )
        localization_encoder = SharedDINOv2(
            DINOv2Config(
                image_size=(config.image_size, config.image_size),
                feature_layers=config.detector_layers,
            ),
            artifact_root=artifact_base,
            device=config.device,
        )
        encoder = exact_encoder
    else:
        encoder = SharedDINOv2(
            DINOv2Config(
                image_size=(config.image_size, config.image_size),
                feature_layers=()
                if config.detector == "exact_knn"
                else config.detector_layers,
            ),
            artifact_root=artifact_base,
            device=config.device,
        )
        exact_encoder = encoder
        localization_encoder = encoder

    classifier = None
    reference_artifact = None
    calibration: dict[str, Any] = {}
    classification_report: dict[str, Any] | None = None
    if len(categories) > 1:
        train_artifact = build_global_embedding_artifact(rows, "train", processed_root, encoder, mapping, config.batch_size)
        validation_artifact = build_global_embedding_artifact(rows, "validation", processed_root, encoder, mapping, config.batch_size)
        embedding_metadata = make_embedding_metadata(rows, mapping, manifest_info["manifest_path"], processed_root, encoder)
        save_embedding_artifacts(artifact_dir / "embeddings", train_artifact, validation_artifact, mapping, embedding_metadata)
        classifier_config = LinearClassifierConfig()
        classifier = fit_linear_classifier(train_artifact, mapping, classifier_config)
        save_linear_classifier(artifact_dir / "classifier", classifier, mapping, classifier_config, embedding_metadata)
        reference_artifact = train_artifact
        calibration = calibrate_open_set_thresholds(
            classifier,
            reference_artifact,
            validation_artifact,
            mapping,
            classification_confidence_threshold=config.classification_confidence_threshold,
        )
        save_calibration(artifact_dir / "calibration.json", calibration)
    else:
        train_artifact = None

    train_paths = {
        category: [processed_root / row["image_path"] for row in rows if row["category"] == category and row["split"] == "train"]
        for category in categories
    }
    banks: dict[str, FeatureBankResult] = {}
    detectors: dict[str, AnomalyDetector] = {}
    localization_detectors: dict[str, dict[str, AnomalyDetector]] = {
        "patchcore": {},
        "padim": {},
    }
    if exact_first or config.detector == "exact_knn":
        for category in categories:
            paths = train_paths[category] or [
                processed_root / row["image_path"]
                for row in rows
                if row["category"] == category
            ]
            bank = load_or_build_feature_bank(
                artifact_dir / "feature_banks",
                category,
                paths,
                processed_root,
                exact_encoder,
                batch_size=config.batch_size,
                rebuild=config.force,
            )
            bank.metadata.setdefault("category", category)
            bank.path.parent.mkdir(parents=True, exist_ok=True)
            (bank.path.parent / "bank_metadata.json").write_text(
                json.dumps(bank.metadata, indent=2) + "\n", encoding="utf-8"
            )
            banks[category] = bank
        image_thresholds, pixel_thresholds = _calibrate_anomaly_thresholds(
            rows, processed_root, banks, exact_encoder, config
        )
        if exact_first:
            multi_layer_config = _detector_config(config)
            for detector_name in ("patchcore", "padim"):
                detector_root = artifact_dir / "detectors" / detector_name
                for category in categories:
                    paths = train_paths[category] or [
                        processed_root / row["image_path"]
                        for row in rows
                        if row["category"] == category
                    ]
                    kwargs: dict[str, Any] = {"config": multi_layer_config}
                    if detector_name == "patchcore":
                        kwargs["k"] = config.k
                    detector = create_detector(detector_name, **kwargs)
                    detector.fit(category, paths, localization_encoder, config.batch_size)
                    detector.save(detector_root / category)
                    localization_detectors[detector_name][category] = detector
            pixel_thresholds = _calibrate_detector_pixel_thresholds(
                rows,
                processed_root,
                localization_detectors["padim"],
                localization_encoder,
                config,
            )
    else:
        multi_layer_config = _detector_config(config)
        detector_root = artifact_dir / "detectors" / config.detector
        for category in categories:
            paths = train_paths[category] or [
                processed_root / row["image_path"]
                for row in rows
                if row["category"] == category
            ]
            kwargs: dict[str, Any] = {"config": multi_layer_config}
            if config.detector == "patchcore":
                kwargs["k"] = config.k
            detector = create_detector(config.detector, **kwargs)
            detector.fit(category, paths, encoder, config.batch_size)
            detector.save(detector_root / category)
            detectors[category] = detector
        image_thresholds, pixel_thresholds = _calibrate_detector_thresholds(
            rows, processed_root, detectors, encoder, config
        )
    threshold_name = "exact_first" if exact_first else config.detector
    threshold_root = artifact_dir / "thresholds" / threshold_name
    _write_json(threshold_root / "image_thresholds.json", image_thresholds)
    _write_json(threshold_root / "pixel_thresholds.json", pixel_thresholds)
    if exact_first or config.detector == "exact_knn":
        # Keep the historical CLI paths valid for the baseline model.
        _write_json(artifact_dir / "image_thresholds.json", image_thresholds)
        _write_json(artifact_dir / "pixel_thresholds.json", pixel_thresholds)
    test_records = discover_mvtec_test_images(processed_root)
    test_manifest = _test_manifest_info(test_records, processed_root)
    inference_rows: list[dict[str, Any]] = []
    inference_latencies: list[float] = []
    for item in test_records:
        synchronize(encoder.device)
        started = time.perf_counter()
        if exact_first:
            from ..anomaly.pipeline import infer_image

            exact_result_holder: dict[str, Any] = {}
            local_features: Any = None

            def exact_runner() -> dict[str, Any]:
                exact_result = infer_image(
                    item["image_path"],
                    exact_encoder,
                    classifier,
                    reference_artifact,
                    mapping,
                    calibration,
                    artifact_dir / "feature_banks",
                    image_thresholds,
                    None,
                    k=config.k,
                    classification_confidence_threshold=config.classification_confidence_threshold,
                )
                exact_result_holder["result"] = exact_result
                return exact_result

            def localization_features() -> Any:
                nonlocal local_features
                if local_features is None:
                    with torch.inference_mode():
                        local_features = localization_encoder.forward(
                            torch.stack(
                                [
                                    image_tensor(
                                        item["image_path"],
                                        localization_encoder.config,
                                    )
                                ]
                            )
                        )
                return local_features

            def patchcore_runner() -> np.ndarray:
                exact_result = exact_result_holder["result"]
                category = exact_result.get("category")
                detector = localization_detectors["patchcore"].get(category)
                if detector is None:
                    raise RuntimeError(f"PatchCore artifact is unavailable for category: {category}")
                return detector.predict(localization_features()).anomaly_map

            def padim_runner() -> np.ndarray:
                exact_result = exact_result_holder["result"]
                category = exact_result.get("category")
                detector = localization_detectors["padim"].get(category)
                if detector is None:
                    raise RuntimeError(f"PaDiM artifact is unavailable for category: {category}")
                detector_result = detector.predict(localization_features())
                return binary_anomaly_masks(
                    category,
                    detector_result.anomaly_map[None, ...],
                    pixel_thresholds,
                )[0].numpy()

            result = infer_exact_first(exact_runner, patchcore_runner, padim_runner)
        elif config.detector == "exact_knn":
            from ..anomaly.pipeline import infer_image

            result = infer_image(
                item["image_path"],
                encoder,
                classifier,
                reference_artifact,
                mapping,
                calibration,
                artifact_dir / "feature_banks",
                image_thresholds,
                pixel_thresholds,
                k=config.k,
                classification_confidence_threshold=config.classification_confidence_threshold,
            )
        else:
            result = _infer_with_detector(
                item,
                encoder,
                classifier,
                reference_artifact,
                mapping,
                calibration,
                detectors,
                image_thresholds,
                pixel_thresholds,
                config,
            )
        synchronize(encoder.device)
        inference_latencies.append(time.perf_counter() - started)
        if result.get("decision_is_defect") is None and result.get("is_defect") is not None:
            result["decision_is_defect"] = bool(result["is_defect"])
            result["decision"] = "DEFECT" if result["is_defect"] else "GOOD"
        if result.get("heatmap") is None and result.get("anomaly_map") is not None:
            result["heatmap"] = result["anomaly_map"]
        result.update({key: value for key, value in item.items() if key != "is_defect"})
        result["ground_truth_is_defect"] = bool(item["is_defect"])
        result["mask"] = _load_mask(result, config.image_size)
        result["anomaly_evaluated"] = (
            result.get("anomaly_score") is not None
            and result.get("decision_is_defect") is not None
        )
        inference_rows.append(result)

    classification_report, per_category = _classification_report(inference_rows, categories)
    anomaly_report = _anomaly_report(inference_rows)
    _write_json(report_dir / "classification_test_summary.json", classification_report)
    _write_json(report_dir / "anomaly_metrics.json", anomaly_report)
    _write_csv(
        report_dir / "per_category_metrics.csv",
        per_category,
        ["category", "support", "correct", "incorrect", "rejected", "accuracy", "rejection_rate"],
    )
    error_rows = [
        {
            "image_path": item["image_path"],
            "true_category": item["true_category"],
            "predicted_category": item.get("predicted_category"),
            "routed_category": item.get("category"),
            "confidence": item.get("confidence"),
            "routing_status": item.get("status"),
            "defect_type": item["defect_type"],
            "is_defect": item["ground_truth_is_defect"],
            "decision": item.get("decision"),
            "global_distance": item.get("nearest_global_distance"),
            "rejection_reason": item.get("reason", "") if item.get("rejected") else "",
        }
        for item in inference_rows
        if item.get("rejected") or item.get("category") != item["true_category"]
    ]
    _write_csv(
        report_dir / "error_analysis.csv",
        error_rows,
        ["image_path", "true_category", "predicted_category", "routed_category", "confidence", "routing_status", "defect_type", "is_defect", "decision", "global_distance", "rejection_reason"],
    )
    true_ids = [mapping[item["true_category"]] for item in inference_rows]
    pred_ids = [
        mapping[item["category"]]
        for item in inference_rows
        if not item.get("rejected") and item.get("category") is not None
    ]
    accepted_true = [
        mapping[item["true_category"]]
        for item in inference_rows
        if not item.get("rejected") and item.get("category") is not None
    ]
    _plot_confusion(
        confusion_matrix(accepted_true, pred_ids, labels=list(range(len(categories)))) if accepted_true else np.zeros((len(categories), len(categories)), dtype=int),
        categories,
        report_dir / "confusion_matrix.png",
        "Accepted product-category predictions",
    )
    unknown_pred = [
        mapping[item["category"]]
        if not item.get("rejected") and item.get("category") is not None
        else len(categories)
        for item in inference_rows
    ]
    _plot_confusion(
        confusion_matrix(true_ids, unknown_pred, labels=list(range(len(categories) + 1))),
        categories + ["UNKNOWN"],
        report_dir / "confusion_matrix_with_unknown.png",
        "Product-category predictions including UNKNOWN",
    )

    for category in categories:
        examples = [
            item
            for item in inference_rows
            if item.get("category") == category
            and item.get("anomaly_evaluated")
            and item.get("heatmap") is not None
        ]
        if not examples:
            continue
        good_examples = [item for item in examples if not item["ground_truth_is_defect"]]
        defect_examples = [item for item in examples if item["ground_truth_is_defect"]]
        selected = (good_examples[:2] + defect_examples[:2])[:4]
        images = [np.asarray(Image.open(item["image_path"]).convert("RGB")) for item in selected]
        masks = [item["mask"] for item in selected]
        maps = np.stack([item["heatmap"] for item in selected])
        figure = plot_anomaly_examples(
            category,
            images,
            masks,
            maps,
            [float(item["anomaly_score"]) for item in selected],
            [bool(item["ground_truth_is_defect"]) for item in selected],
            config.k,
            examples_per_class=max(1, min(2, len(selected))),
        )
        save_figure(figure, report_dir / "heatmaps" / f"{category}.png")

    summary = {
        "report_version": "production-v1",
        "dataset_name": raw_root.name,
        "dataset_root": str(raw_root),
        "processed_root": str(processed_root),
        "report_dir": str(report_dir),
        "categories": categories,
        "classification_enabled": len(categories) > 1,
        "configuration": {
            "image_size": config.image_size,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "k": config.k,
            "detector": config.detector,
            "orchestration": config.orchestration,
            "decision_authority": "exact_knn" if exact_first else config.detector,
            "detector_layers": list(config.detector_layers),
            "neighborhood_size": config.neighborhood_size,
            "coreset_ratio": config.coreset_ratio,
            "coreset_max_points": config.coreset_max_points,
            "padim_reduced_dim": config.padim_reduced_dim,
            "covariance_regularization": config.covariance_regularization,
            "classification_confidence_threshold": config.classification_confidence_threshold,
            "target_acceptance_rate": config.target_acceptance_rate,
            "artifact_namespace": config.artifact_namespace,
        },
        "preparation": preparation,
        "manifest": manifest_info,
        "test_manifest": test_manifest,
        "classification": classification_report,
        "anomaly": anomaly_report,
        "performance": {
            "inference": _latency_summary(inference_latencies, encoder.device),
        },
        "output_files": {
            "final_report": "final_report.md",
            "summary": "final_summary.json",
            "classification_summary": "classification_test_summary.json",
            "anomaly_metrics": "anomaly_metrics.json",
            "per_category": "per_category_metrics.csv",
            "errors": "error_analysis.csv",
            "confusion_matrix": "confusion_matrix.png",
            "confusion_matrix_with_unknown": "confusion_matrix_with_unknown.png",
            "dataset_distribution": "dataset_distribution.png",
            "heatmaps": "heatmaps/",
            "image_thresholds": str(threshold_root / "image_thresholds.json"),
            "pixel_thresholds": str(threshold_root / "pixel_thresholds.json"),
        },
    }
    _write_json(report_dir / "final_summary.json", summary)
    (report_dir / "final_report.md").write_text(_markdown_report(summary, report_dir), encoding="utf-8")
    return summary
