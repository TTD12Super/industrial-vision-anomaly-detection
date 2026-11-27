"""One-command automation for an arbitrary MVTec-style dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import csv
import pickle
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

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
from ..anomaly.refinement import (
    MASK_POSTPROCESS_METHODS,
    postprocess_mask,
    propagate_scores,
    refine_heatmap,
)
from ..anomaly.thresholds import (
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
    load_compatible_embedding_artifacts,
    read_classification_manifest,
    save_embedding_artifacts,
    sha256_file,
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
from ..evaluation.evaluation import evaluate_product_classifier, select_linear_classifier


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
    heatmap_refiner: str = "none"
    affinity_top_k: int = 128
    affinity_temperature: float = 0.10
    affinity_eta: float = 1.0
    mask_postprocess: str = "raw"
    device: str | None = None
    artifact_namespace: str | None = None
    force: bool = False


HEATMAP_REFINERS = (
    "none",
    "dino_affinity",
    "heatmap_open3",
    "heatmap_open5",
    "heatmap_open7",
    "heatmap_open9",
)


def _error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


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
            "anomaly_decision": None,
            "anomaly_decision_is_defect": None,
            "anomaly_decision_source": None,
            "anomaly_decision_status": "unavailable",
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
            "patchcore_heatmap": None,
            "padim_heatmap": None,
            "padim_binary_mask": None,
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
    result["patchcore_heatmap"] = None
    result["padim_heatmap"] = None
    result["padim_binary_mask"] = None
    result["localization_status"] = {"patchcore": "not_run", "padim": "not_run"}
    result["anomaly_decision"] = None
    result["anomaly_decision_is_defect"] = None
    result["anomaly_decision_source"] = None
    result["anomaly_decision_status"] = "unavailable"

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
    result["anomaly_decision"] = decision
    result["anomaly_decision_is_defect"] = decision == "DEFECT"
    result["anomaly_decision_source"] = "exact_knn"
    result["anomaly_decision_status"] = "available"
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
        result["patchcore_heatmap"] = heatmap
        result["patchcore"] = {"status": "available", "heatmap": heatmap, "error": None}
        result["localization_status"]["patchcore"] = "available"
    except Exception as error:
        message = _error_message(error)
        result["patchcore"] = {"status": "failed", "heatmap": None, "error": message}
        result["localization_status"]["patchcore"] = "failed"

    try:
        padim_output = padim_runner()
        padim_heatmap = None
        if isinstance(padim_output, dict):
            padim_heatmap = padim_output.get("heatmap")
            padim_output = padim_output.get("binary_mask")
        mask = np.asarray(padim_output, dtype=bool)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("PaDiM mask must be a non-empty 2D array")
        status = "available" if bool(mask.any()) else "empty"
        result["binary_mask"] = mask
        result["padim_binary_mask"] = mask
        if padim_heatmap is not None:
            padim_heatmap = np.asarray(padim_heatmap, dtype=np.float32)
            if padim_heatmap.ndim != 2 or not np.isfinite(padim_heatmap).all():
                raise ValueError("PaDiM heatmap must be a finite 2D array")
            result["padim_heatmap"] = padim_heatmap
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


_ERROR_ANALYSIS_FIELDS = [
    "image_path",
    "true_category",
    "predicted_category",
    "routed_category",
    "localization_category",
    "confidence",
    "routing_status",
    "defect_type",
    "is_defect",
    "decision",
    "anomaly_decision",
    "anomaly_decision_source",
    "anomaly_decision_status",
    "global_distance",
    "rejection_reason",
]


def _error_analysis_rows(inference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize rejected or misrouted rows without conflating route and anomaly fields."""

    return [
        {
            "image_path": item["image_path"],
            "true_category": item["true_category"],
            "predicted_category": item.get("predicted_category"),
            "routed_category": item.get("category"),
            "localization_category": item.get("localization_category"),
            "confidence": item.get("confidence"),
            "routing_status": item.get("status"),
            "defect_type": item["defect_type"],
            "is_defect": item["ground_truth_is_defect"],
            # Keep the legacy field for existing consumers; the explicit fields
            # identify the anomaly verdict and its provenance unambiguously.
            "decision": item.get("decision"),
            "anomaly_decision": item.get("anomaly_decision"),
            "anomaly_decision_source": item.get("anomaly_decision_source"),
            "anomaly_decision_status": item.get("anomaly_decision_status"),
            "global_distance": item.get("nearest_global_distance"),
            "rejection_reason": item.get("reason", "") if item.get("rejected") else "",
        }
        for item in inference_rows
        if item.get("rejected") or item.get("category") != item["true_category"]
    ]


_LOCALIZATION_SOURCES = ("exact_knn", "patchcore", "padim")
_LOCALIZATION_METRICS = (
    "map_images",
    "mask_images",
    "map_coverage",
    "mask_coverage",
    "rejected_images",
    "rejected_map_images",
    "rejected_mask_images",
    "rejected_map_coverage",
    "rejected_mask_coverage",
    "pixel_auroc",
    "pixel_average_precision",
    "precision",
    "recall",
    "f1",
    "specificity",
    "dice",
    "iou",
    "evaluated_pixels",
)
_LOCALIZATION_SUBGROUP_METRICS = (
    "good_image_count",
    "good_map_images",
    "good_mask_images",
    "good_map_coverage",
    "good_mask_coverage",
    "good_false_positive_pixel_rate",
    "good_predicted_mask_area_mean_pixels",
    "good_predicted_mask_area_mean_ratio",
    "defect_image_count",
    "defect_map_images",
    "defect_mask_images",
    "defect_map_coverage",
    "defect_mask_coverage",
    "defect_rejected_images",
    "defect_rejected_map_images",
    "defect_rejected_mask_images",
    "defect_rejected_map_coverage",
    "defect_rejected_mask_coverage",
    "defect_evaluated_pixels",
    "defect_pixel_auroc",
    "defect_pixel_average_precision",
    "defect_precision",
    "defect_recall",
    "defect_f1",
    "defect_specificity",
    "defect_dice",
    "defect_iou",
)
_LOCALIZATION_CSV_FIELDS = [
    "true_category",
    "image_count",
    *(
        f"{source}_{metric}"
        for source in _LOCALIZATION_SOURCES
        for metric in _LOCALIZATION_METRICS
    ),
    *(
        f"{source}_{metric}"
        for source in _LOCALIZATION_SOURCES
        for metric in _LOCALIZATION_SUBGROUP_METRICS
    ),
]


def _localization_metric_rows(anomaly_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-category, per-detector localization metrics for CSV output."""

    rows: list[dict[str, Any]] = []
    for category_report in anomaly_report.get("per_category", []):
        row: dict[str, Any] = {
            "true_category": category_report["true_category"],
            "image_count": category_report["image_count"],
        }
        for source in _LOCALIZATION_SOURCES:
            metrics = category_report.get("sources", {}).get(source)
            ranking = metrics.get("ranking", {}) if metrics else {}
            thresholded = metrics.get("thresholded", {}) if metrics else {}
            values = {
                "map_images": metrics.get("map_images") if metrics else None,
                "mask_images": metrics.get("mask_images") if metrics else None,
                "map_coverage": metrics.get("map_coverage") if metrics else None,
                "mask_coverage": metrics.get("mask_coverage") if metrics else None,
                "rejected_images": metrics.get("rejected_images") if metrics else None,
                "rejected_map_images": metrics.get("rejected_map_images") if metrics else None,
                "rejected_mask_images": metrics.get("rejected_mask_images") if metrics else None,
                "rejected_map_coverage": metrics.get("rejected_map_coverage") if metrics else None,
                "rejected_mask_coverage": metrics.get("rejected_mask_coverage") if metrics else None,
                "pixel_auroc": ranking.get("auroc"),
                "pixel_average_precision": ranking.get("average_precision"),
                "precision": thresholded.get("precision"),
                "recall": thresholded.get("recall"),
                "f1": thresholded.get("f1"),
                "specificity": thresholded.get("specificity"),
                "dice": thresholded.get("dice"),
                "iou": thresholded.get("iou"),
                "evaluated_pixels": thresholded.get("evaluated_pixels"),
            }
            row.update({f"{source}_{metric}": value for metric, value in values.items()})
            subgroups = metrics.get("subgroups", {}) if metrics else {}
            good = subgroups.get("good", {})
            good_area = good.get("predicted_mask_area", {})
            defect = subgroups.get("defect", {})
            defect_ranking = defect.get("ranking", {})
            defect_thresholded = defect.get("thresholded", {})
            subgroup_values = {
                "good_image_count": good.get("image_count"),
                "good_map_images": good.get("map_images"),
                "good_mask_images": good.get("mask_images"),
                "good_map_coverage": good.get("map_coverage"),
                "good_mask_coverage": good.get("mask_coverage"),
                "good_false_positive_pixel_rate": good.get("false_positive_pixel_rate"),
                "good_predicted_mask_area_mean_pixels": good_area.get("mean_pixels"),
                "good_predicted_mask_area_mean_ratio": good_area.get("mean_ratio"),
                "defect_image_count": defect.get("image_count"),
                "defect_map_images": defect.get("map_images"),
                "defect_mask_images": defect.get("mask_images"),
                "defect_map_coverage": defect.get("map_coverage"),
                "defect_mask_coverage": defect.get("mask_coverage"),
                "defect_rejected_images": defect.get("rejected_images"),
                "defect_rejected_map_images": defect.get("rejected_map_images"),
                "defect_rejected_mask_images": defect.get("rejected_mask_images"),
                "defect_rejected_map_coverage": defect.get("rejected_map_coverage"),
                "defect_rejected_mask_coverage": defect.get("rejected_mask_coverage"),
                "defect_evaluated_pixels": defect_thresholded.get("evaluated_pixels"),
                "defect_pixel_auroc": defect_ranking.get("auroc"),
                "defect_pixel_average_precision": defect_ranking.get("average_precision"),
                "defect_precision": defect_thresholded.get("precision"),
                "defect_recall": defect_thresholded.get("recall"),
                "defect_f1": defect_thresholded.get("f1"),
                "defect_specificity": defect_thresholded.get("specificity"),
                "defect_dice": defect_thresholded.get("dice"),
                "defect_iou": defect_thresholded.get("iou"),
            }
            row.update(
                {f"{source}_{metric}": value for metric, value in subgroup_values.items()}
            )
        rows.append(row)
    return rows


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


def _validation_rows_by_category(
    rows: list[dict[str, str]], categories: Iterable[str]
) -> dict[str, list[dict[str, str]]]:
    selected = {
        category: [
            row for row in rows
            if row["category"] == category and row["split"] == "validation"
        ]
        for category in categories
    }
    missing = sorted(category for category, category_rows in selected.items() if not category_rows)
    if missing:
        raise ValueError(f"No validation images for threshold calibration: {missing}")
    return selected


def _calibrate_anomaly_thresholds(
    rows: list[dict[str, str]],
    processed_root: Path,
    banks: dict[str, FeatureBankResult],
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_scores: dict[str, list[float]] = {}
    pixel_scores: dict[str, list[float]] = {}
    validation_rows = _validation_rows_by_category(rows, banks)
    for category in sorted(banks):
        selected = validation_rows[category]
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


def _validate_category_train_paths(train_paths: dict[str, list[Path]]) -> None:
    missing = [category for category, paths in train_paths.items() if not paths]
    if missing:
        raise ValueError(f"No training images for categories: {missing}")


def _encoder_artifact_identity(encoder: SharedDINOv2) -> dict[str, Any]:
    checkpoints = sorted(
        (encoder.artifact_root / "torch_hub" / "checkpoints").glob(
            f"{encoder.config.model_name}*_pretrain.pth"
        )
    )
    return {
        **encoder.metadata(),
        "checkpoint_sha256": sha256_file(checkpoints[0]) if len(checkpoints) == 1 else None,
    }


def _training_images_sha256(paths: list[Path], processed_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative_path = path.resolve().relative_to(processed_root.resolve()).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _detector_artifact_identity(
    category: str,
    detector: AnomalyDetector,
    config: MultiLayerConfig,
    encoder_identity: dict[str, Any],
    manifest_sha256: str,
    training_images_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "category": category,
        "detector": detector.name,
        "configuration": json.loads(json.dumps(config.to_dict())),
        "encoder": encoder_identity,
        "training_manifest_sha256": manifest_sha256,
        "training_images_sha256": training_images_sha256,
        "cache_reusable": encoder_identity["checkpoint_sha256"] is not None,
    }


def _load_or_fit_detector(
    detector: AnomalyDetector,
    category: str,
    paths: list[Path],
    encoder: SharedDINOv2,
    batch_size: int,
    artifact_path: Path,
    artifact_name: str,
    expected_config: MultiLayerConfig,
    expected_identity: dict[str, Any],
    force: bool,
) -> AnomalyDetector:
    artifact_file = artifact_path / artifact_name
    metadata_file = artifact_path / "artifact_metadata.json"
    can_check_cache = (
        not force
        and expected_identity["cache_reusable"]
        and artifact_file.is_file()
        and metadata_file.is_file()
    )
    if can_check_cache:
        try:
            stored_identity = json.loads(metadata_file.read_text(encoding="utf-8"))
            metadata_matches = isinstance(stored_identity, dict) and (
                stored_identity.get("artifact_file_sha256") == sha256_file(artifact_file)
                and stored_identity.get("artifact_file_size") == artifact_file.stat().st_size
                and {
                    key: value
                    for key, value in stored_identity.items()
                    if key not in {"artifact_file_sha256", "artifact_file_size"}
                }
                == expected_identity
            )
        except (OSError, json.JSONDecodeError):
            metadata_matches = False
        if metadata_matches:
            try:
                detector.load(artifact_path)
                detector.validate_artifacts()
            except (
                AttributeError,
                EOFError,
                KeyError,
                OSError,
                pickle.UnpicklingError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                metadata_matches = False
        if (
            metadata_matches
            and detector.config == expected_config
            and detector.category == expected_identity["category"]
            and detector.patch_grid == encoder.patch_grid
            and detector.padded_size == encoder.padded_size
            and detector.image_size == encoder.config.image_size
        ):
            return detector
    detector.config = expected_config
    detector.fit(category, paths, encoder, batch_size)
    detector.save(artifact_path)
    saved_identity = {
        **expected_identity,
        "artifact_file_sha256": sha256_file(artifact_file),
        "artifact_file_size": artifact_file.stat().st_size,
    }
    metadata_file.write_text(
        json.dumps(saved_identity, indent=2) + "\n", encoding="utf-8"
    )
    return detector


def _score_with_detector(
    image_path: Path,
    encoder: SharedDINOv2,
    detector: AnomalyDetector,
) -> AnomalyResult:
    with torch.inference_mode():
        features = encoder.forward(torch.stack([image_tensor(image_path, encoder.config)]))
    return detector.predict(features)


def _make_refined_anomaly_map(
    patch_scores: np.ndarray,
    patch_features: torch.Tensor,
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the configured patch- or map-space refiner consistently."""

    scores = patch_scores
    if config.heatmap_refiner == "dino_affinity":
        scores = propagate_scores(
            scores,
            patch_features.detach().cpu().numpy(),
            config.affinity_top_k,
            config.affinity_temperature,
            config.affinity_eta,
        )
    anomaly_map = make_anomaly_maps(
        torch.as_tensor(scores)[None, :],
        encoder.patch_grid,
        encoder.padded_size,
        encoder.config.image_size,
    )[0]
    if config.heatmap_refiner.startswith("heatmap_"):
        anomaly_map = refine_heatmap(anomaly_map, config.heatmap_refiner)
    return scores, anomaly_map


def _threshold_artifact_dir(
    artifact_dir: Path, threshold_name: str, heatmap_refiner: str
) -> Path:
    root = artifact_dir / "thresholds" / threshold_name
    if heatmap_refiner != "none":
        root = root / "refiners" / heatmap_refiner
    return root


def _score_detector_paths(
    paths: list[Path],
    encoder: SharedDINOv2,
    detector: AnomalyDetector,
    batch_size: int,
    config: ProductionConfig,
) -> list[AnomalyResult]:
    """Score paths in batches for efficient validation calibration."""

    if not paths or batch_size < 1:
        raise ValueError("paths must be non-empty and batch_size must be positive")
    results: list[AnomalyResult] = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack(
            [image_tensor(path, encoder.config) for path in paths[start : start + batch_size]]
        )
        with torch.inference_mode():
            features = encoder.forward(batch)
        batch_results = detector.predict_batch(features)
        for index, result in enumerate(batch_results):
            patch_scores, anomaly_map = _make_refined_anomaly_map(
                result.patch_scores,
                features.patch_embeddings[index],
                encoder,
                config,
            )
            batch_results[index] = AnomalyResult(
                float(np.max(patch_scores)),
                anomaly_map,
                patch_scores,
                result.metadata,
            )
        results.extend(batch_results)
    return results


def _calibrate_detector_thresholds(
    rows: list[dict[str, str]],
    processed_root: Path,
    detectors: dict[str, AnomalyDetector],
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_scores: dict[str, list[float]] = {}
    pixel_scores: dict[str, list[float]] = {}
    validation_rows = _validation_rows_by_category(rows, detectors)
    for category in sorted(detectors):
        selected = validation_rows[category]
        paths = [processed_root / row["image_path"] for row in selected]
        results = _score_detector_paths(
            paths, encoder, detectors[category], config.batch_size, config
        )
        image_scores[category] = [result.image_score for result in results]
        pixel_scores[category] = np.concatenate(
            [result.anomaly_map.reshape(-1) for result in results]
        ).tolist()
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
    validation_rows = _validation_rows_by_category(rows, detectors)
    for category in sorted(detectors):
        selected = validation_rows[category]
        paths = [processed_root / row["image_path"] for row in selected]
        results = _score_detector_paths(
            paths, encoder, detectors[category], config.batch_size, config
        )
        maps = [result.anomaly_map.reshape(-1) for result in results]
        pixel_scores[category] = np.concatenate(maps).tolist()
    return calibrate_category_pixel_thresholds(pixel_scores, config.target_acceptance_rate)


def _infer_with_detector(
    item: dict[str, Any],
    encoder: SharedDINOv2,
    classifier_encoder: SharedDINOv2,
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
        classifier_features = (
            features
            if classifier_encoder is encoder
            else classifier_encoder.forward(
                torch.stack([image_tensor(item["image_path"], classifier_encoder.config)])
            )
        )
    route = route_product(
        classifier_features.global_embedding,
        classifier,
        reference_artifact,
        mapping,
        calibration,
        config.artifact_root,
        classifier_encoder.metadata(),
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
            "anomaly_detector": config.detector,
            "anomaly_decision": None,
            "anomaly_decision_is_defect": None,
            "anomaly_decision_source": None,
            "anomaly_decision_status": "not_run",
            # Legacy anomaly-decision aliases retained for existing consumers.
            "decision": None,
            "decision_is_defect": None,
            "decision_source": None,
            "decision_status": "not_run",
            # Classification may reject the route while localization still uses
            # the classifier candidate; this field must never affect routing.
            "localization_category": route.predicted_category,
        }
    )
    localization_category = result["localization_category"]
    if localization_category not in detectors:
        return result
    detector_result = detectors[localization_category].predict(features)
    patch_scores, anomaly_map = _make_refined_anomaly_map(
        detector_result.patch_scores,
        features.patch_embeddings[0],
        encoder,
        config,
    )
    result["anomaly_score"] = float(np.max(patch_scores))
    result["patch_scores"] = patch_scores
    result["anomaly_map"] = anomaly_map
    anomaly_decision = classify_image_score(
        localization_category, result["anomaly_score"], image_thresholds
    )
    pixel_threshold = float(
        pixel_thresholds["categories"][localization_category]["threshold"]
    )
    result.update(
        {
            "threshold": anomaly_decision["threshold"],
            "label": anomaly_decision["label"],
            "is_defect": anomaly_decision["is_defect"],
            "anomaly_decision": anomaly_decision["label"],
            "anomaly_decision_is_defect": anomaly_decision["is_defect"],
            "anomaly_decision_source": config.detector,
            "anomaly_decision_status": "available",
            # Legacy aliases for consumers that have not migrated yet.
            "decision": anomaly_decision["label"],
            "decision_source": config.detector,
            "decision_status": "available",
            "decision_is_defect": anomaly_decision["is_defect"],
            "image_score": result["anomaly_score"],
            "image_threshold": anomaly_decision["threshold"],
            "binary_mask": postprocess_mask(
                anomaly_map, pixel_threshold, config.mask_postprocess
            ),
        }
    )
    return result


def _safe_ranking(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    score_values = np.asarray(scores, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    if np.unique(label_values).size < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(label_values, score_values)),
        "average_precision": float(average_precision_score(label_values, score_values)),
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
        report["image"]["thresholded"] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }

    def pixel_report(
        map_key: str,
        mask_key: str | None,
        scoped_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        scoped_rows = rows if scoped_rows is None else scoped_rows
        map_rows = [row for row in scoped_rows if row.get(map_key) is not None]
        score_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        prediction_chunks: list[np.ndarray] = []
        mask_rows = [
            row for row in map_rows
            if mask_key is not None and row.get(mask_key) is not None
        ]
        for row in map_rows:
            truth = np.asarray(row["mask"], dtype=np.uint8).reshape(-1)
            score_map = np.asarray(row[map_key], dtype=np.float32).reshape(-1)
            score_chunks.append(score_map)
            label_chunks.append((truth > 0).astype(np.int64))
        for row in mask_rows:
            prediction_chunks.append(
                np.asarray(row[mask_key], dtype=np.uint8).reshape(-1).astype(np.int64)
            )
        pixel_scores = np.concatenate(score_chunks) if score_chunks else np.empty(0, dtype=np.float32)
        pixel_labels = np.concatenate(label_chunks) if label_chunks else np.empty(0, dtype=np.int64)
        pixel_predictions = (
            np.concatenate(prediction_chunks) if prediction_chunks else np.empty(0, dtype=np.int64)
        )
        rejected_rows = [row for row in scoped_rows if row.get("rejected")]
        rejected_map_rows = [row for row in map_rows if row.get("rejected")]
        rejected_mask_rows = [row for row in mask_rows if row.get("rejected")]
        result: dict[str, Any] = {
            "map_key": map_key,
            "map_images": len(map_rows),
            "mask_images": len(mask_rows),
            "map_coverage": len(map_rows) / len(scoped_rows) if scoped_rows else 0.0,
            "mask_coverage": len(mask_rows) / len(scoped_rows) if scoped_rows else 0.0,
            "rejected_images": len(rejected_rows),
            "rejected_map_images": len(rejected_map_rows),
            "rejected_mask_images": len(rejected_mask_rows),
            "rejected_map_coverage": (
                len(rejected_map_rows) / len(rejected_rows) if rejected_rows else None
            ),
            "rejected_mask_coverage": (
                len(rejected_mask_rows) / len(rejected_rows) if rejected_rows else None
            ),
            "ranking": _safe_ranking(pixel_scores, pixel_labels),
        }
        if mask_rows:
            mask_label_chunks: list[np.ndarray] = []
            for row in mask_rows:
                mask_label_chunks.append(
                    (np.asarray(row["mask"], dtype=np.uint8).reshape(-1) > 0).astype(np.int64)
                )
            mask_labels = np.concatenate(mask_label_chunks)
            tn, fp, fn, tp = confusion_matrix(
                mask_labels, pixel_predictions, labels=[0, 1]
            ).ravel()
            denominator = 2 * tp + fp + fn
            union = tp + fp + fn
            result["thresholded"] = {
                "precision": float(precision_score(mask_labels, pixel_predictions, zero_division=0)),
                "recall": float(recall_score(mask_labels, pixel_predictions, zero_division=0)),
                "f1": float(f1_score(mask_labels, pixel_predictions, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
                "dice": float(2 * tp / denominator) if denominator else 0.0,
                "iou": float(tp / union) if union else 0.0,
                "evaluated_pixels": len(mask_labels),
            }
        return result

    def subgroup_reports(
        scoped_rows: list[dict[str, Any]],
        map_key: str,
        mask_key: str | None,
    ) -> dict[str, Any]:
        """Evaluate GOOD/DEFECT groups after inference without changing routing."""

        good_rows = [row for row in scoped_rows if not row.get("ground_truth_is_defect")]
        defect_rows = [row for row in scoped_rows if row.get("ground_truth_is_defect")]
        good_map_rows = [row for row in good_rows if row.get(map_key) is not None]
        good_mask_rows = [
            row for row in good_map_rows
            if mask_key is not None and row.get(mask_key) is not None
        ]
        predicted_areas = [
            int(np.asarray(row[mask_key], dtype=bool).sum())
            for row in good_mask_rows
        ]
        predicted_sizes = [
            int(np.asarray(row[mask_key], dtype=bool).size)
            for row in good_mask_rows
        ]
        predicted_pixels = sum(predicted_areas)
        total_pixels = sum(predicted_sizes)
        predicted_area_ratios = [
            area / size for area, size in zip(predicted_areas, predicted_sizes) if size
        ]
        good_report = {
            "image_count": len(good_rows),
            "map_images": len(good_map_rows),
            "mask_images": len(good_mask_rows),
            "map_coverage": len(good_map_rows) / len(good_rows) if good_rows else 0.0,
            "mask_coverage": len(good_mask_rows) / len(good_rows) if good_rows else 0.0,
            "false_positive_pixel_rate": (
                float(predicted_pixels / total_pixels) if total_pixels else None
            ),
            "predicted_mask_area": {
                "sample_count": len(predicted_areas),
                "total_pixels": predicted_pixels,
                "mean_pixels": float(np.mean(predicted_areas)) if predicted_areas else None,
                "mean_ratio": (
                    float(np.mean(predicted_area_ratios)) if predicted_area_ratios
                    else None
                ),
            },
        }
        return {
            "good": good_report,
            "defect": {
                **pixel_report(map_key, mask_key, defect_rows),
                "image_count": len(defect_rows),
            },
        }

    def detector_report(
        map_key: str,
        mask_key: str | None,
        scoped_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = pixel_report(map_key, mask_key, scoped_rows)
        result["subgroups"] = subgroup_reports(scoped_rows, map_key, mask_key)
        return result

    detector_reports: dict[str, Any] = {}
    detector_specs: dict[str, tuple[str, str | None]] = {}
    if any(row.get("anomaly_detector") for row in rows):
        detector = next(
            (row.get("anomaly_detector") for row in rows if row.get("anomaly_detector")),
            "unknown",
        )
        detector_specs[detector] = ("heatmap", "binary_mask")
        detector_reports[detector] = detector_report("heatmap", "binary_mask", rows)
    else:
        detector_specs = {
            "patchcore": ("patchcore_heatmap", None),
            "padim": ("padim_heatmap", "padim_binary_mask"),
        }
        detector_reports["patchcore"] = detector_report(
            "patchcore_heatmap", None, rows
        )
        detector_reports["padim"] = detector_report(
            "padim_heatmap", "padim_binary_mask", rows
        )
    report["detectors"] = detector_reports
    # Ground-truth categories are used only to group completed predictions for evaluation.
    per_category = []
    categories = sorted(
        {row.get("true_category") for row in rows if row.get("true_category") is not None}
    )
    for category in categories:
        category_rows = [row for row in rows if row.get("true_category") == category]
        per_category.append(
            {
                "true_category": category,
                "image_count": len(category_rows),
                "sources": {
                    detector: detector_report(map_key, mask_key, category_rows)
                    for detector, (map_key, mask_key) in detector_specs.items()
                },
            }
        )
    report["per_category"] = per_category
    # Backward-compatible headline: independent runs have one detector; exact-first
    # keeps PatchCore as the continuous heatmap headline and exposes PaDiM beside it.
    report["pixel"] = next(iter(detector_reports.values()))
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


def _pipeline_description(summary: dict[str, Any]) -> str:
    """Describe the actual orchestration and detector used by a production run."""

    configuration = summary.get("configuration", {})
    orchestration = configuration.get("orchestration", "exact_first")
    detector = configuration.get("detector", "exact_knn")
    prefix = "raw validation → preprocessing → train/validation split → product classification/routing"
    if orchestration == "exact_first":
        return (
            f"`{prefix} → Exact kNN image decision → DEFECT: "
            "PatchCore heatmap + PaDiM binary mask → evaluation`"
        )
    independent_descriptions = {
        "exact_knn": "independent Exact kNN anomaly decision",
        "patchcore": "independent PatchCore heatmap and mask evaluation",
        "padim": "independent PaDiM anomaly map and binary mask evaluation",
    }
    detector_description = independent_descriptions.get(
        detector, f"independent {detector} detector evaluation"
    )
    return f"`{prefix} → {detector_description} → evaluation`"


def _markdown_report(summary: dict[str, Any], report_dir: Path) -> str:
    classification = summary["classification"]
    anomaly = summary["anomaly"]
    configuration = summary["configuration"]

    def metric_text(value: Any, percent: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.2%}" if percent else f"{float(value):.4f}"

    lines = [
        f"# Production run report: `{summary['dataset_name']}`",
        "",
        "This report was generated automatically after the one-command pipeline completed.",
        "",
        "## Pipeline",
        "",
        _pipeline_description(summary),
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Processed root: `{summary['processed_root']}`",
        f"- Categories: **{len(summary['categories'])}**",
        f"- Classification: **{'enabled' if summary['classification_enabled'] else 'disabled (one category)'}**",
        f"- Orchestration: **{configuration['orchestration']}**",
        f"- Configured detector: **{configuration['detector']}**",
        f"- Image decision authority: **{configuration['decision_authority']}**",
        f"- Product confidence threshold: **{configuration['classification_confidence_threshold']:.2f}**",
        f"- kNN neighbors: **{configuration['k']}**",
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
    ]

    lines.extend(["", "## Localization coverage", ""])
    lines.append(
        f"- Classifier-rejected rows: **{classification['unknown_rejected_predictions']}**"
    )
    for detector_name, detector_report in anomaly.get("detectors", {}).items():
        rejected_images = detector_report.get("rejected_images", 0)
        rejected_map_images = detector_report.get("rejected_map_images", 0)
        rejected_mask_images = detector_report.get("rejected_mask_images", 0)
        lines.append(
            f"- `{detector_name}` rejected localization: maps **{rejected_map_images}/{rejected_images} "
            f"({metric_text(detector_report.get('rejected_map_coverage'), True)})**; "
            f"masks **{rejected_mask_images}/{rejected_images} "
            f"({metric_text(detector_report.get('rejected_mask_coverage'), True)})**"
        )

    lines.extend(["", "## GOOD-vs-DEFECT localization", ""])
    for detector_name, detector_report in anomaly.get("detectors", {}).items():
        subgroups = detector_report.get("subgroups", {})
        good = subgroups.get("good", {})
        good_area = good.get("predicted_mask_area", {})
        defect = subgroups.get("defect", {})
        defect_ranking = defect.get("ranking", {})
        defect_thresholded = defect.get("thresholded", {})
        lines.append(
            f"- `{detector_name}` GOOD: map coverage "
            f"**{good.get('map_images', 0)}/{good.get('image_count', 0)} "
            f"({metric_text(good.get('map_coverage'), True)})**; "
            f"mask coverage **{good.get('mask_images', 0)}/{good.get('image_count', 0)} "
            f"({metric_text(good.get('mask_coverage'), True)})**; "
            f"false-positive pixel rate "
            f"**{metric_text(good.get('false_positive_pixel_rate'), True)}** "
            f"(conditional on available GOOD masks; interpret with GOOD mask coverage); "
            f"mean predicted-mask area **{metric_text(good_area.get('mean_pixels'))} px "
            f"({metric_text(good_area.get('mean_ratio'), True)})**."
        )
        lines.append(
            f"- `{detector_name}` DEFECT: map coverage "
            f"**{defect.get('map_images', 0)}/{defect.get('image_count', 0)} "
            f"({metric_text(defect.get('map_coverage'), True)})**; "
            f"mask coverage **{defect.get('mask_images', 0)}/{defect.get('image_count', 0)} "
            f"({metric_text(defect.get('mask_coverage'), True)})**; "
            f"Pixel AUROC/AP "
            f"**{metric_text(defect_ranking.get('auroc'))} / "
            f"{metric_text(defect_ranking.get('average_precision'))}**; "
            f"precision/recall/F1/specificity/Dice/IoU "
            f"**{metric_text(defect_thresholded.get('precision'))} / "
            f"{metric_text(defect_thresholded.get('recall'))} / "
            f"{metric_text(defect_thresholded.get('f1'))} / "
            f"{metric_text(defect_thresholded.get('specificity'))} / "
            f"{metric_text(defect_thresholded.get('dice'))} / "
            f"{metric_text(defect_thresholded.get('iou'))}**."
        )

    lines.extend(
        [
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
    )
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
    if config.heatmap_refiner not in HEATMAP_REFINERS:
        raise ValueError(f"heatmap_refiner must be one of: {', '.join(HEATMAP_REFINERS)}")
    if config.mask_postprocess not in MASK_POSTPROCESS_METHODS:
        raise ValueError(
            f"mask_postprocess must be one of: {', '.join(MASK_POSTPROCESS_METHODS)}"
        )
    if (
        config.orchestration == "independent"
        and config.detector == "exact_knn"
        and config.heatmap_refiner != "none"
    ):
        raise ValueError(
            "heatmap_refiner is not supported for independent exact_knn; "
            "use patchcore or padim for refined localization"
        )
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
        classifier_encoder = SharedDINOv2(
            DINOv2Config(
                image_size=(config.image_size, config.image_size),
                feature_layers=(),
            ),
            artifact_root=artifact_base,
            device=config.device,
        )
        exact_encoder = classifier_encoder
        localization_encoder = (
            classifier_encoder
            if config.detector == "exact_knn"
            else SharedDINOv2(
                DINOv2Config(
                    image_size=(config.image_size, config.image_size),
                    feature_layers=config.detector_layers,
                ),
                artifact_root=artifact_base,
                device=config.device,
            )
        )
        encoder = localization_encoder

    classifier_encoder = exact_encoder

    classifier = None
    reference_artifact = None
    calibration: dict[str, Any] = {}
    validation_classification: dict[str, Any] | None = None
    classification_report: dict[str, Any] | None = None
    if len(categories) > 1:
        embedding_metadata = make_embedding_metadata(rows, mapping, manifest_info["manifest_path"], processed_root, classifier_encoder)
        cached_embeddings = None if config.force else load_compatible_embedding_artifacts(
            artifact_dir / "embeddings", embedding_metadata
        )
        if cached_embeddings is None:
            train_artifact = build_global_embedding_artifact(
                rows, "train", processed_root, classifier_encoder, mapping, config.batch_size
            )
            validation_artifact = build_global_embedding_artifact(
                rows, "validation", processed_root, classifier_encoder, mapping, config.batch_size
            )
            save_embedding_artifacts(
                artifact_dir / "embeddings",
                train_artifact,
                validation_artifact,
                mapping,
                embedding_metadata,
            )
        else:
            train_artifact, validation_artifact = cached_embeddings
        classifier, classifier_config, validation_classification, _ = select_linear_classifier(
            train_artifact,
            validation_artifact,
            mapping,
            [
                LinearClassifierConfig(C=value, max_iter=500)
                for value in (0.1, 1.0, 10.0)
            ],
        )
        _write_json(
            artifact_dir / "classification_validation_summary.json",
            validation_classification,
        )
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
    _validate_category_train_paths(train_paths)
    manifest_sha256 = sha256_file(manifest_info["manifest_path"])
    detector_encoder_identity = (
        _encoder_artifact_identity(localization_encoder)
        if exact_first or config.detector in {"patchcore", "padim"}
        else {}
    )
    training_images_hashes: dict[str, str] = {}
    banks: dict[str, FeatureBankResult] = {}
    detectors: dict[str, AnomalyDetector] = {}
    localization_detectors: dict[str, dict[str, AnomalyDetector]] = {
        "patchcore": {},
        "padim": {},
    }
    if exact_first or config.detector == "exact_knn":
        for category in categories:
            paths = train_paths[category]
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
                    paths = train_paths[category]
                    kwargs: dict[str, Any] = {"config": multi_layer_config}
                    if detector_name == "patchcore":
                        kwargs["k"] = config.k
                    detector = create_detector(detector_name, **kwargs)
                    artifact_path = detector_root / category
                    if category not in training_images_hashes:
                        training_images_hashes[category] = _training_images_sha256(
                            paths, processed_root
                        )
                    artifact_identity = _detector_artifact_identity(
                        category,
                        detector,
                        multi_layer_config,
                        detector_encoder_identity,
                        manifest_sha256,
                        training_images_hashes[category],
                    )
                    _load_or_fit_detector(
                        detector,
                        category,
                        paths,
                        localization_encoder,
                        config.batch_size,
                        artifact_path,
                        "coreset.pt" if detector_name == "patchcore" else "statistics.pt",
                        multi_layer_config,
                        artifact_identity,
                        config.force,
                    )
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
            paths = train_paths[category]
            kwargs: dict[str, Any] = {"config": multi_layer_config}
            if config.detector == "patchcore":
                kwargs["k"] = config.k
            detector = create_detector(config.detector, **kwargs)
            artifact_path = detector_root / category
            artifact_name = {
                "patchcore": "coreset.pt",
                "padim": "statistics.pt",
            }.get(config.detector)
            if artifact_name:
                if category not in training_images_hashes:
                    training_images_hashes[category] = _training_images_sha256(paths, processed_root)
                artifact_identity = _detector_artifact_identity(
                    category,
                    detector,
                    multi_layer_config,
                    detector_encoder_identity,
                    manifest_sha256,
                    training_images_hashes[category],
                )
                _load_or_fit_detector(
                    detector,
                    category,
                    paths,
                    localization_encoder,
                    config.batch_size,
                    artifact_path,
                    artifact_name,
                    multi_layer_config,
                    artifact_identity,
                    config.force,
                )
            else:
                detector.fit(category, paths, localization_encoder, config.batch_size)
                detector.save(artifact_path)
            detectors[category] = detector
        image_thresholds, pixel_thresholds = _calibrate_detector_thresholds(
            rows, processed_root, detectors, localization_encoder, config
        )
    threshold_name = "exact_first" if exact_first else config.detector
    threshold_root = _threshold_artifact_dir(
        artifact_dir, threshold_name, config.heatmap_refiner
    )
    _write_json(threshold_root / "image_thresholds.json", image_thresholds)
    _write_json(threshold_root / "pixel_thresholds.json", pixel_thresholds)
    if config.heatmap_refiner == "none" and (exact_first or config.detector == "exact_knn"):
        # Keep the historical CLI paths valid for the baseline model.
        _write_json(artifact_dir / "image_thresholds.json", image_thresholds)
        _write_json(artifact_dir / "pixel_thresholds.json", pixel_thresholds)
    test_records = discover_mvtec_test_images(processed_root)
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
                features = localization_features()
                detector_result = detector.predict(features)
                _, heatmap = _make_refined_anomaly_map(
                    detector_result.patch_scores,
                    features.patch_embeddings[0],
                    localization_encoder,
                    config,
                )
                return heatmap

            def padim_runner() -> dict[str, np.ndarray]:
                exact_result = exact_result_holder["result"]
                category = exact_result.get("category")
                detector = localization_detectors["padim"].get(category)
                if detector is None:
                    raise RuntimeError(f"PaDiM artifact is unavailable for category: {category}")
                features = localization_features()
                detector_result = detector.predict(features)
                _, heatmap = _make_refined_anomaly_map(
                    detector_result.patch_scores,
                    features.patch_embeddings[0],
                    localization_encoder,
                    config,
                )
                threshold = float(pixel_thresholds["categories"][category]["threshold"])
                return {
                    "heatmap": heatmap,
                    "binary_mask": postprocess_mask(
                        heatmap, threshold, config.mask_postprocess
                    ),
                }

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
                classifier_encoder,
                classifier,
                reference_artifact,
                mapping,
                calibration,
                detectors,
                image_thresholds,
                pixel_thresholds,
                config,
            )
        if not exact_first and config.detector == "exact_knn":
            result["anomaly_detector"] = "exact_knn"
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
        report_dir / "localization_metrics.csv",
        _localization_metric_rows(anomaly_report),
        _LOCALIZATION_CSV_FIELDS,
    )
    _write_csv(
        report_dir / "per_category_metrics.csv",
        per_category,
        ["category", "support", "correct", "incorrect", "rejected", "accuracy", "rejection_rate"],
    )
    error_rows = _error_analysis_rows(inference_rows)
    _write_csv(
        report_dir / "error_analysis.csv",
        error_rows,
        _ERROR_ANALYSIS_FIELDS,
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
            if item.get("localization_category", item.get("category")) == category
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
            "heatmap_refiner": config.heatmap_refiner,
            "affinity_top_k": config.affinity_top_k,
            "affinity_temperature": config.affinity_temperature,
            "affinity_eta": config.affinity_eta,
            "mask_postprocess": config.mask_postprocess,
            "target_acceptance_rate": config.target_acceptance_rate,
            "artifact_namespace": config.artifact_namespace,
        },
        "preparation": preparation,
        "manifest": manifest_info,
        "classification": classification_report,
        "classification_validation": validation_classification,
        "anomaly": anomaly_report,
        "performance": {
            "inference": _latency_summary(inference_latencies, encoder.device),
        },
        "output_files": {
            "final_report": "final_report.md",
            "summary": "final_summary.json",
            "classification_summary": "classification_test_summary.json",
            "classification_validation": "classification_validation_summary.json",
            "anomaly_metrics": "anomaly_metrics.json",
            "localization_metrics": "localization_metrics.csv",
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
    _write_json(report_dir / "classification_validation_summary.json", validation_classification)
    _write_json(report_dir / "final_summary.json", summary)
    (report_dir / "final_report.md").write_text(_markdown_report(summary, report_dir), encoding="utf-8")
    return summary
