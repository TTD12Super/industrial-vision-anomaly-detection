"""Offline MVTec AD evaluation loop built on top of ``model.inference``.

This module reuses the verified Phase 1 pipeline unchanged (frozen DINOv2
ViT-B/14 patch features + exact Euclidean kNN, k=1, image score = maximum
patch score, bilinear anomaly map) and adds only the evaluation plumbing that
the reference notebook performs in-cell (``notebooks/02_first_model.ipynb``
cells 16, 52, 56, 60, 68, 74, 98):

* per-category test-set scoring in deterministic ``sorted`` order
* image-level and pixel-level AUROC/AP with the notebook's exact semantics
* machine-readable per-category and aggregate artifacts

Deliberately **not** done here (by project constraint):

* no FastAPI / backend / MongoDB interaction
* no feature-bank rebuilding (existing banks are loaded and validated only)
* no threshold selection of any kind; ``threshold_strategy`` is always
  "continuous scores only; no test-label calibration"
* no PRO/AUPRO (unsupported; reported as such)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..inference.config import (
    BANK_CHUNK_SIZE,
    FEATURE_KEY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_AGGREGATION,
    K_NEIGHBORS,
    L2_NORMALIZE_FEATURES,
    MODEL_NAME,
    MODEL_REVISION,
    PROCESSED_SIZE,
    QUERY_CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    default_bank_path,
    get_device,
    validate_category,
)
from .metrics import (
    METRIC_DEFINITIONS,
    image_files,
    image_labels,
    load_masks,
    macro_average,
    pixel_metrics,
    ranking_metrics,
)

logger = logging.getLogger(__name__)

THRESHOLD_STRATEGY = "continuous scores only; no test-label calibration"
AUPRO_STATUS = "not implemented"

METRIC_KEY_ORDER = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap")

CATEGORY_CSV_COLUMNS = (
    "category",
    "train_images",
    "test_images",
    "normal_images",
    "defective_images",
    "image_auroc",
    "image_ap",
    "pixel_auroc",
    "pixel_ap",
    "pipeline_seconds",
    "status",
)


@dataclass
class CategoryResult:
    """Outcome of evaluating one category (success or recorded failure)."""

    category: str
    status: str = "ok"
    train_images: int = 0
    test_images: int = 0
    normal_images: int = 0
    defective_images: int = 0
    image_auroc: float | None = None
    image_ap: float | None = None
    pixel_auroc: float | None = None
    pixel_ap: float | None = None
    pipeline_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    anomaly_maps: np.ndarray | None = None

    @property
    def metrics(self) -> dict[str, float | None]:
        return {
            "image_auroc": self.image_auroc,
            "image_ap": self.image_ap,
            "pixel_auroc": self.pixel_auroc,
            "pixel_ap": self.pixel_ap,
        }


def category_paths(processed_root: str | Path, category: str) -> tuple[list[Path], list[Path]]:
    """Sorted ``train/good`` and ``test`` paths with the notebook's checks.

    Reference: notebook cell 16. Training must contain only ``good``; both
    splits must be non-empty; train and test paths must not overlap.
    """
    processed_root = Path(processed_root)
    category_folder = processed_root / category
    train_folder = category_folder / "train" / "good"
    test_folder = category_folder / "test"

    train_classes = sorted(path.name for path in train_folder.parent.iterdir() if path.is_dir())
    if train_classes != ["good"]:
        raise ValueError(f"{category}: training must contain only the good class.")

    train_paths = image_files(train_folder)
    test_paths = image_files(test_folder)
    if not train_paths or not test_paths:
        raise ValueError(f"{category}: an image split is empty.")
    for path in train_paths:
        if not path.resolve().is_relative_to(train_folder.resolve()):
            raise ValueError(f"{category}: training path leaves train/good: {path}")
    if set(path.resolve() for path in train_paths) & set(path.resolve() for path in test_paths):
        raise ValueError(f"{category}: train/test paths overlap.")
    return train_paths, test_paths


def load_category_bank(
    category: str,
    train_paths: Sequence[Path],
    bank_dir: str | Path,
    extractor: Any,
    processed_root: str | Path,
) -> Any:
    """Load and validate an existing bank; never rebuilds one.

    Every provenance field written by ``build_feature_bank`` is checked
    against the current configuration, the loaded extractor, and the actual
    ``train/good`` files, so a bank built under different settings or from a
    different training set is rejected instead of silently reused.
    """
    from ..inference.feature_bank import load_feature_bank

    bank_path = default_bank_path(category, bank_dir)
    if not bank_path.is_file():
        raise FileNotFoundError(
            f"Feature bank for category '{category}' is not available: {bank_path}."
        )
    bank, metadata = load_feature_bank(bank_path)
    expected_train_paths = [
        str(Path(path).relative_to(Path(processed_root))) for path in train_paths
    ]
    expected: dict[str, Any] = {
        "category": category,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "feature_key": FEATURE_KEY,
        "image_size": list(PROCESSED_SIZE),
        "patch_grid": [int(extractor.patch_grid[0]), int(extractor.patch_grid[1])],
        "padding": [int(extractor.pad_right), int(extractor.pad_bottom)],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "l2_normalization": L2_NORMALIZE_FEATURES,
        "sampling": "none; all train/good patches",
        "train_paths": expected_train_paths,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Feature bank for '{category}' has {key}={metadata.get(key)!r}, expected {value!r}; "
                "refusing to evaluate with incompatible settings."
            )
    embedding_dimension = int(bank.shape[1])
    if metadata.get("embedding_dimension") != embedding_dimension:
        raise ValueError(
            f"Feature bank for '{category}' declares embedding dimension "
            f"{metadata.get('embedding_dimension')} but holds {embedding_dimension}."
        )
    expected_rows = len(train_paths) * int(extractor.patch_count)
    if tuple(bank.shape) != (expected_rows, embedding_dimension):
        raise ValueError(
            f"Feature bank for '{category}' has shape {tuple(bank.shape)}, "
            f"expected {(expected_rows, embedding_dimension)}."
        )
    logger.info(
        "Loaded existing bank for %s: %s from %s (not rebuilt).",
        category,
        tuple(bank.shape),
        bank_path,
    )
    return bank, metadata


def evaluate_category(
    category: str,
    *,
    processed_root: str | Path,
    bank_dir: str | Path,
    extractor: Any,
    batch_size: int,
    limit: int | None = None,
    save_maps: bool = False,
    compute_pixel_metrics: bool = True,
) -> CategoryResult:
    """Score one category's test set and compute the reference metrics.

    Any exception is captured on the returned :class:`CategoryResult` with
    ``status="failed"`` so a multi-category run can continue and still report
    the exact failure.
    """
    import torch

    from ..inference.anomaly_map import create_anomaly_maps
    from ..inference.knn import nearest_neighbor_scores
    from ..inference.preprocessing import image_tensor

    validate_category(category)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    started = time.perf_counter()
    result = CategoryResult(category=category)
    try:
        processed_root = Path(processed_root)
        train_paths, test_paths = category_paths(processed_root, category)
        if limit is not None:
            if limit < 1:
                raise ValueError(f"limit must be positive, got {limit}.")
            test_paths = test_paths[:limit]

        bank, _ = load_category_bank(
            category, train_paths, bank_dir, extractor, processed_root
        )

        patch_score_batches: list[Any] = []
        per_image_seconds: list[float] = []
        for start in range(0, len(test_paths), batch_size):
            chunk = test_paths[start : start + batch_size]
            batch_started = time.perf_counter()
            batch = torch.stack([image_tensor(path) for path in chunk])
            features = extractor.extract_batch(batch)  # (B, P, D) on CPU
            flat_scores = nearest_neighbor_scores(
                features.reshape(-1, features.shape[-1]), bank, k=K_NEIGHBORS
            )
            patch_score_batches.append(
                flat_scores.reshape(len(chunk), extractor.patch_count).cpu()
            )
            elapsed = time.perf_counter() - batch_started
            per_image_seconds.extend([elapsed / len(chunk)] * len(chunk))

        patch_scores = torch.cat(patch_score_batches)  # (N, 361)
        scores = patch_scores.max(dim=1).values.numpy()
        maps = create_anomaly_maps(
            patch_scores, extractor.patch_grid, extractor.padded_size, PROCESSED_SIZE
        )

        labels = image_labels(test_paths)
        masks = (
            load_masks(category, test_paths, processed_root)
            if compute_pixel_metrics
            else None
        )
        metrics = ranking_metrics(labels, scores)
        if compute_pixel_metrics:
            pixel_result = pixel_metrics(masks, maps)
        else:
            pixel_result = {"auroc": None, "average_precision": None}

        result.train_images = len(train_paths)
        result.test_images = len(test_paths)
        result.normal_images = int((~labels).sum())
        result.defective_images = int(labels.sum())
        result.image_auroc = metrics["auroc"]
        result.image_ap = metrics["average_precision"]
        result.pixel_auroc = pixel_result["auroc"]
        result.pixel_ap = pixel_result["average_precision"]
        result.rows = [
            {
                "category": category,
                "image_path": str(path.relative_to(processed_root)),
                "defect_type": path.parent.name,
                "is_anomaly": int(label),
                "anomaly_score": float(score),
                "processing_time_s": round(float(seconds), 6),
            }
            for path, label, score, seconds in zip(test_paths, labels, scores, per_image_seconds)
        ]
        if save_maps:
            result.anomaly_maps = maps
        result.pipeline_seconds = time.perf_counter() - started
        logger.info(
            "%s: %d test images (normal=%d defective=%d) in %.1fs; image_auroc=%s pixel_auroc=%s.",
            category,
            result.test_images,
            result.normal_images,
            result.defective_images,
            result.pipeline_seconds,
            _fmt(result.image_auroc),
            _fmt(result.pixel_auroc),
        )
    except Exception as exc:  # recorded, never hidden
        result.status = "failed"
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.pipeline_seconds = time.perf_counter() - started
        logger.error("%s: evaluation failed (%s: %s).", category, result.error_type, result.error_message)
    return result


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def configuration_hash(settings: dict[str, Any]) -> str:
    """Stable hash of every setting that can change reported metrics."""
    payload = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_config(
    *,
    processed_root: Path,
    bank_dir: Path,
    run_dir: Path,
    extractor: Any,
    categories: Sequence[str],
    batch_size: int,
    limit: int | None,
    save_maps: bool,
    device: str,
) -> dict[str, Any]:
    """Reproducibility payload written to ``run_config.json``."""
    core = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "feature_key": FEATURE_KEY,
        "image_size": list(PROCESSED_SIZE),
        "patch_grid": [int(extractor.patch_grid[0]), int(extractor.patch_grid[1])],
        "padding": [int(extractor.pad_right), int(extractor.pad_bottom)],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "l2_normalization": L2_NORMALIZE_FEATURES,
        "k_neighbors": K_NEIGHBORS,
        "image_aggregation": IMAGE_AGGREGATION,
        "map_method": "bilinear to padded size, then crop to processed image size",
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "bank_chunk_size": BANK_CHUNK_SIZE,
        "batch_size": int(batch_size),
        "limit": None if limit is None else int(limit),
        "save_anomaly_maps": bool(save_maps),
        "threshold_strategy": THRESHOLD_STRATEGY,
    }
    try:
        import platform

        import numpy as _np
        import sklearn
        import torch as _torch

        packages = {
            "python": platform.python_version(),
            "torch": getattr(_torch, "__version__", "unknown"),
            "numpy": _np.__version__,
            "scikit-learn": sklearn.__version__,
        }
    except Exception:  # pragma: no cover - diagnostics only
        packages = {}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_root": str(processed_root),
        "feature_bank_root": str(bank_dir),
        "categories": list(categories),
        "device": device,
        "subset_run": limit is not None,
        "save_anomaly_maps": bool(save_maps),
        "packages": packages,
        "metric_definitions": METRIC_DEFINITIONS,
        "pixel_metric_population": (
            "all test pixels, including normal-image all-zero masks (reference convention)"
        ),
        "aupro": AUPRO_STATUS,
        **core,
        "config_hash": configuration_hash({**core, "categories": list(categories)}),
    }


def write_category_outputs(run_dir: Path, result: CategoryResult, save_maps: bool) -> None:
    """Write ``image_scores.csv``, ``metrics.json`` and optional maps."""
    folder = run_dir / result.category
    folder.mkdir(parents=True, exist_ok=True)
    if result.rows:
        with (folder / "image_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "category",
                    "image_path",
                    "defect_type",
                    "is_anomaly",
                    "anomaly_score",
                    "processing_time_s",
                ],
            )
            writer.writeheader()
            writer.writerows(result.rows)
    payload = {
        "category": result.category,
        "status": result.status,
        "train_images": result.train_images,
        "test_images": result.test_images,
        "normal_images": result.normal_images,
        "defective_images": result.defective_images,
        "pipeline_seconds": round(result.pipeline_seconds, 3),
        "metrics": result.metrics,
        "metric_definitions": METRIC_DEFINITIONS,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "aupro": AUPRO_STATUS,
    }
    if result.status == "failed":
        payload["error"] = {
            "type": result.error_type,
            "message": result.error_message,
        }
    (folder / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if save_maps and result.anomaly_maps is not None:
        np.savez_compressed(
            folder / "anomaly_maps.npz",
            image_paths=np.array([row["image_path"] for row in result.rows]),
            anomaly_maps=result.anomaly_maps,
        )


def write_aggregate_outputs(run_dir: Path, results: Sequence[CategoryResult]) -> None:
    """Write ``category_metrics.csv``, ``macro_metrics.csv``, ``failures.json``."""
    with (run_dir / "category_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CATEGORY_CSV_COLUMNS))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "category": result.category,
                    "train_images": result.train_images,
                    "test_images": result.test_images,
                    "normal_images": result.normal_images,
                    "defective_images": result.defective_images,
                    "image_auroc": _csv(result.image_auroc),
                    "image_ap": _csv(result.image_ap),
                    "pixel_auroc": _csv(result.pixel_auroc),
                    "pixel_ap": _csv(result.pixel_ap),
                    "pipeline_seconds": round(result.pipeline_seconds, 3),
                    "status": result.status,
                }
            )

    with (run_dir / "macro_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "macro_average", "categories_included"])
        writer.writeheader()
        for key in METRIC_KEY_ORDER:
            mean, count = macro_average([getattr(result, key) for result in results])
            writer.writerow(
                {"metric": key, "macro_average": _csv(mean), "categories_included": count}
            )

    failures = [
        {
            "category": result.category,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
        for result in results
        if result.status != "ok"
    ]
    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")


def _csv(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def run_evaluation(
    *,
    categories: Sequence[str],
    processed_root: str | Path,
    bank_dir: str | Path,
    run_root: str | Path,
    batch_size: int = 8,
    limit: int | None = None,
    save_maps: bool = False,
    device: str | None = None,
) -> tuple[Path, list[CategoryResult]]:
    """Evaluate *categories* into a brand-new ``eval_<timestamp>`` run folder.

    The run directory is created with ``exist_ok=False`` so an existing run is
    never overwritten. Returns the run directory and per-category results.
    """
    from ..inference.feature_extractor import DINOv2FeatureExtractor

    processed_root = Path(processed_root)
    bank_dir = Path(bank_dir)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = run_root / f"eval_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_device = str(get_device(device))
    logger.info("Creating evaluation run %s (device=%s).", run_dir, resolved_device)
    extractor = DINOv2FeatureExtractor(device=device)

    run_config = build_run_config(
        processed_root=processed_root,
        bank_dir=bank_dir,
        run_dir=run_dir,
        extractor=extractor,
        categories=categories,
        batch_size=batch_size,
        limit=limit,
        save_maps=save_maps,
        device=resolved_device,
    )
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    results: list[CategoryResult] = []
    for category in categories:
        result = evaluate_category(
            category,
            processed_root=processed_root,
            bank_dir=bank_dir,
            extractor=extractor,
            batch_size=batch_size,
            limit=limit,
            save_maps=save_maps,
        )
        write_category_outputs(run_dir, result, save_maps)
        results.append(result)

    write_aggregate_outputs(run_dir, results)
    logger.info("Evaluation run complete: %s", run_dir)
    return run_dir, results
