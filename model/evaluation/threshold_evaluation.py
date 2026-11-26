"""Offline evaluation of the frozen Phase 4.7C thresholds on the MVTec AD test set.

This module applies already-calibrated, per-category thresholds to test-image
scores produced by the **unchanged** production scoring path. It is an offline
measurement pass only:

* thresholds are loaded from the Phase 4.7C calibration run and never changed;
  the only decision rule is the documented strict comparison
  ``predicted_defect = (image_score > threshold)`` (equality predicts normal)
* test labels (directory names) are used strictly *after* predictions are
  computed and can never influence a threshold or a score
* feature banks are verified against the Phase 4.7D baseline manifest before
  and after scoring, failing closed on any mismatch (never repaired here)
* pixel-level metrics are intentionally omitted: the calibrated threshold is
  image-level (maximum patch score); no pixel-level threshold was calibrated
* measured test FPR/TPR are empirical statistics on a finite held-out sample;
  they do **not** prove the Phase 4.7B marginal guarantee (see the calibration
  run's ``assumptions_and_proof.md``)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass
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
    validate_category,
)
from .calibration import DEFAULT_ALPHA, THRESHOLD_FORMULA, sha256_file
from .evaluate import configuration_hash
from .metrics import macro_average

logger = logging.getLogger(__name__)

DECISION_RULE_APPLIED = (
    "predicted_defect = (image_score > threshold); strict greater-than; "
    "a score exactly equal to the threshold is predicted normal"
)
PIXEL_METRICS_STATUS = (
    "omitted: the calibrated threshold is image-level (max patch score); "
    "no pixel-level threshold was calibrated in Phase 4.7C"
)
GUARANTEE_NOTE = (
    "Measured test-set rates are empirical statistics on a finite held-out sample "
    "and do not prove the Phase 4.7B marginal guarantee; see the calibration run's "
    "assumptions_and_proof.md for the bound, its assumptions, and its limitations."
)
RANKING_METRIC_NOTE = (
    "Phase 4.6 image AUROC/AP are ranking metrics over continuous scores; the "
    "threshold-based results below are a separate evaluation and must not be "
    "conflated with them."
)
LABELS_POLICY = (
    "Test labels (directory names) are used only to score frozen predictions "
    "after they are generated; they never affect a threshold or a score."
)

THRESHOLD_METRIC_DEFINITIONS = {
    "fpr": "FP / (FP + TN) over normal test images; None when FP + TN == 0.",
    "recall": "TP / (TP + FN) over defective test images (true-positive rate); None when TP + FN == 0.",
    "specificity": "TN / (TN + FP) over normal test images (= 1 - fpr); None when TN + FP == 0.",
    "precision": "TP / (TP + FP) among images predicted defective; None (undefined) when TP + FP == 0.",
    "f1": (
        "2 * precision * recall / (precision + recall); None when precision or "
        "recall is undefined, or when precision + recall == 0."
    ),
    "macro_average": (
        "Unweighted mean over categories of the per-category metric; categories "
        "where the metric is undefined are excluded and counted, never zero-filled."
    ),
    "micro_average": (
        "Metric computed once from TP/TN/FP/FN pooled across all categories; "
        "not a macro average."
    ),
}

THRESHOLD_METRICS_COLUMNS = (
    "category",
    "threshold",
    "alpha",
    "calibration_train_images",
    "calibration_index_k",
    "n_normal",
    "n_defective",
    "predicted_normal_when_normal",
    "predicted_defect_when_normal",
    "predicted_normal_when_defective",
    "predicted_defect_when_defective",
    "fpr",
    "recall",
    "specificity",
    "precision",
    "f1",
    "score_min",
    "score_max",
    "score_mean",
    "score_median",
    "score_std",
    "normal_score_min",
    "normal_score_max",
    "defective_score_min",
    "defective_score_max",
    "pipeline_seconds",
    "scoring_consistency_max_abs_diff_vs_phase_4_6",
    "status",
)

CONFUSION_COLUMNS = (
    "category",
    "threshold",
    "n_normal",
    "n_defective",
    "tp",
    "tn",
    "fp",
    "fn",
)

AGGREGATE_METRIC_KEYS = ("fpr", "recall", "specificity", "precision", "f1")


class ThresholdDataError(ValueError):
    """Frozen threshold inputs are missing, malformed, duplicated, or non-finite."""


class BaselineIntegrityError(RuntimeError):
    """Feature-bank baseline verification failed; evaluation must not proceed."""


@dataclass(frozen=True)
class ThresholdEntry:
    """One frozen Phase 4.7C category threshold (never recomputed here)."""

    category: str
    threshold: float
    alpha: float
    index_k: int
    train_images: int


@dataclass
class CategoryThresholdResult:
    """Confusion counts, metrics, and score summaries for one category."""

    category: str
    threshold_entry: ThresholdEntry
    n_normal: int
    n_defective: int
    tp: int
    tn: int
    fp: int
    fn: int
    metrics: dict[str, float | None]
    score_summary: dict[str, Any]
    normal_score_summary: dict[str, Any]
    defective_score_summary: dict[str, Any]
    predicted: np.ndarray
    pipeline_seconds: float = 0.0
    scoring_consistency: dict[str, Any] | None = None
    status: str = "ok"


def load_frozen_thresholds(
    calibration_run: str | Path, expected_categories: Sequence[str]
) -> dict[str, ThresholdEntry]:
    """Load the Phase 4.7C thresholds without ever recomputing them.

    Raises :class:`ThresholdDataError` for a missing/unreadable CSV, missing
    columns, duplicate or missing categories, non-finite thresholds, an alpha
    other than the pre-registered ``DEFAULT_ALPHA``, infeasible rows, or an
    invalid index.
    """
    calibration_run = Path(calibration_run)
    csv_path = calibration_run / "thresholds.csv"
    if not csv_path.is_file():
        raise ThresholdDataError(f"thresholds.csv not found in {calibration_run}.")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        required = {"category", "train_images", "alpha", "index_k", "threshold", "alpha_feasible"}
        missing = sorted(required - header)
        if missing:
            raise ThresholdDataError(f"thresholds.csv is missing columns: {missing}.")
        rows = list(reader)

    entries: dict[str, ThresholdEntry] = {}
    for row in rows:
        category = row["category"]
        if category in entries:
            raise ThresholdDataError(f"duplicate threshold row for category '{category}'.")
        try:
            threshold = float(row["threshold"])
            alpha = float(row["alpha"])
            index_k = int(row["index_k"])
            train_images = int(row["train_images"])
        except (TypeError, ValueError) as exc:
            raise ThresholdDataError(f"{category}: unparsable threshold row ({exc}).") from exc
        if not math.isfinite(threshold):
            raise ThresholdDataError(
                f"{category}: threshold must be finite, got {row['threshold']!r}."
            )
        if alpha != DEFAULT_ALPHA:
            raise ThresholdDataError(
                f"{category}: alpha={alpha!r} does not match the pre-registered "
                f"alpha={DEFAULT_ALPHA!r}; refusing to evaluate."
            )
        if row["alpha_feasible"] != "true":
            raise ThresholdDataError(
                f"{category}: calibration row is not alpha-feasible "
                f"(alpha_feasible={row['alpha_feasible']!r})."
            )
        if train_images < 1:
            raise ThresholdDataError(f"{category}: train_images={train_images} is invalid.")
        if not (1 <= index_k <= train_images):
            raise ThresholdDataError(
                f"{category}: index_k={index_k} outside 1..{train_images}."
            )
        entries[category] = ThresholdEntry(
            category=category,
            threshold=threshold,
            alpha=alpha,
            index_k=index_k,
            train_images=train_images,
        )

    expected = list(expected_categories)
    missing_categories = sorted(set(expected) - set(entries))
    extra_categories = sorted(set(entries) - set(expected))
    if missing_categories:
        raise ThresholdDataError(f"missing thresholds for: {missing_categories}.")
    if extra_categories:
        raise ThresholdDataError(f"unexpected threshold categories: {extra_categories}.")
    if len(entries) != len(expected):
        raise ThresholdDataError(
            f"expected {len(expected)} thresholds, found {len(entries)}."
        )
    logger.info(
        "Loaded %d frozen thresholds (alpha=%s) from %s.",
        len(entries),
        DEFAULT_ALPHA,
        csv_path,
    )
    return {category: entries[category] for category in sorted(entries)}


def verify_bank_baseline(
    bank_dir: str | Path, manifest_path: str | Path, expected_categories: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Verify every feature bank against the Phase 4.7D manifest (fail closed).

    Raises :class:`BaselineIntegrityError` on an unreadable/malformed manifest,
    an unsupported schema, missing or duplicate categories, an unexpected path,
    a missing bank file, or any SHA-256/size mismatch. Returns the verified
    per-category ``{path, sha256, size_bytes}`` mapping.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise BaselineIntegrityError(f"bank baseline manifest not found: {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BaselineIntegrityError(f"manifest is not valid JSON: {exc}.") from exc
    if not isinstance(manifest, dict):
        raise BaselineIntegrityError("manifest root must be a JSON object.")
    if manifest.get("schema_version") != "1.0":
        raise BaselineIntegrityError(
            f"unsupported manifest schema_version: {manifest.get('schema_version')!r}."
        )
    if manifest.get("manifest_type") != "feature_bank_sha256_baseline":
        raise BaselineIntegrityError(
            f"unexpected manifest_type: {manifest.get('manifest_type')!r}."
        )
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        raise BaselineIntegrityError("manifest entries must be a list.")

    expected = set(expected_categories)
    problems: list[str] = []
    by_category: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        category = entry.get("category") if isinstance(entry, dict) else None
        if not isinstance(category, str):
            problems.append("manifest entry without a string category.")
            continue
        if category in by_category:
            problems.append(f"duplicate manifest entry for '{category}'.")
            continue
        by_category[category] = entry
    missing = sorted(expected - set(by_category))
    extra = sorted(set(by_category) - expected)
    if missing:
        problems.append(f"manifest missing categories: {missing}.")
    if extra:
        problems.append(f"manifest has unexpected categories: {extra}.")

    verified: dict[str, dict[str, Any]] = {}
    for category in sorted(expected & set(by_category)):
        entry = by_category[category]
        expected_rel = f"artifacts/dinov2_knn/banks/{category}/feature_bank.pt"
        if entry.get("path") != expected_rel:
            problems.append(
                f"{category}: manifest path {entry.get('path')!r} != {expected_rel!r}."
            )
            continue
        bank_path = Path(bank_dir) / category / "feature_bank.pt"
        if not bank_path.is_file():
            problems.append(f"{category}: bank file missing: {bank_path}.")
            continue
        size = bank_path.stat().st_size
        digest = sha256_file(bank_path)
        if size != entry.get("size_bytes"):
            problems.append(
                f"{category}: size {size} != manifest {entry.get('size_bytes')}."
            )
        if digest != entry.get("sha256"):
            problems.append(
                f"{category}: sha256 {digest} != manifest {entry.get('sha256')}."
            )
        verified[category] = {"path": entry["path"], "sha256": digest, "size_bytes": size}

    if problems:
        raise BaselineIntegrityError(
            "bank baseline verification failed (fail closed): " + "; ".join(problems)
        )
    logger.info("Verified %d feature banks against %s.", len(verified), manifest_path)
    return verified


def predict_defect(image_scores: Sequence[float], threshold: float) -> np.ndarray:
    """Strict frozen decision rule: ``image_score > threshold`` (never ``>=``)."""
    scores = np.asarray(image_scores, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise ThresholdDataError("image scores contain non-finite values.")
    if not math.isfinite(threshold):
        raise ThresholdDataError(f"threshold must be finite, got {threshold!r}.")
    return scores > threshold


def confusion_counts(
    is_defective: Sequence[bool], predicted_defect: Sequence[bool]
) -> tuple[int, int, int, int]:
    """Return ``(tp, tn, fp, fn)`` for boolean label/prediction arrays."""
    labels = np.asarray(is_defective, dtype=bool).reshape(-1)
    predicted = np.asarray(predicted_defect, dtype=bool).reshape(-1)
    if labels.shape != predicted.shape:
        raise ThresholdDataError(
            f"labels and predictions differ in shape: {labels.shape} vs {predicted.shape}."
        )
    tp = int((labels & predicted).sum())
    tn = int((~labels & ~predicted).sum())
    fp = int((~labels & predicted).sum())
    fn = int((labels & ~predicted).sum())
    return tp, tn, fp, fn


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def threshold_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | None]:
    """FPR, recall, specificity, precision, F1 from confusion counts.

    Undefined metrics are returned as ``None`` and are never zero-filled:
    precision when ``TP + FP == 0``, F1 when precision/recall are undefined or
    their sum is zero, and the population metrics when their denominator is zero.
    """
    fpr = _ratio(fp, fp + tn)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "fpr": fpr,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def score_summary(values: Sequence[float]) -> dict[str, Any]:
    """Basic distribution summary of image scores (None-safe for empty input)."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    if not np.isfinite(array).all():
        raise ThresholdDataError("score summary received non-finite values.")
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def evaluate_category_thresholds(
    category: str,
    image_scores: Sequence[float],
    is_defective: Sequence[bool],
    threshold_entry: ThresholdEntry,
    pipeline_seconds: float = 0.0,
) -> CategoryThresholdResult:
    """Apply one frozen threshold to one category's test scores.

    Predictions are computed from the scores and the frozen threshold **before**
    the labels are touched, so labels cannot influence predictions or the
    threshold (they only determine how predictions are counted afterwards).
    """
    validate_category(category)
    if threshold_entry.category != category:
        raise ThresholdDataError(
            f"threshold for '{threshold_entry.category}' applied to '{category}'."
        )
    scores = np.asarray(image_scores, dtype=float).reshape(-1)
    labels = np.asarray(is_defective, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ThresholdDataError(
            f"scores and labels differ in shape: {scores.shape} vs {labels.shape}."
        )

    predicted = predict_defect(scores, threshold_entry.threshold)  # labels not involved
    tp, tn, fp, fn = confusion_counts(labels, predicted)

    return CategoryThresholdResult(
        category=category,
        threshold_entry=threshold_entry,
        n_normal=int((~labels).sum()),
        n_defective=int(labels.sum()),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        metrics=threshold_metrics(tp, tn, fp, fn),
        score_summary=score_summary(scores),
        normal_score_summary=score_summary(scores[~labels]),
        defective_score_summary=score_summary(scores[labels]),
        predicted=predicted,
        pipeline_seconds=float(pipeline_seconds),
    )


def macro_threshold_metrics(
    results: Sequence[CategoryThresholdResult],
) -> dict[str, dict[str, Any]]:
    """Unweighted per-metric macro averages with explicit undefined tracking."""
    macro: dict[str, dict[str, Any]] = {}
    for key in AGGREGATE_METRIC_KEYS:
        values = [result.metrics[key] for result in results]
        mean, count = macro_average(values)
        undefined = sorted(
            result.category for result in results if result.metrics[key] is None
        )
        macro[key] = {
            "value": mean,
            "categories_included": count,
            "categories_undefined": undefined,
        }
    return macro


def micro_threshold_metrics(
    results: Sequence[CategoryThresholdResult],
) -> dict[str, Any]:
    """Micro averages from TP/TN/FP/FN pooled across categories (not macro)."""
    pooled = {
        "tp": sum(result.tp for result in results),
        "tn": sum(result.tn for result in results),
        "fp": sum(result.fp for result in results),
        "fn": sum(result.fn for result in results),
    }
    metrics = threshold_metrics(pooled["tp"], pooled["tn"], pooled["fp"], pooled["fn"])
    return {**metrics, "pooled_counts": pooled}


def scoring_consistency(
    category: str,
    rows: Sequence[dict[str, Any]],
    reference_eval_dir: str | Path | None,
) -> dict[str, Any] | None:
    """Compare re-scored image scores with the Phase 4.6 stored scores (read-only).

    Returns ``None`` when no reference is configured or the reference file is
    absent. This is a consistency *observation* only: it never feeds back into
    thresholds, predictions, or metrics.
    """
    if reference_eval_dir is None:
        return None
    reference_path = Path(reference_eval_dir) / category / "image_scores.csv"
    if not reference_path.is_file():
        return None
    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference = {
            row["image_path"]: float(row["anomaly_score"]) for row in csv.DictReader(handle)
        }
    current = {row["image_path"]: float(row["anomaly_score"]) for row in rows}
    if set(reference) != set(current):
        return {
            "status": "path_set_differs",
            "reference_images": len(reference),
            "current_images": len(current),
            "max_abs_diff": None,
        }
    if not current:
        return {"status": "compared", "images": 0, "max_abs_diff": 0.0}
    max_abs_diff = max(
        abs(reference[key] - current[key]) for key in current
    )
    return {"status": "compared", "images": len(current), "max_abs_diff": max_abs_diff}


def _source_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def build_threshold_eval_config(
    *,
    run_dir: Path,
    calibration_run: Path,
    bank_baseline_manifest: Path,
    processed_root: Path,
    bank_dir: Path,
    extractor: Any,
    batch_size: int,
    device: str,
    thresholds: dict[str, ThresholdEntry],
    reference_eval_dir: Path | None,
) -> dict[str, Any]:
    """Reproducibility payload for ``config.json`` (written exactly once)."""
    thresholds_csv = calibration_run / "thresholds.csv"
    calibration_config = calibration_run / "calibration_config.json"
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
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "bank_chunk_size": BANK_CHUNK_SIZE,
        "batch_size": int(batch_size),
        "alpha": DEFAULT_ALPHA,
        "decision_rule": DECISION_RULE_APPLIED,
        "threshold_formula": THRESHOLD_FORMULA,
        "pixel_metrics": PIXEL_METRICS_STATUS,
        "labels_policy": LABELS_POLICY,
        "scoring_path": (
            "model.evaluation.evaluate.evaluate_category (unchanged production "
            "scoring path; compute_pixel_metrics=False, save_maps=False)"
        ),
    }
    try:
        import platform

        import sklearn
        import torch

        packages = {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        }
    except Exception:  # pragma: no cover - diagnostics only
        packages = {}
    return {
        "run_type": "threshold_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_root": str(processed_root),
        "feature_bank_root": str(bank_dir),
        "calibration_run": str(calibration_run),
        "bank_baseline_manifest": str(bank_baseline_manifest),
        "inputs": {
            "thresholds_csv": {
                "path": str(thresholds_csv),
                "sha256": sha256_file(thresholds_csv),
            },
            "calibration_config": {
                "path": str(calibration_config),
                "sha256": sha256_file(calibration_config),
            },
            "bank_baseline_manifest": {
                "path": str(bank_baseline_manifest),
                "sha256": sha256_file(bank_baseline_manifest),
            },
        },
        "thresholds": {
            category: {
                "threshold": entry.threshold,
                "alpha": entry.alpha,
                "index_k": entry.index_k,
                "calibration_train_images": entry.train_images,
            }
            for category, entry in thresholds.items()
        },
        "bank_integrity": {
            "verified_against_manifest_before_scoring": True,
            "verified_against_manifest_after_scoring": True,
            "fail_closed_on_mismatch": True,
        },
        "reference_eval_dir": None if reference_eval_dir is None else str(reference_eval_dir),
        "scoring_consistency_note": (
            "Re-scored image scores are compared read-only against the Phase 4.6 "
            "stored scores; the comparison never influences thresholds or metrics."
        ),
        "metric_definitions": THRESHOLD_METRIC_DEFINITIONS,
        "guarantee_note": GUARANTEE_NOTE,
        "ranking_metric_note": RANKING_METRIC_NOTE,
        "pixel_metrics": PIXEL_METRICS_STATUS,
        "packages": packages,
        "source_revision": _source_revision(),
        "device": device,
        **core,
        "config_hash": configuration_hash({**core, "categories": sorted(thresholds)}),
    }


def _write_threshold_metrics_csv(run_dir: Path, results: Sequence[CategoryThresholdResult]) -> None:
    with (run_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(THRESHOLD_METRICS_COLUMNS))
        writer.writeheader()
        for result in results:
            entry = result.threshold_entry
            consistency = result.scoring_consistency or {}
            writer.writerow(
                {
                    "category": result.category,
                    "threshold": repr(entry.threshold),
                    "alpha": repr(entry.alpha),
                    "calibration_train_images": entry.train_images,
                    "calibration_index_k": entry.index_k,
                    "n_normal": result.n_normal,
                    "n_defective": result.n_defective,
                    "predicted_normal_when_normal": result.tn,
                    "predicted_defect_when_normal": result.fp,
                    "predicted_normal_when_defective": result.fn,
                    "predicted_defect_when_defective": result.tp,
                    "fpr": _csv_value(result.metrics["fpr"]),
                    "recall": _csv_value(result.metrics["recall"]),
                    "specificity": _csv_value(result.metrics["specificity"]),
                    "precision": _csv_value(result.metrics["precision"]),
                    "f1": _csv_value(result.metrics["f1"]),
                    "score_min": _csv_value(result.score_summary["min"]),
                    "score_max": _csv_value(result.score_summary["max"]),
                    "score_mean": _csv_value(result.score_summary["mean"]),
                    "score_median": _csv_value(result.score_summary["median"]),
                    "score_std": _csv_value(result.score_summary["std"]),
                    "normal_score_min": _csv_value(result.normal_score_summary["min"]),
                    "normal_score_max": _csv_value(result.normal_score_summary["max"]),
                    "defective_score_min": _csv_value(result.defective_score_summary["min"]),
                    "defective_score_max": _csv_value(result.defective_score_summary["max"]),
                    "pipeline_seconds": round(result.pipeline_seconds, 3),
                    "scoring_consistency_max_abs_diff_vs_phase_4_6": (
                        ""
                        if consistency.get("max_abs_diff") is None
                        else repr(float(consistency["max_abs_diff"]))
                    ),
                    "status": result.status,
                }
            )


def _write_confusion_csv(run_dir: Path, results: Sequence[CategoryThresholdResult]) -> None:
    with (run_dir / "confusion_matrices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONFUSION_COLUMNS))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "category": result.category,
                    "threshold": repr(result.threshold_entry.threshold),
                    "n_normal": result.n_normal,
                    "n_defective": result.n_defective,
                    "tp": result.tp,
                    "tn": result.tn,
                    "fp": result.fp,
                    "fn": result.fn,
                }
            )


def build_summary(
    results: Sequence[CategoryThresholdResult],
    failures: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate metrics, totals, distributions, and undefined-metric tracking."""
    macro = macro_threshold_metrics(results)
    micro = micro_threshold_metrics(results)
    totals = {
        "images": sum(r.n_normal + r.n_defective for r in results),
        "normal": sum(r.n_normal for r in results),
        "defective": sum(r.n_defective for r in results),
        "tp": sum(r.tp for r in results),
        "tn": sum(r.tn for r in results),
        "fp": sum(r.fp for r in results),
        "fn": sum(r.fn for r in results),
    }
    undefined = {
        key: sorted(
            result.category for result in results if result.metrics[key] is None
        )
        for key in AGGREGATE_METRIC_KEYS
    }
    return {
        "calibration_run": None,
        "alpha": DEFAULT_ALPHA,
        "decision_rule": DECISION_RULE_APPLIED,
        "categories_evaluated": len(results),
        "categories_failed": len(failures),
        "test_totals": totals,
        "macro": {
            "definition": THRESHOLD_METRIC_DEFINITIONS["macro_average"],
            **macro,
        },
        "micro": {
            "definition": THRESHOLD_METRIC_DEFINITIONS["micro_average"],
            **micro,
        },
        "undefined_metrics_by_category": undefined,
        "metric_definitions": THRESHOLD_METRIC_DEFINITIONS,
        "per_category": [
            {
                "category": result.category,
                "threshold": result.threshold_entry.threshold,
                "n_normal": result.n_normal,
                "n_defective": result.n_defective,
                "tp": result.tp,
                "tn": result.tn,
                "fp": result.fp,
                "fn": result.fn,
                "fpr": result.metrics["fpr"],
                "recall": result.metrics["recall"],
                "specificity": result.metrics["specificity"],
                "precision": result.metrics["precision"],
                "f1": result.metrics["f1"],
                "score_summary": result.score_summary,
                "normal_score_summary": result.normal_score_summary,
                "defective_score_summary": result.defective_score_summary,
                "scoring_consistency_vs_phase_4_6": result.scoring_consistency,
            }
            for result in results
        ],
        "pixel_metrics": PIXEL_METRICS_STATUS,
        "guarantee_note": GUARANTEE_NOTE,
        "ranking_metric_note": RANKING_METRIC_NOTE,
        "labels_policy": LABELS_POLICY,
    }


def write_report(
    run_dir: Path,
    results: Sequence[CategoryThresholdResult],
    summary: dict[str, Any],
    failures: Sequence[dict[str, Any]],
) -> None:
    """Human-readable ``report.md`` with per-category, macro, and micro tables."""
    lines: list[str] = []
    lines.append("# Phase 4.7E threshold evaluation report")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Calibration run: `{summary['calibration_run']}`")
    lines.append(f"- Bank baseline manifest: `{summary['bank_baseline_manifest']}`")
    lines.append(f"- Alpha (pre-registered): `{summary['alpha']}`")
    lines.append(f"- Decision rule: `{summary['decision_rule']}`")
    lines.append(f"- Created (UTC): {summary['created_at_utc']}")
    lines.append("")
    lines.append("## Per-category threshold metrics")
    lines.append("")
    lines.append(
        "| category | threshold | normal | defective | TN | FP | FN | TP | "
        "FPR | recall | specificity | precision | F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        m = result.metrics
        lines.append(
            f"| {result.category} | {result.threshold_entry.threshold:.6f} "
            f"| {result.n_normal} | {result.n_defective} "
            f"| {result.tn} | {result.fp} | {result.fn} | {result.tp} "
            f"| {_fmt(m['fpr'])} | {_fmt(m['recall'])} | {_fmt(m['specificity'])} "
            f"| {_fmt(m['precision'])} | {_fmt(m['f1'])} |"
        )
    lines.append("")
    totals = summary["test_totals"]
    lines.append(
        f"Totals: images={totals['images']} normal={totals['normal']} "
        f"defective={totals['defective']} TP={totals['tp']} TN={totals['tn']} "
        f"FP={totals['fp']} FN={totals['fn']}"
    )
    lines.append("")
    lines.append("## Score distributions (image scores)")
    lines.append("")
    lines.append("| category | min | max | mean | median | std | normal min/max | defective min/max |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for result in results:
        s = result.score_summary
        ns = result.normal_score_summary
        ds = result.defective_score_summary
        lines.append(
            f"| {result.category} | {s['min']:.6f} | {s['max']:.6f} | {s['mean']:.6f} "
            f"| {s['median']:.6f} | {s['std']:.6f} "
            f"| {ns['min']:.6f} / {ns['max']:.6f} | {ds['min']:.6f} / {ds['max']:.6f} |"
        )
    lines.append("")
    lines.append("## Macro averages (unweighted mean over categories)")
    lines.append("")
    lines.append("| metric | macro average | categories included | categories undefined |")
    lines.append("|---|---:|---:|---|")
    for key in AGGREGATE_METRIC_KEYS:
        block = summary["macro"][key]
        undefined = ", ".join(block["categories_undefined"]) or "-"
        lines.append(
            f"| {key} | {_fmt(block['value'])} | {block['categories_included']} | {undefined} |"
        )
    lines.append("")
    lines.append(summary["macro"]["definition"])
    lines.append("")
    lines.append("## Micro averages (pooled counts — NOT a macro average)")
    lines.append("")
    lines.append("| metric | micro value |")
    lines.append("|---|---:|")
    for key in AGGREGATE_METRIC_KEYS:
        lines.append(f"| {key} | {_fmt(summary['micro'][key])} |")
    lines.append("")
    lines.append(summary["micro"]["definition"])
    lines.append("")
    lines.append("## Metric definitions")
    lines.append("")
    for key in AGGREGATE_METRIC_KEYS:
        lines.append(f"- **{key}**: {THRESHOLD_METRIC_DEFINITIONS[key]}")
    lines.append("")
    lines.append("## Scope and limitations")
    lines.append("")
    lines.append(f"- Decision rule: {DECISION_RULE_APPLIED}.")
    lines.append(f"- Labels policy: {LABELS_POLICY}")
    lines.append(f"- Pixel metrics: {PIXEL_METRICS_STATUS}.")
    lines.append(f"- {RANKING_METRIC_NOTE}")
    lines.append(f"- {GUARANTEE_NOTE}")
    consistency_diffs = [
        r.scoring_consistency.get("max_abs_diff")
        for r in results
        if r.scoring_consistency and r.scoring_consistency.get("max_abs_diff") is not None
    ]
    if consistency_diffs:
        lines.append(
            f"- Scoring consistency vs Phase 4.6 stored scores: compared for "
            f"{len(consistency_diffs)} categories, max |difference| = "
            f"{max(consistency_diffs):.3e} (read-only observation)."
        )
    if failures:
        lines.append(f"- Failures recorded: {len(failures)} (see failures.json).")
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_threshold_evaluation(
    *,
    categories: Sequence[str],
    calibration_run: str | Path,
    bank_baseline_manifest: str | Path,
    processed_root: str | Path,
    bank_dir: str | Path,
    run_root: str | Path,
    batch_size: int = 8,
    device: str | None = None,
    reference_eval_dir: str | Path | None = None,
) -> tuple[Path, list[CategoryThresholdResult]]:
    """Evaluate the frozen thresholds into a brand-new ``threshold_eval_*`` run.

    Never overwrites an existing run directory (``exist_ok=False``), never
    modifies thresholds, banks, or prior artifacts, and fails closed when the
    bank baseline verification fails before or after scoring.
    """
    from ..inference.feature_extractor import DINOv2FeatureExtractor
    from .evaluate import category_paths, evaluate_category

    calibration_run = Path(calibration_run)
    manifest_path = Path(bank_baseline_manifest)
    processed_root = Path(processed_root)
    bank_dir = Path(bank_dir)
    reference_dir = None if reference_eval_dir is None else Path(reference_eval_dir)

    expected = []
    for category in categories:
        validate_category(category)
        if category not in expected:
            expected.append(category)
    expected = sorted(expected)

    thresholds = load_frozen_thresholds(calibration_run, expected)
    banks_before = verify_bank_baseline(bank_dir, manifest_path, expected)

    for category in expected:  # dataset must exist before any run dir is created
        category_paths(processed_root, category)

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = run_root / f"threshold_eval_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Creating threshold evaluation run %s.", run_dir)

    _dump_json(
        run_dir / "bank_hashes_before.json",
        {category: value["sha256"] for category, value in banks_before.items()},
    )

    extractor = DINOv2FeatureExtractor(device=device)
    resolved_device = str(extractor.device)
    results: list[CategoryThresholdResult] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for category in expected:
        eval_result = evaluate_category(
            category,
            processed_root=processed_root,
            bank_dir=bank_dir,
            extractor=extractor,
            batch_size=batch_size,
            limit=None,
            save_maps=False,
            compute_pixel_metrics=False,
        )
        if eval_result.status != "ok":
            failures.append(
                {
                    "category": category,
                    "stage": "scoring",
                    "error_type": eval_result.error_type,
                    "error_message": eval_result.error_message,
                }
            )
            logger.error(
                "%s scoring failed (%s: %s).",
                category,
                eval_result.error_type,
                eval_result.error_message,
            )
            continue
        scores = [row["anomaly_score"] for row in eval_result.rows]
        labels = [bool(row["is_anomaly"]) for row in eval_result.rows]
        result = evaluate_category_thresholds(
            category,
            scores,
            labels,
            thresholds[category],
            pipeline_seconds=eval_result.pipeline_seconds,
        )
        result.scoring_consistency = scoring_consistency(
            category, eval_result.rows, reference_dir
        )
        results.append(result)
        logger.info(
            "%s: n=%d (normal=%d defective=%d) threshold=%.6f FP=%d FN=%d "
            "fpr=%s recall=%s (%.1fs scoring).",
            category,
            result.n_normal + result.n_defective,
            result.n_normal,
            result.n_defective,
            result.threshold_entry.threshold,
            result.fp,
            result.fn,
            _fmt(result.metrics["fpr"]),
            _fmt(result.metrics["recall"]),
            result.pipeline_seconds,
        )
    total_seconds = time.perf_counter() - started

    banks_after = verify_bank_baseline(bank_dir, manifest_path, expected)
    _dump_json(
        run_dir / "bank_hashes_after.json",
        {category: value["sha256"] for category, value in banks_after.items()},
    )
    if banks_after != banks_before:
        _dump_json(
            run_dir / "failures.json",
            [
                {
                    "stage": "post_scoring_integrity",
                    "error_type": "BaselineIntegrityError",
                    "error_message": (
                        "feature bank hashes changed during the evaluation; "
                        "metrics were not written"
                    ),
                }
            ],
        )
        raise BaselineIntegrityError(
            "feature banks changed during threshold evaluation; results discarded."
        )

    if not results:
        _dump_json(run_dir / "failures.json", failures)
        raise RuntimeError(
            "no category could be evaluated; see failures.json in the run directory."
        )

    summary = build_summary(results, failures)
    summary["calibration_run"] = str(calibration_run)
    summary["bank_baseline_manifest"] = str(manifest_path)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scoring_seconds_total"] = round(total_seconds, 3)

    config = build_threshold_eval_config(
        run_dir=run_dir,
        calibration_run=calibration_run,
        bank_baseline_manifest=manifest_path,
        processed_root=processed_root,
        bank_dir=bank_dir,
        extractor=extractor,
        batch_size=batch_size,
        device=resolved_device,
        thresholds=thresholds,
        reference_eval_dir=reference_dir,
    )

    _write_threshold_metrics_csv(run_dir, results)
    _write_confusion_csv(run_dir, results)
    _dump_json(run_dir / "summary.json", summary)
    _dump_json(run_dir / "config.json", config)
    _dump_json(run_dir / "failures.json", failures)
    write_report(run_dir, results, summary, failures)
    logger.info(
        "Threshold evaluation run complete: %s (scoring %.1fs, %d categories).",
        run_dir,
        total_seconds,
        len(results),
    )
    return run_dir, results
