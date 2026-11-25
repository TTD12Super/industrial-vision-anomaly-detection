"""Standalone DINOv2 PatchCore-lite cross-domain experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from mvtec_system.anomaly.knn import nearest_neighbor_scores
from mvtec_system.anomaly.maps import make_anomaly_maps
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
GOOD_GROUP = "0_good"
PATCH_MEMORY_ARTIFACT_TYPE = "dinov2_patch_memory_classifier"
PATCH_MEMORY_SCHEMA_VERSION = "1.0"


def _image_tensor(path: Path, config: DINOv2Config) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def discover_cad_sd(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Discover CAD-SD train/0_good calibration and grouped test records."""

    root = Path(root)
    train_good = root / "train" / GOOD_GROUP
    test_root = root / "test"
    if not train_good.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"CAD-SD train/test layout is incomplete: {root}")
    calibration = [
        {"path": path, "label": 0, "group": GOOD_GROUP, "mask_path": None}
        for path in _image_files(train_good)
    ]
    test: list[dict[str, Any]] = []
    for group_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        label = 0 if group_dir.name == GOOD_GROUP else 1
        test.extend(
            {"path": path, "label": label, "group": group_dir.name, "mask_path": None}
            for path in _image_files(group_dir)
        )
    if not calibration or not test or {row["label"] for row in test} != {0, 1}:
        raise ValueError("CAD-SD must contain train/0_good and both GOOD/DEFECT test records")
    return {"calibration": calibration, "test": test}


def discover_mvtec_screw(root: str | Path) -> list[dict[str, Any]]:
    """Discover the official MVTec screw test split and its masks."""

    root = Path(root)
    test_root = root / "test"
    ground_truth_root = root / "ground_truth"
    if not test_root.is_dir():
        raise FileNotFoundError(f"MVTec screw test directory is missing: {test_root}")
    records: list[dict[str, Any]] = []
    for group_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        label = 0 if group_dir.name == "good" else 1
        for path in _image_files(group_dir):
            mask_path = None if label == 0 else ground_truth_root / group_dir.name / f"{path.stem}_mask.png"
            records.append(
                {
                    "path": path,
                    "label": label,
                    "group": group_dir.name,
                    "mask_path": mask_path,
                }
            )
    if not records or {row["label"] for row in records} != {0, 1}:
        raise ValueError(f"MVTec screw test must contain GOOD and DEFECT images: {test_root}")
    return records


def build_patch_memory_bank(
    image_paths: Sequence[str | Path],
    encoder: SharedDINOv2,
    batch_size: int = 8,
) -> torch.Tensor:
    """Build an in-memory patch bank from the supplied GOOD paths only."""

    if not image_paths or batch_size < 1:
        raise ValueError("image_paths must be non-empty and batch_size must be positive")
    chunks: list[torch.Tensor] = []
    paths = [Path(path) for path in image_paths]
    for start in range(0, len(paths), batch_size):
        images = torch.stack([_image_tensor(path, encoder.config) for path in paths[start : start + batch_size]])
        with torch.inference_mode():
            features = encoder.forward(images).patch_embeddings
        patches = features.reshape(-1, features.shape[-1]).detach().cpu().float()
        if not torch.isfinite(patches).all().item():
            raise ValueError("DINOv2 memory-bank features contain non-finite values")
        chunks.append(patches)
    bank = torch.cat(chunks, dim=0)
    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError("DINOv2 memory bank is empty")
    return bank


def patchcore_scores(
    query_patches: torch.Tensor,
    memory_bank: torch.Tensor,
    k: int = 1,
) -> tuple[float, np.ndarray]:
    """Return max patch distance and its patch-score vector."""

    if query_patches.ndim != 2 or memory_bank.ndim != 2:
        raise ValueError("query_patches and memory_bank must be two-dimensional")
    scores = nearest_neighbor_scores(query_patches.float(), memory_bank.float(), k=k)
    return float(scores.max().item()), scores.numpy()


def save_patch_memory_artifact(
    path: str | Path,
    memory_bank: torch.Tensor,
    encoder: SharedDINOv2,
    threshold: float,
    k: int = 1,
    aggregation: str = "max",
    model_version: str = "dinov2-patch-memory",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a validated patch-memory classifier candidate for runtime use.

    The artifact contains only GOOD patch embeddings, a threshold selected on a
    declared calibration split, and the encoder contract needed to reject stale
    or incompatible model bundles.  It is deliberately separate from the
    current production classifier until an external holdout is approved.
    """

    if not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
        raise ValueError("memory_bank must be a non-empty two-dimensional tensor")
    if not torch.isfinite(memory_bank).all().item():
        raise ValueError("memory_bank contains non-finite values")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if k < 1 or k > memory_bank.shape[0]:
        raise ValueError("k must be between 1 and the memory-bank size")
    # Validate the name against the same fixed set used by benchmark reports.
    aggregate_patch_scores(np.ones(4, dtype=np.float32), aggregation)
    payload: dict[str, Any] = {
        "artifact_type": PATCH_MEMORY_ARTIFACT_TYPE,
        "schema_version": PATCH_MEMORY_SCHEMA_VERSION,
        "memory_bank": memory_bank.detach().cpu().float().contiguous(),
        "threshold": float(threshold),
        "k": int(k),
        "aggregation": aggregation,
        "model_version": str(model_version),
        "encoder": encoder.metadata(),
        "metadata": dict(metadata or {}),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "artifact_type": payload["artifact_type"],
        "schema_version": payload["schema_version"],
        "path": str(output),
        "memory_count": int(memory_bank.shape[0]),
        "embedding_dimension": int(memory_bank.shape[1]),
        "threshold": float(threshold),
        "k": int(k),
        "aggregation": aggregation,
        "model_version": str(model_version),
    }


def load_patch_memory_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a CPU patch-memory classifier artifact."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Patch-memory classifier artifact is missing: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("artifact_type") != PATCH_MEMORY_ARTIFACT_TYPE:
        raise ValueError(f"Invalid patch-memory classifier artifact: {artifact}")
    if payload.get("schema_version") != PATCH_MEMORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported patch-memory classifier schema: {artifact}")
    memory_bank = payload.get("memory_bank")
    encoder = payload.get("encoder")
    threshold = payload.get("threshold")
    k = payload.get("k")
    aggregation = payload.get("aggregation")
    if (
        not isinstance(memory_bank, torch.Tensor)
        or memory_bank.ndim != 2
        or memory_bank.shape[0] == 0
        or not torch.isfinite(memory_bank).all().item()
        or not isinstance(encoder, dict)
        or not isinstance(k, int)
        or k < 1
        or k > memory_bank.shape[0]
        or not isinstance(aggregation, str)
    ):
        raise ValueError(f"Malformed patch-memory classifier artifact: {artifact}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Malformed patch-memory threshold: {artifact}") from error
    if not np.isfinite(threshold):
        raise ValueError(f"Malformed patch-memory threshold: {artifact}")
    aggregate_patch_scores(np.ones(4, dtype=np.float32), aggregation)
    return {
        **payload,
        "memory_bank": memory_bank.float().cpu().contiguous(),
        "threshold": threshold,
        "k": k,
    }


PATCH_SCORE_AGGREGATIONS: tuple[tuple[str, float | None], ...] = (
    ("max", None),
    ("top_1pct", 0.01),
    ("top_5pct", 0.05),
    ("top_10pct", 0.10),
    ("top_20pct", 0.20),
)


def aggregate_patch_scores(patch_scores: Sequence[float] | np.ndarray, aggregation: str) -> float:
    """Aggregate one image's patch distances without another encoder forward.

    ``top_*`` uses the mean of the largest ``ceil(fraction * patch_count)``
    distances, with at least one patch included. The fractions are fixed by
    ``PATCH_SCORE_AGGREGATIONS`` and are not learned from test data.
    """

    scores = np.asarray(patch_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("patch_scores must be non-empty and finite")
    fractions = dict(PATCH_SCORE_AGGREGATIONS)
    if aggregation not in fractions:
        raise ValueError(f"unknown patch-score aggregation: {aggregation}")
    fraction = fractions[aggregation]
    if fraction is None:
        return float(scores.max())
    top_count = max(1, int(np.ceil(fraction * scores.size)))
    top_scores = np.partition(scores, scores.size - top_count)[-top_count:]
    return float(top_scores.mean())


def deterministic_cad_good_split(
    paths: Sequence[str | Path],
    root: str | Path | None = None,
    memory_fraction: float = 0.5,
    salt: str = "cad_sd_train_0_good_v1",
    excluded_paths: Sequence[str | Path] = (),
    memory_count_target: int | None = None,
) -> dict[str, Any]:
    """Split CAD-SD GOOD paths after exact-content de-duplication.

    SHA-256 bytes are the identity. One lexicographically first path is retained
    for each candidate content hash; candidate hashes present in ``excluded_paths``
    (the CAD-SD test set in the benchmark) are excluded before the salted path
    ranking split. This makes memory, calibration, and excluded test content
    disjoint rather than merely path-disjoint.
    """

    if not paths:
        raise ValueError("paths must be non-empty")
    if not 0.0 < memory_fraction < 1.0:
        raise ValueError("memory_fraction must be in (0, 1)")
    if not salt:
        raise ValueError("salt must be non-empty")

    root_path = Path(root).resolve() if root is not None else None
    if memory_count_target is not None and memory_count_target < 1:
        raise ValueError("memory_count_target must be positive when provided")

    def normalize_path(raw_path: str | Path) -> tuple[Path, str]:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"content-hash split path is not a file: {path}")
        if root_path is None:
            relative = path.as_posix()
        else:
            try:
                relative = path.relative_to(root_path).as_posix()
            except ValueError as exc:
                raise ValueError(f"path is outside split root: {path}") from exc
        return path, relative

    candidate_records: list[dict[str, str | Path]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path, relative = normalize_path(raw_path)
        if path in seen:
            raise ValueError(f"paths must be unique: {path}")
        seen.add(path)
        candidate_records.append(
            {"path": path, "relative_path": relative, "content_hash": _sha256(path)}
        )
    if len(candidate_records) < 2:
        raise ValueError("at least two unique paths are required for a memory/calibration split")

    excluded_records: list[dict[str, str | Path]] = []
    excluded_seen: set[Path] = set()
    for raw_path in excluded_paths:
        path, relative = normalize_path(raw_path)
        if path in excluded_seen:
            continue
        excluded_seen.add(path)
        excluded_records.append(
            {"path": path, "relative_path": relative, "content_hash": _sha256(path)}
        )
    excluded_hashes = {str(record["content_hash"]) for record in excluded_records}

    by_content_hash: dict[str, list[dict[str, str | Path]]] = {}
    for record in candidate_records:
        by_content_hash.setdefault(str(record["content_hash"]), []).append(record)
    retained_records: list[dict[str, str | Path]] = []
    duplicate_exclusions: list[dict[str, str]] = []
    exclusion_counts = {"duplicate_candidate_content": 0, "excluded_test_content": 0}
    for content_hash in sorted(by_content_hash):
        group = sorted(by_content_hash[content_hash], key=lambda record: str(record["relative_path"]))
        canonical = group[0]
        for duplicate in group[1:]:
            duplicate_exclusions.append(
                {
                    "path": str(duplicate["relative_path"]),
                    "canonical_path": str(canonical["relative_path"]),
                    "content_hash": content_hash,
                    "reason": "duplicate_candidate_content",
                }
            )
            exclusion_counts["duplicate_candidate_content"] += 1
        if content_hash in excluded_hashes:
            duplicate_exclusions.append(
                {
                    "path": str(canonical["relative_path"]),
                    "canonical_path": str(canonical["relative_path"]),
                    "content_hash": content_hash,
                    "reason": "excluded_test_content",
                }
            )
            exclusion_counts["excluded_test_content"] += 1
            continue
        retained_records.append(canonical)
    if len(retained_records) < 2:
        raise ValueError("content-hash exclusions left fewer than two CAD GOOD candidates")

    for record in retained_records:
        record["digest"] = hashlib.sha256(
            f"{salt}\n{record['relative_path']}".encode("utf-8")
        ).hexdigest()
    ordered = sorted(retained_records, key=lambda record: (str(record["digest"]), str(record["relative_path"])))
    requested_memory_count = (
        memory_count_target if memory_count_target is not None else int(round(len(candidate_records) * memory_fraction))
    )
    memory_count = max(1, min(len(ordered) - 1, requested_memory_count))
    memory_records = ordered[:memory_count]
    calibration_records = ordered[memory_count:]
    memory_relative_paths = sorted(str(record["relative_path"]) for record in memory_records)
    calibration_relative_paths = sorted(str(record["relative_path"]) for record in calibration_records)
    memory_content_hashes = sorted(str(record["content_hash"]) for record in memory_records)
    calibration_content_hashes = sorted(str(record["content_hash"]) for record in calibration_records)
    excluded_content_hashes = sorted(excluded_hashes)
    split_payload = {
        "salt": salt,
        "memory_fraction": float(memory_fraction),
        "memory_count_target": requested_memory_count,
        "memory_relative_paths": memory_relative_paths,
        "calibration_relative_paths": calibration_relative_paths,
        "memory_content_hashes": memory_content_hashes,
        "calibration_content_hashes": calibration_content_hashes,
        "excluded_content_hashes": excluded_content_hashes,
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def hash_paths(relative_paths: list[str]) -> str:
        return hashlib.sha256(
            json.dumps(relative_paths, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return {
        "memory_paths": [Path(record["path"]) for record in memory_records],
        "calibration_paths": [Path(record["path"]) for record in calibration_records],
        "memory_relative_paths": memory_relative_paths,
        "calibration_relative_paths": calibration_relative_paths,
        "memory_content_hashes": memory_content_hashes,
        "calibration_content_hashes": calibration_content_hashes,
        "excluded_content_hashes": excluded_content_hashes,
        "memory_path_hash": hash_paths(memory_relative_paths),
        "calibration_path_hash": hash_paths(calibration_relative_paths),
        "split_hash": split_hash,
        "memory_fraction": float(memory_fraction),
        "memory_count_target": requested_memory_count,
        "effective_memory_fraction": float(memory_count / len(ordered)),
        "candidate_path_count": len(candidate_records),
        "candidate_unique_content_count": len(by_content_hash),
        "retained_unique_content_count": len(ordered),
        "excluded_path_count": len(duplicate_exclusions),
        "exclusion_reference_path_count": len(excluded_records),
        "exclusion_reference_unique_content_count": len(excluded_hashes),
        "exclusion_reference_duplicate_path_count": len(excluded_records) - len(excluded_hashes),
        "duplicate_exclusions": duplicate_exclusions,
        "exclusion_counts": exclusion_counts,
        "hash_group_policy": (
            "sha256 exact file bytes; retain lexicographically first candidate path per content hash; "
            "exclude candidate hashes present in the CAD-SD test set; salted relative-path ranking splits "
            "the remaining unique candidates"
        ),
        "content_hash_disjoint": (
            set(memory_content_hashes).isdisjoint(calibration_content_hashes)
            and set(memory_content_hashes).isdisjoint(excluded_content_hashes)
            and set(calibration_content_hashes).isdisjoint(excluded_content_hashes)
        ),
        "salt": salt,
    }


def _score_records(
    records: Sequence[dict[str, Any]],
    encoder: SharedDINOv2,
    memory_bank: torch.Tensor,
    k: int,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scored: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = list(records[start : start + batch_size])
        images = torch.stack([_image_tensor(record["path"], encoder.config) for record in batch_records])
        with torch.inference_mode():
            features = encoder.forward(images)
        patch_embeddings = features.patch_embeddings
        batch_patch_scores = nearest_neighbor_scores(
            patch_embeddings.reshape(-1, patch_embeddings.shape[-1]).float(), memory_bank.float(), k=k
        ).reshape(len(batch_records), -1)
        anomaly_maps = make_anomaly_maps(
            batch_patch_scores,
            encoder.patch_grid,
            encoder.padded_size,
            encoder.config.image_size,
        )
        for index, record in enumerate(batch_records):
            patch_scores = batch_patch_scores[index]
            scored.append(
                {
                    **record,
                    "image_score": aggregate_patch_scores(patch_scores.numpy(), "max"),
                    "patch_scores": patch_scores.numpy(),
                    "anomaly_map": anomaly_maps[index],
                }
            )
    return scored


def select_good_quantile_threshold(
    good_scores: Sequence[float],
    allowed_fpr: float = 0.05,
    source_label: str = "CAD-SD train/0_good image scores only",
) -> dict[str, Any]:
    """Select a locked image threshold from GOOD-only scores using a declared FPR rule."""

    scores = np.asarray(good_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("good_scores must be non-empty and finite")
    if not 0.0 <= allowed_fpr < 1.0:
        raise ValueError("allowed_fpr must be in [0, 1)")
    quantile = 1.0 - allowed_fpr
    if allowed_fpr == 0.0:
        threshold = float(np.nextafter(scores.max(), np.inf))
        method = "strictly_above_max_good_score_for_zero_empirical_fpr"
    else:
        threshold = float(np.quantile(scores, quantile, method="higher"))
        method = "higher_empirical_good_quantile"
    empirical_fpr = float((scores >= threshold).mean())
    return {
        "threshold": threshold,
        "allowed_good_fpr": allowed_fpr,
        "good_quantile": quantile,
        "empirical_good_fpr": empirical_fpr,
        "rule": f"{method}(q={quantile:.6f}) on {source_label}",
    }


def image_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_array.shape != scores_array.shape or set(np.unique(labels_array)) != {0, 1}:
        raise ValueError("image metrics require matching GOOD/DEFECT labels")
    predicted = scores_array >= threshold
    goods = labels_array == 0
    defects = labels_array == 1
    return {
        "count": int(labels_array.size),
        "good_count": int(goods.sum()),
        "defect_count": int(defects.sum()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "average_precision": float(average_precision_score(labels_array, scores_array)),
        "defect_recall": float((predicted & defects).sum() / defects.sum()),
        "good_false_positive_count": int((predicted & goods).sum()),
        "good_false_positive_rate": float((predicted & goods).sum() / goods.sum()),
    }


def per_group_metrics(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    groups = sorted({str(record["group"]) for record in records})
    output: dict[str, dict[str, Any]] = {}
    for group in groups:
        selected = [record for record in records if record["group"] == group]
        labels = [record["label"] for record in selected]
        scores = [record["image_score"] for record in selected]
        if set(labels) == {0, 1}:
            output[group] = image_metrics(labels, scores, threshold)
        else:
            labels_array = np.asarray(labels)
            predicted = np.asarray(scores) >= threshold
            output[group] = {
                "count": len(selected),
                "good_count": int(sum(label == 0 for label in labels)),
                "defect_count": int(sum(label == 1 for label in labels)),
                "threshold": float(threshold),
                "defect_recall": float((predicted & (labels_array == 1)).sum() / (labels_array == 1).sum()) if (labels_array == 1).any() else None,
                "good_false_positive_count": int(sum(predicted & (np.asarray(labels) == 0))),
                "good_false_positive_rate": float((predicted & (labels_array == 0)).sum() / (labels_array == 0).sum()) if (labels_array == 0).any() else None,
                "auroc": None,
                "average_precision": None,
            }
    return output


def _records_with_aggregated_scores(
    records: Sequence[dict[str, Any]], aggregation: str
) -> tuple[list[dict[str, Any]], list[float]]:
    scores = [aggregate_patch_scores(record["patch_scores"], aggregation) for record in records]
    return [
        {**record, "image_score": score}
        for record, score in zip(records, scores)
    ], scores


def _valid_mvtec_masks(records: Sequence[dict[str, Any]], map_shape: tuple[int, int]) -> tuple[list[dict[str, Any]], list[str]]:
    prepared: list[dict[str, Any]] = []
    invalid: list[str] = []
    for record in records:
        if record["label"] == 0:
            truth = np.zeros(map_shape, dtype=bool)
        else:
            mask_path = record.get("mask_path")
            source = None
            if mask_path:
                try:
                    with Image.open(mask_path) as mask_image:
                        source = np.asarray(mask_image.convert("L"), dtype=np.uint8)
                except (OSError, ValueError):
                    source = None
            if source is None or not np.any(source > 0):
                invalid.append(str(record["path"]))
                continue
            truth = cv2.resize((source > 0).astype(np.uint8), (map_shape[1], map_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        prepared.append({**record, "truth_mask": truth})
    return prepared, invalid


def pixel_metrics(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    true_positive = false_positive = false_negative = 0
    for record in records:
        prediction = record["anomaly_map"] >= threshold
        truth = record["truth_mask"]
        scores.append(record["anomaly_map"].reshape(-1))
        truths.append(truth.reshape(-1))
        true_positive += int(np.logical_and(prediction, truth).sum())
        false_positive += int(np.logical_and(prediction, ~truth).sum())
        false_negative += int(np.logical_and(~prediction, truth).sum())
    flat_truth = np.concatenate(truths).astype(np.int64)
    flat_scores = np.concatenate(scores)
    union = true_positive + false_positive + false_negative
    return {
        "pixel_average_precision": float(average_precision_score(flat_truth, flat_scores)) if flat_truth.any() else None,
        "pixel_precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        "pixel_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "pixel_dice": 2 * true_positive / (2 * true_positive + false_positive + false_negative) if union else 1.0,
        "pixel_iou": true_positive / union if union else 1.0,
        "pixel_threshold": float(threshold),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_provenance(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(paths):
        entries.append({"path": str(path.resolve()), "relative_path": str(path.resolve().relative_to(root.resolve())), "sha256": _sha256(path), "bytes": path.stat().st_size})
    manifest_digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"root": str(root.resolve()), "count": len(entries), "sha256": manifest_digest, "files": entries}


def evaluate_dinov2_patchcore_lite(
    mvtec_root: str | Path = "datasets/mvtec_ad/screw",
    cad_root: str | Path = "datasets/external/cad_sd/data/unpacked/Co-occurrence Anomaly Detection Screw Dataset",
    report_path: str | Path = "reports/experiments/dinov2_patchcore_lite.json",
    encoder: SharedDINOv2 | None = None,
    batch_size: int = 8,
    k: int = 1,
    allowed_good_fpr: float = 0.05,
    device: str | None = None,
    mixed_nominal: bool = False,
    cad_memory_fraction: float = 0.5,
    cad_split_salt: str = "cad_sd_train_0_good_v1",
    classifier_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the default MVTec-memory protocol or an explicit mixed-nominal variant.

    The default keeps the original experiment unchanged: MVTec train/good is the
    memory bank and all CAD-SD train/0_good images calibrate the locked threshold.
    With ``mixed_nominal=True``, a content-hash-deduplicated deterministic CAD-SD
    GOOD subset is added to memory and the disjoint remainder calibrates the
    threshold; CAD-SD test content is excluded from both groups.
    """

    mvtec_root = Path(mvtec_root)
    cad_root = Path(cad_root)
    cad = discover_cad_sd(cad_root)
    mvtec_test = discover_mvtec_screw(mvtec_root)
    mvtec_train_good = sorted((mvtec_root / "train" / "good").glob("*"))
    mvtec_train_good = [path for path in mvtec_train_good if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not mvtec_train_good:
        raise FileNotFoundError(f"MVTec screw train/good is empty: {mvtec_root / 'train' / 'good'}")
    cad_good_paths = [record["path"] for record in cad["calibration"]]
    cad_test_paths = [record["path"] for record in cad["test"]]
    cad_split = None
    memory_paths = list(mvtec_train_good)
    calibration_records = list(cad["calibration"])
    if mixed_nominal:
        cad_split = deterministic_cad_good_split(
            cad_good_paths,
            root=cad_root,
            memory_fraction=cad_memory_fraction,
            salt=cad_split_salt,
            excluded_paths=cad_test_paths,
            memory_count_target=int(round(len(cad_good_paths) * cad_memory_fraction)),
        )
        memory_paths.extend(cad_split["memory_paths"])
        calibration_paths = {Path(path).resolve() for path in cad_split["calibration_paths"]}
        calibration_records = [
            record for record in cad["calibration"] if Path(record["path"]).resolve() in calibration_paths
        ]
        if not calibration_records or not cad_split["memory_paths"]:
            raise ValueError("mixed-nominal protocol requires non-empty CAD memory and calibration splits")
    encoder = encoder or SharedDINOv2(
        DINOv2Config(image_size=(256, 256)),
        artifact_root="artifacts/dinov2_knn",
        device=device or str(select_device()),
    )
    memory_bank = build_patch_memory_bank(memory_paths, encoder, batch_size)
    cad_calibration = _score_records(calibration_records, encoder, memory_bank, k, batch_size=batch_size)
    calibration_source_label = (
        "CAD-SD train/0_good image scores only"
        if not mixed_nominal
        else "disjoint CAD-SD train/0_good calibration holdout image scores only"
    )
    threshold = select_good_quantile_threshold(
        [record["image_score"] for record in cad_calibration],
        allowed_good_fpr,
        source_label=calibration_source_label,
    )
    protocol = "mixed_nominal" if mixed_nominal else "mvtec_memory_cad_calibration"
    classifier_artifact = None
    if classifier_artifact_path is not None:
        classifier_artifact = save_patch_memory_artifact(
            classifier_artifact_path,
            memory_bank,
            encoder,
            threshold["threshold"],
            k=k,
            aggregation="max",
            model_version="dinov2-patch-memory-mixed-nominal" if mixed_nominal else "dinov2-patch-memory",
            metadata={
                "protocol": protocol,
                "calibration": threshold,
                "memory_image_count": len(memory_paths),
                "calibration_image_count": len(calibration_records),
                "test_locked": True,
            },
        )
    cad_test = _score_records(cad["test"], encoder, memory_bank, k, batch_size=batch_size)
    mvtec_scored = _score_records(mvtec_test, encoder, memory_bank, k, batch_size=batch_size)
    cad_metrics = image_metrics(
        [record["label"] for record in cad_test],
        [record["image_score"] for record in cad_test],
        threshold["threshold"],
    )
    mvtec_metrics = image_metrics(
        [record["label"] for record in mvtec_scored],
        [record["image_score"] for record in mvtec_scored],
        threshold["threshold"],
    )
    mvtec_pixel_records, invalid_masks = _valid_mvtec_masks(mvtec_scored, encoder.config.image_size)
    split_report = None
    if cad_split is not None:
        split_report = {
            "hash_group_policy": cad_split["hash_group_policy"],
            "candidate_path_count": cad_split["candidate_path_count"],
            "candidate_unique_content_count": cad_split["candidate_unique_content_count"],
            "retained_unique_content_count": cad_split["retained_unique_content_count"],
            "excluded_path_count": cad_split["excluded_path_count"],
            "exclusion_reference_path_count": cad_split["exclusion_reference_path_count"],
            "exclusion_reference_unique_content_count": cad_split["exclusion_reference_unique_content_count"],
            "exclusion_reference_duplicate_path_count": cad_split["exclusion_reference_duplicate_path_count"],
            "exclusion_counts": cad_split["exclusion_counts"],
            "duplicate_exclusions": cad_split["duplicate_exclusions"],
            "memory_relative_paths": cad_split["memory_relative_paths"],
            "calibration_relative_paths": cad_split["calibration_relative_paths"],
            "memory_content_hashes": cad_split["memory_content_hashes"],
            "calibration_content_hashes": cad_split["calibration_content_hashes"],
            "excluded_content_hashes": cad_split["excluded_content_hashes"],
            "memory_path_hash": cad_split["memory_path_hash"],
            "calibration_path_hash": cad_split["calibration_path_hash"],
            "split_hash": cad_split["split_hash"],
            "memory_fraction": cad_split["memory_fraction"],
            "memory_count_target": cad_split["memory_count_target"],
            "effective_memory_fraction": cad_split["effective_memory_fraction"],
            "salt": cad_split["salt"],
            "content_hash_disjoint": cad_split["content_hash_disjoint"],
        }
    warnings = [
        (
            "Memory bank uses only MVTec screw train/good; CAD-SD images are never added to memory."
            if not mixed_nominal
            else "Mixed-nominal memory adds only the content-hash-deduplicated CAD-SD train/0_good memory subset."
        ),
        (
            "The image threshold is calibrated only on CAD-SD train/0_good with the declared GOOD quantile/FPR rule."
            if not mixed_nominal
            else "The image threshold is calibrated only on the content-hash-disjoint CAD-SD train/0_good holdout."
        ),
        "CAD-SD test and MVTec screw test are locked evaluations; no test scores are used for tuning.",
        "Cross-domain results are diagnostic and subject to CAD-SD/MVTec domain shift in capture, background, scale, and defect appearance.",
    ]
    if mixed_nominal:
        warnings.insert(
            2,
            "Mixed-nominal CAD memory/calibration/test groups use SHA-256 exact-byte identity; duplicate and test-overlap exclusions are recorded in provenance.",
        )
    if invalid_masks:
        warnings.append(f"MVTec pixel metrics omitted because {len(invalid_masks)} defect masks are missing or invalid.")
    aggregation_comparison: dict[str, Any] = {}
    for aggregation, top_fraction in PATCH_SCORE_AGGREGATIONS:
        _, calibration_scores = _records_with_aggregated_scores(cad_calibration, aggregation)
        aggregation_threshold = select_good_quantile_threshold(
            calibration_scores,
            allowed_good_fpr,
            source_label=calibration_source_label,
        )
        cad_aggregated, cad_scores = _records_with_aggregated_scores(cad_test, aggregation)
        mvtec_aggregated, mvtec_scores = _records_with_aggregated_scores(mvtec_scored, aggregation)
        aggregation_comparison[aggregation] = {
            "top_fraction": top_fraction,
            "top_patch_count_rule": (
                "max(1, ceil(fraction * patch_count))" if top_fraction is not None else "single largest patch"
            ),
            "calibration": {
                "source": calibration_source_label,
                "image_count": len(calibration_scores),
                **aggregation_threshold,
            },
            "cad_sd_test": {
                "metrics": image_metrics(
                    [record["label"] for record in cad_aggregated],
                    cad_scores,
                    aggregation_threshold["threshold"],
                ),
                "per_defect_group": per_group_metrics(cad_aggregated, aggregation_threshold["threshold"]),
            },
            "mvtec_screw_test": {
                "metrics": image_metrics(
                    [record["label"] for record in mvtec_aggregated],
                    mvtec_scores,
                    aggregation_threshold["threshold"],
                ),
                "per_defect_group": per_group_metrics(mvtec_aggregated, aggregation_threshold["threshold"]),
            },
        }
    report = {
        "schema_version": "1.0",
        "experiment": "dinov2_patchcore_lite",
        "protocol": protocol,
        "memory_bank": {
            "source": (
                "MVTec screw train/good only"
                if not mixed_nominal
                else "MVTec screw train/good + deterministic CAD-SD train/0_good subset"
            ),
            "image_count": len(memory_paths),
            "patch_count": int(memory_bank.shape[0]),
            "embedding_dimension": int(memory_bank.shape[1]),
            "k": k,
            "memory_sha256": hashlib.sha256(memory_bank.numpy().tobytes()).hexdigest(),
        },
        "calibration": {
            "source": (
                "CAD-SD train/0_good only"
                if not mixed_nominal
                else "CAD-SD train/0_good deterministic disjoint calibration holdout"
            ),
            "image_count": len(calibration_records),
            **threshold,
        },
        "runtime_classifier_artifact": classifier_artifact,
        "score_aggregation_comparison": {
            "default": "max",
            "same_patch_distances": True,
            "strategies": aggregation_comparison,
        },
        "cad_sd_test": {
            "metrics": cad_metrics,
            "per_defect_group": per_group_metrics(cad_test, threshold["threshold"]),
            "pixel_metrics": None,
        },
        "mvtec_screw_test": {
            "metrics": mvtec_metrics,
            "per_defect_group": per_group_metrics(mvtec_scored, threshold["threshold"]),
            "pixel_metrics": pixel_metrics(mvtec_pixel_records, threshold["threshold"]) if not invalid_masks else None,
            "valid_defect_mask_count": len(mvtec_pixel_records) - sum(record["label"] == 0 for record in mvtec_pixel_records),
            "invalid_defect_mask_count": len(invalid_masks),
        },
        "provenance": {
            "protocol": protocol,
            "encoder": encoder.metadata(),
            "mvtec_train_good": _dataset_provenance(mvtec_train_good, mvtec_root),
            "cad_sd_train_0_good": _dataset_provenance(cad_good_paths, cad_root),
            "memory_bank_sources": {
                "mvtec_train_good": _dataset_provenance(mvtec_train_good, mvtec_root),
                "cad_sd_train_0_good_memory": (
                    _dataset_provenance(cad_split["memory_paths"], cad_root) if cad_split is not None else None
                ),
            },
            "calibration_sources": {
                "cad_sd_train_0_good_calibration": _dataset_provenance(
                    [record["path"] for record in calibration_records], cad_root
                ),
            },
            "cad_good_split": split_report,
            "cad_sd_test": _dataset_provenance([record["path"] for record in cad["test"]], cad_root),
            "mvtec_screw_test": _dataset_provenance([record["path"] for record in mvtec_test], mvtec_root),
            "test_locked": True,
        },
        "warnings": warnings,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "PATCH_SCORE_AGGREGATIONS",
    "aggregate_patch_scores",
    "build_patch_memory_bank",
    "deterministic_cad_good_split",
    "discover_cad_sd",
    "discover_mvtec_screw",
    "evaluate_dinov2_patchcore_lite",
    "image_metrics",
    "patchcore_scores",
    "per_group_metrics",
    "load_patch_memory_artifact",
    "save_patch_memory_artifact",
    "pixel_metrics",
    "select_good_quantile_threshold",
]
