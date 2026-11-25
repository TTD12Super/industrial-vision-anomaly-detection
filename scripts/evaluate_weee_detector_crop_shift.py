#!/usr/bin/env python3
"""Measure WEEE detector-crop shift against the frozen PatchCore-lite protocol.

WEEE ``Test_set.json`` supplies screw geometry only. This is a GOOD/domain-shift
stress experiment: it reports detector coverage and crop-to-score shift, but it
does not report defect recall or any WEEE defect metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata

try:
    from screw_yolo_obb import annotation_to_corners
except ImportError:
    from scripts.screw_yolo_obb import annotation_to_corners


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
AGGREGATIONS = ("max", "top_1pct")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_iou(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    """Return exact convex-polygon IoU for two four-corner OBBs."""

    first_array = np.asarray(first, dtype=np.float32).reshape(4, 2)
    second_array = np.asarray(second, dtype=np.float32).reshape(4, 2)
    first_area = abs(float(cv2.contourArea(first_array)))
    second_area = abs(float(cv2.contourArea(second_array)))
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_array, second_array)
    union = first_area + second_area - float(intersection)
    return float(intersection / union) if union > 0.0 else 0.0


def _corners(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value["corners"]
    return np.asarray(value, dtype=np.float32).reshape(4, 2)


def match_objects(
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match one-class OBBs by descending polygon IoU."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            candidates.append(
                (polygon_iou(_corners(gt), _corners(prediction)), gt_index, prediction_index)
            )
    matches = []
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    for iou, gt_index, prediction_index in sorted(candidates, reverse=True):
        if iou < iou_threshold or gt_index in used_gt or prediction_index in used_predictions:
            continue
        matches.append(
            {"gt_index": gt_index, "prediction_index": prediction_index, "crop_iou": float(iou)}
        )
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
    return {
        "matches": sorted(matches, key=lambda row: (row["gt_index"], row["prediction_index"])),
        "unmatched_gt_indices": [index for index in range(len(ground_truth)) if index not in used_gt],
        "unmatched_prediction_indices": [
            index for index in range(len(predictions)) if index not in used_predictions
        ],
    }


def _resolve_image(root: Path, file_name: str) -> Path:
    raw = Path(file_name)
    candidates = [root / "images" / "test" / raw, root / "images" / raw, root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(
        path
        for path in root.rglob(raw.name)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "Annotations" not in path.parts
    )
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"WEEE image not found: {file_name}")


def load_weee_records(root: str | Path, annotation_path: str | Path) -> list[dict[str, Any]]:
    """Load WEEE test images and oriented GT screw boxes from COCO polygons."""

    root = Path(root).resolve()
    annotation_path = Path(annotation_path).resolve()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: str(item.get("name", "")) for item in payload.get("categories", [])}
    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        category = categories.get(annotation.get("category_id"), "")
        if category.lower() != "screw":
            continue
        annotations_by_image.setdefault(annotation.get("image_id"), []).append(annotation)

    records = []
    for image in payload.get("images", []):
        annotations = annotations_by_image.get(image.get("id"), [])
        if not annotations:
            continue
        objects = []
        for annotation in annotations:
            corners = annotation_to_corners(annotation, allow_axis_fallback=False)
            objects.append(
                {
                    "annotation_id": annotation.get("id"),
                    "corners": np.asarray(corners, dtype=np.float32).tolist(),
                }
            )
        records.append(
            {
                "image_id": image.get("id"),
                "file_name": image.get("file_name"),
                "path": _resolve_image(root, image.get("file_name", "")),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "ground_truth": objects,
            }
        )
    if not records:
        raise ValueError("WEEE COCO test annotations contain no screw records")
    return records


def _load_detector_crop_manifest(
    path: str | Path,
    records: Sequence[dict[str, Any]],
    expected_test_annotations_sha256: str,
) -> dict[str, Any]:
    """Load precomputed detector crops without running the detector again."""

    from inspection.geometry import oriented_crop

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Crop manifest not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Crop manifest is empty: {path}")
    valid_statuses = {"matched", "missed_ground_truth", "unmatched_prediction"}
    record_by_id = {str(record["image_id"]): record for record in records}
    confidence_values = set()
    annotation_hashes = set()
    split_hashes = set()
    source_rgb_cache: dict[str, np.ndarray] = {}
    matched_pairs = []
    unmatched_gt = []
    unmatched_predictions = []
    all_confidences = []
    all_ious = []
    status_counts = {status: 0 for status in valid_statuses}

    def source_rgb(record: dict[str, Any]) -> np.ndarray:
        key = str(record["image_id"])
        if key not in source_rgb_cache:
            with Image.open(record["path"]).convert("RGB") as image:
                source_rgb_cache[key] = np.asarray(image).copy()
        return source_rgb_cache[key]

    for row in rows:
        status = row.get("match_status")
        if status not in valid_statuses:
            raise ValueError(f"Unsupported crop-manifest status: {status!r}")
        record = record_by_id.get(str(row.get("source_image_id")))
        if record is None:
            raise ValueError(f"Crop manifest references unknown image: {row.get('source_image_id')}")
        if str(row.get("source_file_name")) != str(record["file_name"]):
            raise ValueError(f"Crop manifest file name does not match image id: {row}")
        confidence_values.add(float(row["confidence_threshold"]))
        if row.get("test_annotations_sha256") is not None:
            annotation_hashes.add(str(row["test_annotations_sha256"]))
        if row.get("split_manifest_sha256") is not None:
            split_hashes.add(str(row["split_manifest_sha256"]))
        status_counts[status] += 1

        if status == "matched":
            crop_path = Path(row["crop_path"])
            if not crop_path.is_absolute():
                crop_path = path.parent / crop_path
            if not crop_path.is_file():
                raise FileNotFoundError(f"Missing detector crop: {crop_path}")
            actual_crop = np.asarray(Image.open(crop_path).convert("RGB")).copy()
            pair = {
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "image_path": str(record["path"]),
                "annotation_id": row.get("ground_truth_object_id"),
                "ground_truth_index": row.get("ground_truth_index"),
                "prediction_index": row.get("prediction_index"),
                "detector_confidence": float(row["prediction_confidence"]),
                "crop_iou": float(row["match_iou"]),
                "oracle_crop": oriented_crop(source_rgb(record), row["ground_truth_box"]).image,
                "actual_crop": actual_crop,
            }
            matched_pairs.append(pair)
            all_confidences.append(pair["detector_confidence"])
            all_ious.append(pair["crop_iou"])
        elif status == "missed_ground_truth":
            unmatched_gt.append({
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "annotation_id": row.get("ground_truth_object_id"),
                "ground_truth_index": row.get("ground_truth_index"),
            })
        else:
            confidence = float(row["prediction_confidence"])
            unmatched_predictions.append({
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "prediction_index": row.get("prediction_index"),
                "detector_confidence": confidence,
            })
            all_confidences.append(confidence)

    if len(confidence_values) != 1:
        raise ValueError("Crop manifest must use one confidence threshold")
    if annotation_hashes and annotation_hashes != {expected_test_annotations_sha256}:
        raise ValueError("Crop manifest test annotation hash does not match the evaluator input")
    return {
        "matched_pairs": matched_pairs,
        "unmatched_gt": unmatched_gt,
        "unmatched_predictions": unmatched_predictions,
        "all_confidences": all_confidences,
        "all_ious": all_ious,
        "confidence": confidence_values.pop(),
        "total_predictions": len(matched_pairs) + len(unmatched_predictions),
        "manifest": {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "status_counts": status_counts,
            "detector_name": sorted({str(row["detector_name"]) for row in rows}),
            "split_manifest_sha256": sorted(split_hashes),
            "test_annotations_sha256": sorted(annotation_hashes),
        },
    }


def predict_obb_records(
    model: Any,
    records: Sequence[dict[str, Any]],
    image_size: int,
    confidence: float,
    device: str | None,
) -> tuple[list[list[dict[str, Any]]], dict[str, float]]:
    options: dict[str, Any] = {
        "source": [str(record["path"]) for record in records],
        "task": "obb",
        "imgsz": image_size,
        "conf": confidence,
        "verbose": False,
    }
    if device is not None:
        options["device"] = device
    started = time.perf_counter()
    results = model.predict(**options)
    elapsed = time.perf_counter() - started
    if len(results) != len(records):
        raise RuntimeError(f"Detector returned {len(results)} results for {len(records)} images")
    all_predictions = []
    for result in results:
        predictions = []
        obb = getattr(result, "obb", None)
        if obb is not None:
            corners = obb.xyxyxyxy.detach().cpu().numpy()
            confidences = obb.conf.detach().cpu().numpy()
            classes = obb.cls.detach().cpu().numpy()
            for index, (box, conf, class_id) in enumerate(zip(corners, confidences, classes)):
                if int(class_id) != 0:
                    continue
                predictions.append(
                    {
                        "prediction_index": index,
                        "class_id": int(class_id),
                        "confidence": float(conf),
                        "corners": np.asarray(box, dtype=np.float32).tolist(),
                    }
                )
        all_predictions.append(predictions)
    return all_predictions, {
        "total_seconds": float(elapsed),
        "per_image_seconds": float(elapsed / len(records)) if records else 0.0,
        "per_image_milliseconds": float(1000.0 * elapsed / len(records)) if records else 0.0,
    }


def _crop_tensor(crop: np.ndarray, config: DINOv2Config) -> torch.Tensor:
    import torch

    image = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize(
        (config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def score_crop_pairs(
    pairs: Sequence[dict[str, Any]],
    encoder: SharedDINOv2,
    memory_bank: torch.Tensor,
    batch_size: int = 8,
    k: int = 1,
) -> list[dict[str, Any]]:
    """Score oracle and actual crops in batches using one frozen memory bank."""

    import torch
    from inspection.dinov2_patchcore_lite import aggregate_patch_scores
    from mvtec_system.anomaly.knn import nearest_neighbor_scores

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = [dict(pair) for pair in pairs]
    crops = []
    for pair in output:
        crops.extend([pair["oracle_crop"], pair["actual_crop"]])
    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start : start + batch_size]
        images = torch.stack([_crop_tensor(crop, encoder.config) for crop in batch_crops])
        with torch.inference_mode():
            features = encoder.forward(images).patch_embeddings
        patch_scores = nearest_neighbor_scores(
            features.reshape(-1, features.shape[-1]).float(), memory_bank.float(), k=k
        ).reshape(len(batch_crops), -1)
        for offset in range(len(batch_crops)):
            pair_index = (start + offset) // 2
            side = "oracle_score" if (start + offset) % 2 == 0 else "actual_score"
            output[pair_index][side] = {
                aggregation: aggregate_patch_scores(patch_scores[offset].numpy(), aggregation)
                for aggregation in AGGREGATIONS
            }
    return output


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _correlation(first: Sequence[float], second: Sequence[float]) -> dict[str, float | None]:
    if len(first) < 2:
        return {"pearson": None, "spearman": None}
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    pearson = float(np.corrcoef(first_array, second_array)[0, 1])
    spearman = float(np.corrcoef(rankdata(first_array), rankdata(second_array))[0, 1])
    return {
        "pearson": pearson if np.isfinite(pearson) else None,
        "spearman": spearman if np.isfinite(spearman) else None,
    }


def score_comparison_summary(
    pairs: Sequence[dict[str, Any]], aggregation: str, threshold: float
) -> dict[str, Any]:
    oracle = [float(pair["oracle_score"][aggregation]) for pair in pairs]
    actual = [float(pair["actual_score"][aggregation]) for pair in pairs]
    shift = [actual_value - oracle_value for actual_value, oracle_value in zip(actual, oracle)]

    def threshold_stats(scores: Sequence[float]) -> dict[str, Any]:
        above = np.asarray(scores) >= threshold
        return {
            **_stats(scores),
            "threshold": float(threshold),
            "above_threshold_count": int(above.sum()),
            "above_threshold_rate": float(above.mean()) if len(above) else None,
        }

    return {
        "threshold": float(threshold),
        "matched_pair_count": len(pairs),
        "oracle": threshold_stats(oracle),
        "actual_detector_crop": threshold_stats(actual),
        "score_shift_actual_minus_oracle": _stats(shift),
        "score_correlation": _correlation(oracle, actual),
        "crop_iou": _stats([float(pair["crop_iou"]) for pair in pairs]),
    }


def _manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(paths):
        resolved = path.resolve()
        entries.append(
            {
                "relative_path": resolved.relative_to(root.resolve()).as_posix(),
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"root": str(root.resolve()), "count": len(entries), "sha256": digest, "files": entries}


def _verified_entries(entries: Sequence[dict[str, Any]], label: str) -> list[Path]:
    paths = []
    for entry in entries:
        path = Path(entry["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"{label} hash changed: {path}")
        paths.append(path)
    return paths


def _frozen_memory_paths(patch_report: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    provenance = patch_report["provenance"]
    sources = provenance["memory_bank_sources"]
    mvtec_paths = _verified_entries(sources["mvtec_train_good"]["files"], "MVTec memory")
    cad_paths = _verified_entries(sources["cad_sd_train_0_good_memory"]["files"], "CAD memory")
    split_info = provenance["cad_good_split"]
    cad_all = _verified_entries(provenance["cad_sd_train_0_good"]["files"], "CAD train")
    cad_test = _verified_entries(provenance["cad_sd_test"]["files"], "CAD test")
    cad_root = Path(provenance["cad_sd_train_0_good"]["root"])
    split = deterministic_cad_good_split(
        cad_all,
        root=cad_root,
        memory_fraction=float(split_info["memory_fraction"]),
        salt=str(split_info["salt"]),
        excluded_paths=cad_test,
        memory_count_target=int(split_info["memory_count_target"]),
    )
    if split["split_hash"] != split_info["split_hash"]:
        raise ValueError("Frozen CAD content-hash split cannot be reproduced")
    if {path.resolve() for path in split["memory_paths"]} != {path.resolve() for path in cad_paths}:
        raise ValueError("Frozen CAD memory membership differs from the PatchCore report")
    return mvtec_paths + split["memory_paths"], {
        "mvtec_count": len(mvtec_paths),
        "cad_count": len(cad_paths),
        "split_hash": split["split_hash"],
        "reference_memory_sha256": patch_report["memory_bank"]["memory_sha256"],
    }


def evaluate_weee_detector_crop_shift(
    weights: str | Path = "runs/obb/runs/screw_yolo_obb/m4_fixed1024_10/weights/best.pt",
    weee_root: str | Path = "datasets/external/weee_screw",
    annotation_path: str | Path = "datasets/external/weee_screw/Annotations/Object Detection & Instance Segmentation COCO-format/Test_set.json",
    patchcore_report_path: str | Path = "reports/experiments/dinov2_patchcore_lite_mixed_contenthash_aggregations_rnd2.json",
    report_path: str | Path = "reports/experiments/weee_detector_crop_shift_patchcore_lite.json",
    image_size: int = 1024,
    confidence: float = 0.25,
    match_iou_threshold: float = 0.5,
    batch_size: int = 8,
    device: str | None = None,
    crop_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run frozen oracle-vs-detected crop scoring, optionally from saved crops."""

    from inspection.dinov2_patchcore_lite import (
        build_patch_memory_bank,
        deterministic_cad_good_split,
    )
    from inspection.geometry import oriented_crop
    from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device

    weights = Path(weights).resolve()
    weee_root = Path(weee_root).resolve()
    annotation_path = Path(annotation_path).resolve()
    patchcore_report_path = Path(patchcore_report_path).resolve()
    patch_report = json.loads(patchcore_report_path.read_text(encoding="utf-8"))
    if patch_report.get("protocol") != "mixed_nominal" or not patch_report["provenance"]["test_locked"]:
        raise ValueError("PatchCore report must be the locked mixed-nominal protocol")
    memory_paths, memory_provenance = _frozen_memory_paths(patch_report)
    encoder_metadata = patch_report["provenance"]["encoder"]
    config = DINOv2Config(
        model_name=encoder_metadata["model_name"],
        model_revision=encoder_metadata["model_revision"],
        patch_feature_key=encoder_metadata["patch_feature_key"],
        global_feature_key=encoder_metadata["global_feature_key"],
        image_size=tuple(encoder_metadata["image_size"]),
        mean=tuple(encoder_metadata["mean"]),
        std=tuple(encoder_metadata["std"]),
        l2_normalize_patches=bool(encoder_metadata["l2_normalize_patches"]),
        feature_layers=tuple(encoder_metadata["feature_layers"]),
    )
    runtime_device = device or str(select_device())
    encoder = SharedDINOv2(config, artifact_root="artifacts/dinov2_knn", device=runtime_device)
    memory_bank = build_patch_memory_bank(memory_paths, encoder, batch_size=batch_size)
    rebuilt_memory_hash = hashlib.sha256(memory_bank.numpy().tobytes()).hexdigest()

    records = load_weee_records(weee_root, annotation_path)
    manifest_info = None
    if crop_manifest is not None:
        loaded_manifest = _load_detector_crop_manifest(
            crop_manifest, records, sha256_file(annotation_path)
        )
        matched_pairs = loaded_manifest["matched_pairs"]
        unmatched_gt = loaded_manifest["unmatched_gt"]
        unmatched_predictions = loaded_manifest["unmatched_predictions"]
        all_confidences = loaded_manifest["all_confidences"]
        all_ious = loaded_manifest["all_ious"]
        confidence = float(loaded_manifest["confidence"])
        total_predictions = int(loaded_manifest["total_predictions"])
        manifest_info = loaded_manifest["manifest"]
        detector_latency = {
            "total_seconds": 0.0,
            "per_image_seconds": 0.0,
            "per_image_milliseconds": 0.0,
            "source": "precomputed_crop_manifest",
            "inference_skipped": True,
        }
    else:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        predictions, detector_latency = predict_obb_records(model, records, image_size, confidence, device)
        matched_pairs = []
        unmatched_gt = []
        unmatched_predictions = []
        all_confidences = []
        all_ious = []
        for record, image_predictions in zip(records, predictions):
            all_confidences.extend(float(prediction["confidence"]) for prediction in image_predictions)
            matching = match_objects(record["ground_truth"], image_predictions, match_iou_threshold)
            for match in matching["matches"]:
                gt = record["ground_truth"][match["gt_index"]]
                prediction = image_predictions[match["prediction_index"]]
                image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Unable to read WEEE image: {record['path']}")
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                matched_pairs.append(
                    {
                        "image_id": record["image_id"],
                        "file_name": record["file_name"],
                        "image_path": str(record["path"]),
                        "annotation_id": gt["annotation_id"],
                        "ground_truth_index": match["gt_index"],
                        "prediction_index": prediction["prediction_index"],
                        "detector_confidence": prediction["confidence"],
                        "crop_iou": match["crop_iou"],
                        "oracle_crop": oriented_crop(rgb, gt["corners"]).image,
                        "actual_crop": oriented_crop(rgb, prediction["corners"]).image,
                    }
                )
                all_ious.append(match["crop_iou"])
            for gt_index in matching["unmatched_gt_indices"]:
                gt = record["ground_truth"][gt_index]
                unmatched_gt.append(
                    {
                        "image_id": record["image_id"],
                        "file_name": record["file_name"],
                        "annotation_id": gt["annotation_id"],
                        "ground_truth_index": gt_index,
                    }
                )
            for prediction_index in matching["unmatched_prediction_indices"]:
                prediction = image_predictions[prediction_index]
                unmatched_predictions.append(
                    {
                        "image_id": record["image_id"],
                        "file_name": record["file_name"],
                        "prediction_index": prediction["prediction_index"],
                        "detector_confidence": prediction["confidence"],
                    }
                )
        total_predictions = sum(len(item) for item in predictions)
    scored_pairs = score_crop_pairs(matched_pairs, encoder, memory_bank, batch_size=batch_size, k=int(patch_report["memory_bank"]["k"]))
    strategies = patch_report["score_aggregation_comparison"]["strategies"]
    score_reports = {
        aggregation: score_comparison_summary(
            scored_pairs, aggregation, float(strategies[aggregation]["calibration"]["threshold"])
        )
        for aggregation in AGGREGATIONS
    }
    matched_rows = []
    for pair in scored_pairs:
        matched_rows.append(
            {
                key: value
                for key, value in pair.items()
                if key not in {"oracle_crop", "actual_crop"}
            }
        )
    all_image_paths = [record["path"] for record in records]
    total_gt = sum(len(record["ground_truth"]) for record in records)
    report = {
        "schema_version": "1.0",
        "experiment": "weee_detector_crop_shift_patchcore_lite",
        "dataset_semantics": {
            "dataset": "WEEE Test_set.json",
            "label_status": "GOOD/domain-shift stress only",
            "defect_labels_available": False,
            "defect_recall_reported": False,
            "no_weee_recalibration": True,
        },
        "detector": {
            "task": "obb",
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "image_size": image_size,
            "confidence_threshold": confidence,
            "match_iou_threshold": match_iou_threshold,
            "image_count": len(records),
            "ground_truth_object_count": total_gt,
            "prediction_count": total_predictions,
            "matched_count": len(matched_rows),
            "unmatched_ground_truth_count": len(unmatched_gt),
            "unmatched_prediction_count": len(unmatched_predictions),
            "ground_truth_match_rate": len(matched_rows) / total_gt if total_gt else None,
            "ground_truth_miss_rate": len(unmatched_gt) / total_gt if total_gt else None,
            "prediction_match_rate": len(matched_rows) / total_predictions if total_predictions else None,
            "latency": detector_latency,
            "crop_source": "precomputed_manifest" if manifest_info is not None else "live_detector_inference",
            "crop_manifest": manifest_info,
            "confidence": _stats(all_confidences),
            "matched_crop_iou": _stats(all_ious),
        },
        "patchcore_protocol": {
            "source_report": str(patchcore_report_path),
            "source_report_sha256": sha256_file(patchcore_report_path),
            "protocol": patch_report["protocol"],
            "test_locked": patch_report["provenance"]["test_locked"],
            "no_weee_recalibration": True,
            "encoder": encoder.metadata(),
            "preprocessing": {
                "image_size": list(config.image_size),
                "mean": list(config.mean),
                "std": list(config.std),
                "l2_normalize_patches": config.l2_normalize_patches,
            },
            "memory_bank": {
                **memory_provenance,
                "rebuilt_memory_sha256": rebuilt_memory_hash,
                "memory_sha256_match": rebuilt_memory_hash == memory_provenance["reference_memory_sha256"],
                "image_count": int(memory_bank.shape[0] // encoder.patch_count),
                "patch_count": int(memory_bank.shape[0]),
                "embedding_dimension": int(memory_bank.shape[1]),
                "k": patch_report["memory_bank"]["k"],
            },
            "calibration": {
                aggregation: {
                    "source": strategies[aggregation]["calibration"]["source"],
                    "threshold": strategies[aggregation]["calibration"]["threshold"],
                    "allowed_good_fpr": strategies[aggregation]["calibration"]["allowed_good_fpr"],
                    "rule": strategies[aggregation]["calibration"]["rule"],
                }
                for aggregation in AGGREGATIONS
            },
            "aggregation_candidates": list(AGGREGATIONS),
        },
        "score_comparison": score_reports,
        "unmatched": {
            "ground_truth": unmatched_gt,
            "predictions": unmatched_predictions,
        },
        "matched_records": matched_rows,
        "provenance": {
            "weee_root": str(weee_root),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "images": _manifest(all_image_paths, weee_root),
            "runtime_device": runtime_device,
            "crop_manifest": manifest_info,
        },
        "warnings": [
            "WEEE has screw geometry annotations but no GOOD/DEFECT labels; no defect recall is reported.",
            "Results are GOOD/domain-shift stress only and must not be treated as a WEEE anomaly benchmark.",
            "Unmatched GT objects and unmatched detector predictions are excluded from oracle-vs-actual crop score correlation.",
            "Thresholds and memory membership are copied from the locked CAD-SD mixed-nominal PatchCore-lite report; WEEE is not used for calibration.",
        ],
    }
    if manifest_info is not None:
        report["warnings"].append(
            "Detector inference was skipped; matched, missed and unmatched objects came from the hashed crop manifest."
        )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("runs/obb/runs/screw_yolo_obb/m4_fixed1024_10/weights/best.pt"))
    parser.add_argument("--weee-root", type=Path, default=Path("datasets/external/weee_screw"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/external/weee_screw/Annotations/Object Detection & Instance Segmentation COCO-format/Test_set.json"),
    )
    parser.add_argument(
        "--patchcore-report",
        type=Path,
        default=Path("reports/experiments/dinov2_patchcore_lite_mixed_contenthash_aggregations_rnd2.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/weee_detector_crop_shift_patchcore_lite.json"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--crop-manifest", type=Path, help="Use a previously generated detector crop manifest")
    args = parser.parse_args()
    report = evaluate_weee_detector_crop_shift(
        weights=args.weights,
        weee_root=args.weee_root,
        annotation_path=args.annotations,
        patchcore_report_path=args.patchcore_report,
        report_path=args.output,
        image_size=args.imgsz,
        confidence=args.conf,
        match_iou_threshold=args.match_iou,
        batch_size=args.batch_size,
        device=args.device,
        crop_manifest=args.crop_manifest,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
