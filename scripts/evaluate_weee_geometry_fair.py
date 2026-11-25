#!/usr/bin/env python3
"""Compare YOLO-OBB and RT-DETR under one WEEE AABB holdout protocol.

The two models expose different geometries.  This evaluator projects both
outputs to enclosing axis-aligned boxes before matching, so recall, matched
precision, crop IoU and latency are directly comparable.  It uses validation
only for confidence selection and keeps the WEEE test annotations locked until
the selected operating point is applied once.

This is a detector/crop comparison, not an OBB-quality benchmark.  The
separate YOLO-OBB polygon metrics remain useful for production crop quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    from calibrate_weee_detector_confidence import select_operating_point
    from evaluate_weee_detector_crop_shift import load_weee_records
except ImportError:
    from scripts.calibrate_weee_detector_confidence import select_operating_point
    from scripts.evaluate_weee_detector_crop_shift import load_weee_records


DEFAULT_GRID = tuple(round(value, 2) for value in np.arange(0.05, 0.81, 0.05))


def _avoid_slow_torchvision_warmup_import() -> None:
    """Skip Ultralytics' optional full torchvision import during warmup on macOS."""

    from ultralytics.nn.autobackend import AutoBackend

    if getattr(AutoBackend, "_codex_warmup_patch", False):
        return
    original_warmup = AutoBackend.warmup

    def warmup_without_torchvision(self: Any, *args: Any, **kwargs: Any) -> None:
        # Ultralytics' warmup imports every torchvision dataset module only to make
        # torchvision NMS available; its NMS function already has a torch fallback.
        previous_end2end = self.end2end
        self.end2end = True
        try:
            original_warmup(self, *args, **kwargs)
        finally:
            self.end2end = previous_end2end

    AutoBackend.warmup = warmup_without_torchvision
    AutoBackend._codex_warmup_patch = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enclosing_aabb(corners: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return a four-corner axis-aligned box enclosing ``corners``."""

    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("corners must contain at least four finite (x, y) points")
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("corners must form a positive-area box")
    return [[float(x_min), float(y_min)], [float(x_max), float(y_min)],
            [float(x_max), float(y_max)], [float(x_min), float(y_max)]]


def project_records_to_aabb(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy WEEE records and project all polygon GT boxes to AABB geometry."""

    output = []
    for record in records:
        copied = dict(record)
        copied["ground_truth"] = [
            {**object_row, "corners": enclosing_aabb(object_row["corners"])}
            for object_row in record["ground_truth"]
        ]
        output.append(copied)
    return output


def predict_aabb_records(
    model_kind: str,
    weights: str | Path,
    records: Sequence[dict[str, Any]],
    image_size: int,
    confidence: float,
    device: str | None,
) -> tuple[list[list[dict[str, Any]]], dict[str, float]]:
    """Run either OBB YOLO or AABB RT-DETR and normalize output geometry."""

    from ultralytics import RTDETR, YOLO

    _avoid_slow_torchvision_warmup_import()
    weights = Path(weights).resolve()
    model = YOLO(str(weights)) if model_kind == "yolo_obb" else RTDETR(str(weights))
    options: dict[str, Any] = {
        "source": [str(record["path"]) for record in records],
        "task": "obb" if model_kind == "yolo_obb" else "detect",
        "imgsz": image_size,
        "conf": confidence,
        "verbose": False,
        "workers": 0,
        "batch": 1,
    }
    if device is not None:
        options["device"] = device
    all_predictions: list[list[dict[str, Any]]] = []
    started = time.perf_counter()
    # RT-DETR at 1024px can retain a large MPS workspace.  Keep inference
    # chunks small and materialize only CPU-side rows before the next chunk.
    for start in range(0, len(records), 8):
        chunk = records[start : start + 8]
        chunk_options = {**options, "source": [str(record["path"]) for record in chunk]}
        results = model.predict(**chunk_options)
        if len(results) != len(chunk):
            raise RuntimeError(f"{model_kind} returned {len(results)} results for {len(chunk)} images")
        for result in results:
            rows: list[dict[str, Any]] = []
            if model_kind == "yolo_obb":
                output = getattr(result, "obb", None)
                if output is not None:
                    boxes = output.xyxyxyxy.detach().cpu().numpy()
                    confidences = output.conf.detach().cpu().numpy()
                    classes = output.cls.detach().cpu().numpy()
                    for index, (box, conf, class_id) in enumerate(zip(boxes, confidences, classes)):
                        if int(class_id) != 0:
                            continue
                        rows.append({
                            "prediction_index": index,
                            "class_id": int(class_id),
                            "confidence": float(conf),
                            "corners": enclosing_aabb(box),
                        })
            else:
                output = getattr(result, "boxes", None)
                if output is not None:
                    boxes = output.xyxy.detach().cpu().numpy()
                    confidences = output.conf.detach().cpu().numpy()
                    classes = output.cls.detach().cpu().numpy()
                    for index, (box, conf, class_id) in enumerate(zip(boxes, confidences, classes)):
                        if int(class_id) != 0:
                            continue
                        x1, y1, x2, y2 = [float(value) for value in box[:4]]
                        rows.append({
                            "prediction_index": index,
                            "class_id": int(class_id),
                            "confidence": float(conf),
                            "corners": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        })
            all_predictions.append(rows)
        del results
    elapsed = time.perf_counter() - started
    return all_predictions, {
        "total_seconds": float(elapsed),
        "per_image_seconds": float(elapsed / len(records)) if records else 0.0,
        "per_image_milliseconds": float(1000.0 * elapsed / len(records)) if records else 0.0,
    }


def _iou(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    first_x1, first_y1 = first_array.min(axis=0)
    first_x2, first_y2 = first_array.max(axis=0)
    second_x1, second_y1 = second_array.min(axis=0)
    second_x2, second_y2 = second_array.max(axis=0)
    intersection = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1)) * max(
        0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
    )
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def match_aabb(
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    candidates = sorted(
        (_iou(gt["corners"], prediction["corners"]), gt_index, prediction_index)
        for gt_index, gt in enumerate(ground_truth)
        for prediction_index, prediction in enumerate(predictions)
    )[::-1]
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for iou, gt_index, prediction_index in candidates:
        if iou < iou_threshold or gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append({"gt_index": gt_index, "prediction_index": prediction_index, "crop_iou": float(iou)})
    return {
        "matches": matches,
        "unmatched_gt_indices": [i for i in range(len(ground_truth)) if i not in used_gt],
        "unmatched_prediction_indices": [i for i in range(len(predictions)) if i not in used_predictions],
    }


def score_predictions(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[Sequence[dict[str, Any]]],
    confidence: float,
    latency: dict[str, float],
) -> dict[str, Any]:
    gt_count = sum(len(record["ground_truth"]) for record in records)
    prediction_count = matched_count = unmatched_gt = unmatched_predictions = 0
    matched_ious: list[float] = []
    for record, rows in zip(records, predictions):
        filtered = [row for row in rows if float(row["confidence"]) >= confidence]
        prediction_count += len(filtered)
        matching = match_aabb(record["ground_truth"], filtered)
        matched_count += len(matching["matches"])
        unmatched_gt += len(matching["unmatched_gt_indices"])
        unmatched_predictions += len(matching["unmatched_prediction_indices"])
        matched_ious.extend(float(row["crop_iou"]) for row in matching["matches"])
    return {
        "confidence_threshold": float(confidence),
        "image_count": len(records),
        "ground_truth_object_count": gt_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "unmatched_ground_truth_count": unmatched_gt,
        "unmatched_prediction_count": unmatched_predictions,
        "ground_truth_recall": matched_count / gt_count if gt_count else None,
        "prediction_match_precision": matched_count / prediction_count if prediction_count else None,
        "unmatched_prediction_rate": unmatched_predictions / prediction_count if prediction_count else None,
        "matched_crop_iou_mean": float(np.mean(matched_ious)) if matched_ious else None,
        "matched_crop_iou_p05": float(np.quantile(matched_ious, 0.05)) if matched_ious else None,
        "latency_ms_per_image": latency["per_image_milliseconds"],
    }


def write_detector_crop_manifest(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[Sequence[dict[str, Any]]],
    confidence: float,
    detector_name: str,
    output: str | Path,
    split_manifest_sha256: str,
    dataset_provenance_sha256: str | None,
    test_annotations_sha256: str,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Persist matched, missed and unmatched detector objects with real crops."""

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_root = output.parent / f"{output.stem}_crops" / detector_name
    rows: list[dict[str, Any]] = []
    status_counts = {"matched": 0, "missed_ground_truth": 0, "unmatched_prediction": 0}
    if len(records) != len(predictions):
        raise ValueError(f"Detector returned {len(predictions)} image results for {len(records)} records")
    for image_index, (record, image_predictions) in enumerate(zip(records, predictions)):
        filtered = [row for row in image_predictions if float(row["confidence"]) >= confidence]
        matching = match_aabb(record["ground_truth"], filtered, iou_threshold=iou_threshold)
        matches_by_gt = {row["gt_index"]: row for row in matching["matches"]}
        matches_by_prediction = {row["prediction_index"]: row for row in matching["matches"]}

        def crop_path_for(prediction_index: int) -> str:
            destination = crop_root / f"image_{image_index:04d}" / f"prediction_{prediction_index:04d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(record["path"]).convert("RGB") as image:
                points = np.asarray(filtered[prediction_index]["corners"], dtype=np.float32)
                left, top = np.floor(points.min(axis=0)).astype(int)
                right, bottom = np.ceil(points.max(axis=0)).astype(int)
                left = max(0, min(left, image.width - 1))
                top = max(0, min(top, image.height - 1))
                right = max(left + 1, min(right, image.width))
                bottom = max(top + 1, min(bottom, image.height))
                image.crop((left, top, right, bottom)).save(destination)
            return str(destination.relative_to(output.parent))

        def base_row(status: str, prediction_index: int | None, gt_index: int | None) -> dict[str, Any]:
            prediction = filtered[prediction_index] if prediction_index is not None else None
            ground_truth = record["ground_truth"][gt_index] if gt_index is not None else None
            match = matches_by_prediction.get(prediction_index) if prediction_index is not None else None
            return {
                "source_image_id": record["image_id"],
                "source_file_name": record["file_name"],
                "prediction_index": prediction_index,
                "ground_truth_index": gt_index,
                "detector_name": detector_name,
                "confidence_threshold": float(confidence),
                "predicted_box": prediction["corners"] if prediction is not None else None,
                "ground_truth_object_id": ground_truth.get("annotation_id") if ground_truth is not None else None,
                "ground_truth_box": ground_truth["corners"] if ground_truth is not None else None,
                "prediction_confidence": float(prediction["confidence"]) if prediction is not None else None,
                "match_iou": match["crop_iou"] if match is not None else None,
                "crop_path": crop_path_for(prediction_index) if prediction_index is not None else None,
                "match_status": status,
                "split_manifest_sha256": split_manifest_sha256,
                "dataset_provenance_sha256": dataset_provenance_sha256,
                "test_annotations_sha256": test_annotations_sha256,
            }

        for gt_index, match in matches_by_gt.items():
            rows.append(base_row("matched", match["prediction_index"], gt_index))
            status_counts["matched"] += 1
        for prediction_index in matching["unmatched_prediction_indices"]:
            rows.append(base_row("unmatched_prediction", prediction_index, None))
            status_counts["unmatched_prediction"] += 1
        for gt_index in matching["unmatched_gt_indices"]:
            rows.append(base_row("missed_ground_truth", None, gt_index))
            status_counts["missed_ground_truth"] += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "manifest_path": output.name,
        "manifest_sha256": sha256_file(output),
        "crop_root": str(crop_root.relative_to(output.parent)),
        "record_count": len(rows),
        "status_counts": status_counts,
        "iou_threshold": float(iou_threshold),
    }


def _load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload["source_split_manifest"]
    if not manifest.get("content_disjoint", True):
        raise ValueError(f"Source split is not marked content-disjoint: {path}")
    return manifest


def evaluate(
    yolo_weights: str | Path,
    rtdetr_weights: str | Path,
    weee_root: str | Path,
    validation_annotations: str | Path,
    test_annotations: str | Path,
    yolo_manifest: str | Path,
    rtdetr_manifest: str | Path,
    output: str | Path,
    image_size: int = 1024,
    device: str | None = None,
    grid: Sequence[float] = DEFAULT_GRID,
) -> dict[str, Any]:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    yolo_manifest_path = Path(yolo_manifest).resolve()
    rtdetr_manifest_path = Path(rtdetr_manifest).resolve()
    yolo_split_manifest = _load_split_manifest(yolo_manifest_path)
    rtdetr_split_manifest = _load_split_manifest(rtdetr_manifest_path)
    if yolo_split_manifest["manifest_sha256"] != rtdetr_split_manifest["manifest_sha256"]:
        raise ValueError("YOLO and RT-DETR source split hashes differ")

    root = Path(weee_root).resolve()
    validation = project_records_to_aabb(load_weee_records(root, Path(validation_annotations).resolve()))
    test = project_records_to_aabb(load_weee_records(root, Path(test_annotations).resolve()))
    models = {
        "yolo_obb_projected_aabb": ("yolo_obb", Path(yolo_weights).resolve()),
        "rtdetr_aabb": ("rtdetr_aabb", Path(rtdetr_weights).resolve()),
    }
    report_models: dict[str, Any] = {}
    for label, (model_kind, weights) in models.items():
        min_conf = min(float(value) for value in grid)
        val_predictions, val_latency = predict_aabb_records(model_kind, weights, validation, image_size, min_conf, device)
        validation_sweep = [
            score_predictions(validation, val_predictions, float(confidence), val_latency)
            for confidence in sorted(set(float(value) for value in grid), reverse=True)
        ]
        selection = select_operating_point(validation_sweep, baseline_confidence=0.25)
        selected_confidence = float(selection["selected"]["confidence_threshold"])
        test_predictions, test_latency = predict_aabb_records(model_kind, weights, test, image_size, selected_confidence, device)
        crop_manifest = write_detector_crop_manifest(
            test,
            test_predictions,
            selected_confidence,
            label,
            output.with_name(f"{output.stem}_{label}_crop_manifest.jsonl"),
            yolo_split_manifest["manifest_sha256"],
            None,
            sha256_file(Path(test_annotations).resolve()),
        )
        locked_test = score_predictions(test, test_predictions, selected_confidence, test_latency)
        locked_test["crop_manifest"] = crop_manifest
        report_models[label] = {
            "model_kind": model_kind,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "training_protocol": "WEEE train/validation only; 5 epochs, imgsz=1024, batch=2, seed=0, device=mps",
            "validation_sweep": validation_sweep,
            "selection": selection,
            "selected_confidence": selected_confidence,
            "locked_test": locked_test,
        }

    report = {
        "schema_version": "1.0",
        "experiment": "weee_yolo_obb_rtdetr_fair_aabb_holdout",
        "protocol": {
            "geometry": "AABB for both models; YOLO-OBB predictions projected to enclosing AABB",
            "same_source_split_manifest_sha256": yolo_split_manifest["manifest_sha256"],
            "same_image_splits": True,
            "same_image_size": image_size,
            "same_confidence_grid": list(sorted(set(float(value) for value in grid))),
            "same_iou_rule": 0.5,
            "validation_selects_threshold": True,
            "test_locked": True,
            "validation_annotations_sha256": sha256_file(Path(validation_annotations).resolve()),
            "test_annotations_sha256": sha256_file(Path(test_annotations).resolve()),
            "split_manifest_sources": {
                "yolo_obb": str(yolo_manifest_path),
                "rtdetr_aabb": str(rtdetr_manifest_path),
            },
        },
        "dataset": {
            "root": str(root),
            "validation_images": len(validation),
            "validation_objects": sum(len(row["ground_truth"]) for row in validation),
            "locked_test_images": len(test),
            "locked_test_objects": sum(len(row["ground_truth"]) for row in test),
        },
        "models": report_models,
        "warnings": [
            "The common score is AABB crop coverage, not oriented-box quality.",
            "This comparison does not report anomaly or defect metrics because WEEE has geometry labels only.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--rtdetr-weights", type=Path, required=True)
    parser.add_argument("--weee-root", type=Path, default=Path("datasets/external/weee_screw"))
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--yolo-manifest", type=Path, required=True)
    parser.add_argument("--rtdetr-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device")
    parser.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    args = parser.parse_args()
    result = evaluate(
        yolo_weights=args.yolo_weights,
        rtdetr_weights=args.rtdetr_weights,
        weee_root=args.weee_root,
        validation_annotations=args.validation_annotations,
        test_annotations=args.test_annotations,
        yolo_manifest=args.yolo_manifest,
        rtdetr_manifest=args.rtdetr_manifest,
        output=args.output,
        image_size=args.imgsz,
        device=args.device,
        grid=args.grid,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
