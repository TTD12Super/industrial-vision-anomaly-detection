#!/usr/bin/env python3
"""Select a WEEE detector confidence threshold on validation only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from evaluate_weee_detector_crop_shift import (
        load_weee_records,
        match_objects,
        predict_obb_records,
        sha256_file,
    )
except ImportError:
    from scripts.evaluate_weee_detector_crop_shift import (
        load_weee_records,
        match_objects,
        predict_obb_records,
        sha256_file,
    )


DEFAULT_GRID = tuple(round(value, 2) for value in np.arange(0.05, 0.81, 0.05))


def filter_predictions(predictions: Sequence[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    """Filter one image's predictions without rerunning the detector."""

    return [prediction for prediction in predictions if float(prediction["confidence"]) >= confidence]


def score_predictions(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[Sequence[dict[str, Any]]],
    confidence: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Compute detector-only metrics for one confidence operating point."""

    gt_count = sum(len(record["ground_truth"]) for record in records)
    prediction_count = 0
    matched_count = 0
    unmatched_ground_truth = 0
    unmatched_predictions = 0
    matched_ious: list[float] = []
    for record, image_predictions in zip(records, predictions):
        filtered = filter_predictions(image_predictions, confidence)
        prediction_count += len(filtered)
        matching = match_objects(record["ground_truth"], filtered, iou_threshold=0.5)
        matched_count += len(matching["matches"])
        unmatched_ground_truth += len(matching["unmatched_gt_indices"])
        unmatched_predictions += len(matching["unmatched_prediction_indices"])
        matched_ious.extend(float(item["crop_iou"]) for item in matching["matches"])
    return {
        "confidence_threshold": float(confidence),
        "image_count": len(records),
        "ground_truth_object_count": gt_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "unmatched_ground_truth_count": unmatched_ground_truth,
        "unmatched_prediction_count": unmatched_predictions,
        "ground_truth_recall": matched_count / gt_count if gt_count else None,
        "prediction_match_precision": matched_count / prediction_count if prediction_count else None,
        "unmatched_prediction_rate": unmatched_predictions / prediction_count if prediction_count else None,
        "matched_crop_iou_mean": float(np.mean(matched_ious)) if matched_ious else None,
        "matched_crop_iou_p05": float(np.quantile(matched_ious, 0.05)) if matched_ious else None,
        "latency_ms_per_image": (elapsed_seconds * 1000.0 / len(records)) if records else None,
    }


def select_operating_point(
    rows: Sequence[dict[str, Any]],
    baseline_confidence: float = 0.25,
    recall_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Select from validation rows without consulting a locked test split."""

    if not rows:
        raise ValueError("validation sweep is empty")
    baseline = min(rows, key=lambda row: abs(float(row["confidence_threshold"]) - baseline_confidence))
    iou_floor = baseline["matched_crop_iou_mean"]
    unmatched_rate_ceiling = baseline["unmatched_prediction_rate"]
    candidates = [
        row
        for row in rows
        if row["ground_truth_recall"] is not None
        and row["matched_crop_iou_mean"] is not None
        and float(row["matched_crop_iou_mean"]) >= float(iou_floor)
        and float(row["unmatched_prediction_rate"] or 1.0) <= float(unmatched_rate_ceiling)
    ]
    if not candidates:
        return {
            "selected": baseline,
            "baseline": baseline,
            "iou_floor": iou_floor,
            "unmatched_rate_ceiling": unmatched_rate_ceiling,
            "rule": "fallback to baseline because no candidate preserved validation IoU and unmatched rate",
        }
    max_recall = max(float(row["ground_truth_recall"]) for row in candidates)
    near_max = [
        row
        for row in candidates
        if float(row["ground_truth_recall"]) >= max_recall - recall_tolerance
    ]
    selected = min(
        near_max,
        key=lambda row: (
            float(row["unmatched_prediction_rate"] or 1.0),
            -float(row["ground_truth_recall"]),
            float(row["confidence_threshold"]),
        ),
    )
    return {
        "selected": selected,
        "baseline": baseline,
        "iou_floor": iou_floor,
        "unmatched_rate_ceiling": unmatched_rate_ceiling,
        "max_validation_recall": max_recall,
        "recall_tolerance": recall_tolerance,
        "rule": (
            "maximize validation GT recall while preserving baseline validation matched IoU and "
            "not increasing unmatched prediction rate; within 1 percentage point of maximum recall, "
            "minimize unmatched prediction rate"
        ),
    }


def run_calibration(
    weights: str | Path,
    weee_root: str | Path,
    validation_annotations: str | Path,
    test_annotations: str | Path,
    output: str | Path,
    test_output: str | Path,
    image_size: int = 1024,
    grid: Sequence[float] = DEFAULT_GRID,
    baseline_confidence: float = 0.25,
    match_iou_threshold: float = 0.5,
    device: str | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    weights = Path(weights).resolve()
    root = Path(weee_root).resolve()
    validation_annotations = Path(validation_annotations).resolve()
    test_annotations = Path(test_annotations).resolve()
    validation_records = load_weee_records(root, validation_annotations)
    model = YOLO(str(weights))
    minimum_confidence = min(float(value) for value in grid)
    started = time.perf_counter()
    validation_predictions, detector_latency = predict_obb_records(
        model, validation_records, image_size, minimum_confidence, device
    )
    elapsed = detector_latency["total_seconds"] if detector_latency else time.perf_counter() - started
    sweep = [
        score_predictions(validation_records, validation_predictions, float(confidence), elapsed)
        for confidence in sorted(set(float(value) for value in grid), reverse=True)
    ]
    selection = select_operating_point(sweep, baseline_confidence=baseline_confidence)
    selected_confidence = float(selection["selected"]["confidence_threshold"])
    test_report = None
    if test_annotations.is_file():
        try:
            from evaluate_weee_detector_crop_shift import evaluate_weee_detector_crop_shift
        except ImportError:
            from scripts.evaluate_weee_detector_crop_shift import evaluate_weee_detector_crop_shift

        test_report = evaluate_weee_detector_crop_shift(
            weights=weights,
            weee_root=root,
            annotation_path=test_annotations,
            report_path=test_output,
            image_size=image_size,
            confidence=selected_confidence,
            match_iou_threshold=match_iou_threshold,
            device=device,
        )
    report = {
        "schema_version": "1.0",
        "experiment": "weee_detector_confidence_calibration",
        "selection_split": "WEEE validation only",
        "test_locked": True,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "validation_annotations": str(validation_annotations),
        "validation_annotations_sha256": sha256_file(validation_annotations),
        "test_annotations": str(test_annotations),
        "test_annotations_sha256": sha256_file(test_annotations),
        "image_size": image_size,
        "match_iou_threshold": match_iou_threshold,
        "confidence_grid": list(sorted(set(float(value) for value in grid), reverse=True)),
        "validation_sweep": sweep,
        "selection": selection,
        "selected_confidence": selected_confidence,
        "locked_test_report": test_report,
        "warnings": [
            "WEEE provides object geometry only; this report has no anomaly or defect metric.",
            "The threshold was selected from validation and applied once to the locked test split.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weee-root", type=Path, default=Path("datasets/external/weee_screw"))
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/weee_detector_confidence_calibration.json"))
    parser.add_argument("--test-output", type=Path, default=Path("reports/experiments/weee_detector_confidence_selected_test.json"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--baseline-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(json.dumps(run_calibration(
        weights=args.weights,
        weee_root=args.weee_root,
        validation_annotations=args.validation_annotations,
        test_annotations=args.test_annotations,
        output=args.output,
        test_output=args.test_output,
        image_size=args.imgsz,
        grid=args.grid,
        baseline_confidence=args.baseline_conf,
        match_iou_threshold=args.match_iou,
        device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
