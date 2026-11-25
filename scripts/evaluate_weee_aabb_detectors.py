#!/usr/bin/env python3
"""Compare YOLO and RT-DETR on identical WEEE AABB splits and gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


GRID = tuple(round(value, 2) for value in np.arange(0.05, 0.81, 0.05))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyxy_corners(box: Sequence[float]) -> list[list[float]]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def predict_aabb_records(model: Any, records: Sequence[dict[str, Any]], image_size: int, confidence: float, device: str | None) -> tuple[list[list[dict[str, Any]]], dict[str, float]]:
    # Keep inference chunks small: RT-DETR retains substantial MPS workspace at
    # 1024px, and evaluating the full list at once can exhaust shared memory.
    chunk_size = 8
    common_options: dict[str, Any] = {
        "task": "detect",
        "imgsz": image_size,
        "conf": confidence,
        "verbose": False,
        # Ultralytics' default workers can deadlock on macOS/MPS during a
        # subprocess benchmark. Keep evaluation deterministic and single-process.
        "workers": 0,
        "batch": 1,
    }
    if device is not None:
        common_options["device"] = device
    started = time.perf_counter()
    predictions = []
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        results = model.predict(
            source=[str(record["path"]) for record in chunk],
            **common_options,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"Detector returned {len(results)} results for {len(chunk)} images")
        for result in results:
            rows = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                corners = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy()
                for index, (box, conf, class_id) in enumerate(zip(corners, confidences, classes)):
                    if int(class_id) != 0:
                        continue
                    rows.append({
                        "prediction_index": index,
                        "class_id": int(class_id),
                        "confidence": float(conf),
                        "corners": _xyxy_corners(box),
                    })
            predictions.append(rows)
        # Results are deliberately converted to CPU rows before the next chunk.
        del results
    elapsed = time.perf_counter() - started
    return predictions, {
        "total_seconds": float(elapsed),
        "per_image_seconds": float(elapsed / len(records)) if records else 0.0,
        "per_image_milliseconds": float(1000.0 * elapsed / len(records)) if records else 0.0,
    }


def evaluate_detector(model_kind: str, weights: str | Path, validation_annotations: str | Path, test_annotations: str | Path, weee_root: str | Path, image_size: int, device: str | None, grid: Sequence[float] = GRID) -> dict[str, Any]:
    try:
        from calibrate_weee_detector_confidence import score_predictions, select_operating_point
        from evaluate_weee_detector_crop_shift import load_weee_records
    except ImportError:
        from scripts.calibrate_weee_detector_confidence import score_predictions, select_operating_point
        from scripts.evaluate_weee_detector_crop_shift import load_weee_records
    from ultralytics import RTDETR, YOLO

    weights = Path(weights).resolve()
    root = Path(weee_root).resolve()
    validation = load_weee_records(root, Path(validation_annotations).resolve())
    test = load_weee_records(root, Path(test_annotations).resolve())
    model = YOLO(str(weights)) if model_kind == "yolo" else RTDETR(str(weights))
    minimum = min(float(value) for value in grid)
    validation_predictions, validation_latency = predict_aabb_records(model, validation, image_size, minimum, device)
    validation_rows = [
        score_predictions(validation, validation_predictions, float(confidence), validation_latency["total_seconds"])
        for confidence in sorted(set(float(value) for value in grid), reverse=True)
    ]
    selection = select_operating_point(validation_rows, baseline_confidence=0.25)
    selected_confidence = float(selection["selected"]["confidence_threshold"])
    test_predictions, test_latency = predict_aabb_records(model, test, image_size, selected_confidence, device)
    test_metrics = score_predictions(test, test_predictions, selected_confidence, test_latency["total_seconds"])
    report = {
        "model_kind": model_kind,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "image_size": image_size,
        "device": device,
        "match_iou_threshold": 0.5,
        "validation_sweep": validation_rows,
        "selection": selection,
        "selected_confidence": selected_confidence,
        "locked_test": test_metrics,
        "test_locked": True,
        "warnings": [
            "AABB comparison only; no OBB metric or oriented-crop claim is made.",
            "WEEE has geometry labels only; no anomaly/defect metric is reported.",
        ],
    }
    return report


def _run_model_subprocess(args: argparse.Namespace, model_kind: str, output: Path) -> dict[str, Any]:
    weight = args.yolo_weights if model_kind == "yolo" else args.rtdetr_weights
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-summary", str(args.data_summary),
        "--yolo-weights", str(args.yolo_weights),
        "--rtdetr-weights", str(args.rtdetr_weights),
        "--weee-root", str(args.weee_root),
        "--validation-annotations", str(args.validation_annotations),
        "--test-annotations", str(args.test_annotations),
        "--output", str(output),
        "--imgsz", str(args.imgsz),
        "--model-kind", model_kind,
        "--weights", str(weight),
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{model_kind} evaluation failed:\n{completed.stderr[-4000:]}")
    return json.loads(output.read_text(encoding="utf-8"))["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-summary", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--rtdetr-weights", type=Path, required=True)
    parser.add_argument("--weee-root", type=Path, default=Path("datasets/external/weee_screw"))
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device")
    parser.add_argument("--model-kind", choices=("yolo", "rtdetr"))
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.data_summary.read_text(encoding="utf-8"))
    if args.model_kind:
        if args.weights is None:
            parser.error("--weights is required with --model-kind")
        result = evaluate_detector(args.model_kind, args.weights, args.validation_annotations, args.test_annotations, args.weee_root, args.imgsz, args.device)
        payload = {"schema_version": "1.0", "experiment": "weee_yolo_rtdetr_aabb_comparison", "model": result}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    report = {
        "schema_version": "1.0",
        "experiment": "weee_yolo_rtdetr_aabb_comparison",
        "protocol": {
            "data_summary": str(args.data_summary.resolve()),
            "data_summary_sha256": sha256_file(args.data_summary),
            "same_image_splits": True,
            "same_image_size": args.imgsz,
            "same_confidence_grid": list(GRID),
            "same_iou_rule": 0.5,
            "test_locked": True,
            "geometry": "AABB",
        },
        "dataset": summary,
        "models": {},
    }
    with tempfile.TemporaryDirectory(prefix="weee_aabb_eval_") as temporary:
        temporary_root = Path(temporary)
        report["models"]["yolo"] = _run_model_subprocess(args, "yolo", temporary_root / "yolo.json")
        report["models"]["rtdetr"] = _run_model_subprocess(args, "rtdetr", temporary_root / "rtdetr.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
