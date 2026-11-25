#!/usr/bin/env python3
"""Benchmark GOOD/DEFECT scores on detector-generated validation crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspection.backends import StaticLocalizer
from inspection.config import load_product_config
from inspection.dataset import load_dataset_records
from inspection.evaluation import match_objects, polygon_iou
from inspection.pipeline import InspectionPipeline
from inspection.runtime import (
    _detector_settings,
    _path_settings,
    build_classifier,
    build_detector,
    build_product_recognizer,
    load_policy,
)
from inspection.interfaces import KNOWN_PRODUCT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _operating_point(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = np.asarray([1 if row["quality_label"] == "DEFECT" else 0 for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["p_defect"]) for row in rows], dtype=np.float64)
    predicted = scores >= threshold
    defects = labels == 1
    good = labels == 0
    return {
        "threshold": float(threshold),
        "recall": float((predicted & defects).sum() / defects.sum()),
        "good_fp": float((predicted & good).sum() / good.sum()),
        "defect_count": int((predicted & defects).sum()),
        "good_flagged_count": int((predicted & good).sum()),
    }


def _sweep(rows: list[dict[str, Any]], target_recall: float, max_good_fp: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("No matched KNOWN_PRODUCT detector crops were scored")
    labels = np.asarray([1 if row["quality_label"] == "DEFECT" else 0 for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["p_defect"]) for row in rows], dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Detector-crop validation must contain GOOD and DEFECT labels")
    points: list[dict[str, Any]] = []
    for threshold in sorted(set([0.0, 1.0, *scores.tolist()])):
        points.append(_operating_point(rows, float(threshold)))
    eligible = [
        point
        for point in points
        if point["recall"] >= target_recall and point["good_fp"] <= max_good_fp
    ]
    selected = max(
        eligible,
        key=lambda point: (point["recall"], -point["good_fp"], point["threshold"]),
    ) if eligible else None
    return {
        "target_defect_recall": target_recall,
        "max_good_fp": max_good_fp,
        "status": "selected" if selected else "no_validation_gate",
        "selected": selected,
        "candidate_count": len(eligible),
        "points": points,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_product_config(args.config)
    images, objects = load_dataset_records(config)
    objects_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in objects:
        objects_by_image.setdefault(row["image_id"], []).append(row)

    detector = build_detector(_detector_settings(config))
    classifier_settings = dict(config.model("classifier"))
    if args.classifier_artifact_dir:
        classifier_settings.update(
            {
                "backend": "dinov2_logreg",
                "artifact_dir": str(args.classifier_artifact_dir),
                "model_version": "detector-crop-dinov2-logreg",
            }
        )
    classifier = build_classifier(_path_settings(config, classifier_settings))
    recognizer_settings = config.models.get("product_recognizer")
    recognizer = (
        build_product_recognizer(_path_settings(config, recognizer_settings))
        if recognizer_settings else None
    )
    pipeline = InspectionPipeline(
        product_key=config.product_key,
        model_bundle_version="detector-crop-calibration",
        detector=detector,
        classifier=classifier,
        policy=load_policy(config),
        localizer=StaticLocalizer(),
        runs_root=args.runs_root,
        reporter=None,
        product_recognizer=recognizer,
    )

    totals = {
        "ground_truth_objects": 0,
        "predicted_objects": 0,
        "matched_objects": 0,
        "missed_ground_truth_objects": 0,
        "unmatched_predictions": 0,
        "known_product_matches": 0,
        "unknown_product_matches": 0,
        "review_product_matches": 0,
    }
    scored: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for index, image in enumerate(row for row in images if row["split"] == args.split):
        result = pipeline.inspect_path(
            config.dataset_path(Path(image["relative_path"])),
            f"{args.split}_{index:04d}",
        )
        predictions = result["objects"]
        truths = objects_by_image[image["image_id"]]
        matches = match_objects(truths, predictions, args.iou_threshold)
        totals["ground_truth_objects"] += len(truths)
        totals["predicted_objects"] += len(predictions)
        totals["matched_objects"] += len(matches)
        totals["missed_ground_truth_objects"] += len(truths) - len(matches)
        totals["unmatched_predictions"] += len(predictions) - len(matches)
        for truth_index, prediction_index in matches.items():
            truth = truths[truth_index]
            prediction = predictions[prediction_index]
            status = prediction.get("product_recognition_status")
            key = {
                "KNOWN_PRODUCT": "known_product_matches",
                "UNKNOWN_PRODUCT": "unknown_product_matches",
                "REVIEW_PRODUCT": "review_product_matches",
            }.get(status)
            if key:
                totals[key] += 1
            row = {
                "image_id": image["image_id"],
                "object_id": truth["object_id"],
                "quality_label": truth["quality_label"],
                "prediction_object_id": prediction["object_id"],
                "crop_path": str(
                    (Path(args.runs_root).resolve() / f"{args.split}_{index:04d}" / "objects"
                     / prediction["object_id"] / "crop_original.png").resolve()
                ),
                "obb_iou": polygon_iou(truth["obb_xyxyxyxy"], prediction["obb_xyxyxyxy"]),
                "detection_score": prediction["detection_score"],
                "product_recognition_status": status,
                "recognition_score": prediction.get("recognition_score"),
                "p_defect": prediction.get("p_defect"),
            }
            if status == KNOWN_PRODUCT and isinstance(row["p_defect"], (int, float)):
                scored.append(row)
        per_image.append(
            {
                "image_id": image["image_id"],
                "matched_objects": len(matches),
                "ground_truth_objects": len(truths),
                "predicted_objects": len(predictions),
                "missed_ground_truth_objects": len(truths) - len(matches),
                "unmatched_predictions": len(predictions) - len(matches),
            }
        )

    if args.split == "test" and args.threshold is None:
        raise ValueError("--threshold is required for test; select it on validation only")
    selection = (
        {
            "target_defect_recall": args.target_defect_recall,
            "max_good_fp": args.max_good_fp,
            "status": "fixed_threshold",
            "selected": _operating_point(scored, args.threshold),
            "candidate_count": 0,
            "points": [],
        }
        if args.threshold is not None
        else _sweep(scored, args.target_defect_recall, args.max_good_fp)
    )
    output = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "protocol": "real detector OBB -> deterministic crop -> calibrated product routing -> classifier score; localizer and VLM disabled",
        "totals": totals,
        "scored_known_product_objects": len(scored),
        "selection": selection,
        "objects": scored,
        "per_image": per_image,
        "provenance": {
            "config": str(config.source_path),
            "config_sha256": _sha256(config.source_path),
            "image_manifest": str(config.dataset_path(config.image_manifest)),
            "image_manifest_sha256": _sha256(config.dataset_path(config.image_manifest)),
            "object_manifest": str(config.dataset_path(config.object_manifest)),
            "object_manifest_sha256": _sha256(config.dataset_path(config.object_manifest)),
            "detector_weights": str(config.project_path(config.model("detector")["weights"])),
            "classifier_artifact": str(
                Path(args.classifier_artifact_dir).resolve()
                if args.classifier_artifact_dir
                else config.project_path(config.model("classifier")["artifact_path"])
            ),
            "product_gallery": str(config.project_path(config.model("product_recognizer")["gallery_path"])),
            "product_calibration": str(config.project_path(config.model("product_recognizer")["calibration_path"])),
        },
        "warning": (
            "Fixed threshold was selected on validation and evaluated here; do not tune it on this test report."
            if args.split == "test"
            else "Validation-only calibration evidence; do not apply the selected threshold to a locked test or production bundle without an independent holdout."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/screw_patchcore_padim.yaml"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/screw_detector_crop_calibration_rnd1"))
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/screw_detector_crop_calibration_rnd1.json"))
    parser.add_argument("--classifier-artifact-dir", type=Path, default=None, help="Optional DINOv2 logreg artifact directory override")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--max-good-fp", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=None, help="Locked threshold from validation; required for test")
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in (0, 1]")
    if not 0.0 <= args.max_good_fp <= 1.0:
        raise ValueError("--max-good-fp must be in [0, 1]")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
