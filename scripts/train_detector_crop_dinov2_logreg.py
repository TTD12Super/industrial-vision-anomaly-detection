#!/usr/bin/env python3
"""Train a frozen-DINOv2 GOOD/DEFECT head on detector-generated crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from mvtec_system.classification.classifier import LinearClassifierConfig, fit_linear_classifier, predict_linear_classifier, save_linear_classifier
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(report_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No detector-crop objects in {report_path}")
    selected = [row for row in rows if row.get("product_recognition_status") == "KNOWN_PRODUCT"]
    if not selected or {row.get("quality_label") for row in selected} != {"GOOD", "DEFECT"}:
        raise ValueError(f"Detector-crop report must contain GOOD and DEFECT KNOWN_PRODUCT rows: {report_path}")
    for row in selected:
        crop_path = Path(str(row.get("crop_path", "")))
        if not crop_path.is_file():
            raise FileNotFoundError(crop_path)
    return selected


def _embedding(crop_path: Path, encoder: SharedDINOv2) -> torch.Tensor:
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read crop: {crop_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = encoder.config.image_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(encoder.config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(encoder.config.std, dtype=tensor.dtype).view(3, 1, 1)
    return encoder.forward(((tensor - mean) / std).unsqueeze(0)).global_embedding[0].detach().cpu()


def _embeddings(rows: list[dict[str, Any]], encoder: SharedDINOv2) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([_embedding(Path(row["crop_path"]), encoder) for row in rows]).float()
    labels = torch.tensor([1 if row["quality_label"] == "DEFECT" else 0 for row in rows], dtype=torch.long)
    if features.ndim != 2 or features.shape[0] != len(rows) or not torch.isfinite(features).all().item():
        raise ValueError("Detector-crop embeddings are malformed")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both GOOD and DEFECT detector crops are required")
    return features, labels


def _scores(classifier: dict[str, Any], features: torch.Tensor) -> np.ndarray:
    _, probabilities = predict_linear_classifier(classifier, features)
    return probabilities[:, 1].numpy()


def _choose_threshold(labels: np.ndarray, scores: np.ndarray, target_recall: float, max_good_fp: float) -> tuple[float, str]:
    candidates = sorted(set([0.0, 1.0, *scores.tolist()]))
    eligible = []
    for threshold in candidates:
        predicted = scores >= threshold
        defects = labels == 1
        good = labels == 0
        recall = float((predicted & defects).sum() / defects.sum())
        good_fp = float((predicted & good).sum() / good.sum())
        if recall >= target_recall and good_fp <= max_good_fp:
            eligible.append((recall, -good_fp, threshold))
    if eligible:
        return float(max(eligible)[2]), "highest threshold meeting validation recall/FP gate"
    best = max(
        candidates,
        key=lambda threshold: (
            float(recall_score(labels, scores >= threshold, zero_division=0)),
            -float((scores[labels == 0] >= threshold).mean()),
            float(threshold),
        ),
    )
    return float(best), "no validation threshold met the requested gate"


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "count": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float((prediction == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "defect_recall": float(recall_score(labels, prediction, zero_division=0)),
        "good_fp": float(fp / (tn + fp) if tn + fp else 0.0),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix_good_defect": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_report = args.train_report.resolve()
    validation_report = args.validation_report.resolve()
    train_rows = _load_rows(train_report)
    validation_rows = _load_rows(validation_report)

    encoder = SharedDINOv2(
        DINOv2Config(image_size=(256, 256)),
        artifact_root=args.encoder_artifact_root,
        device=args.device,
    )
    train_features, train_labels = _embeddings(train_rows, encoder)
    validation_features, validation_labels = _embeddings(validation_rows, encoder)

    # Balance the small DEFECT train class without changing the frozen encoder.
    defect_indices = torch.where(train_labels == 1)[0]
    good_indices = torch.where(train_labels == 0)[0]
    repeats = int(np.ceil(len(good_indices) / len(defect_indices)))
    balanced_indices = torch.cat((good_indices, defect_indices.repeat(repeats)[: len(good_indices)]))
    train_artifact = {"embeddings": train_features[balanced_indices], "labels": train_labels[balanced_indices]}
    classifier_config = LinearClassifierConfig(C=args.C, max_iter=args.max_iter, random_state=args.seed)
    classifier = fit_linear_classifier(train_artifact, {"GOOD": 0, "DEFECT": 1}, classifier_config)
    validation_scores = _scores(classifier, validation_features)
    validation_labels_np = validation_labels.numpy()
    threshold, threshold_rule = _choose_threshold(
        validation_labels_np, validation_scores, args.target_defect_recall, args.max_good_fp
    )

    output_classifier_dir = args.artifact_root.resolve() / "classifier"
    output_classifier_dir.mkdir(parents=True, exist_ok=True)
    save_linear_classifier(
        output_classifier_dir,
        classifier,
        {"GOOD": 0, "DEFECT": 1},
        classifier_config,
        {"artifact_version": "detector-crop-v1", "encoder": encoder.metadata(), "train_report_sha256": _sha256(train_report)},
    )
    policy = {
        "schema_version": "1.0",
        "defect_threshold": float(threshold),
        "review_low_threshold": float(max(0.0, threshold - 0.10)),
        "threshold_rule": threshold_rule,
        "target_defect_recall": args.target_defect_recall,
        "max_good_fp": args.max_good_fp,
        "calibration_report": str(validation_report),
        "calibration_report_sha256": _sha256(validation_report),
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "policy_calibration.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    results = {
        "train": {"GOOD": int((train_labels == 0).sum()), "DEFECT": int((train_labels == 1).sum())},
        "validation": {"GOOD": int((validation_labels == 0).sum()), "DEFECT": int((validation_labels == 1).sum())},
        "validation_metrics": _metrics(validation_labels_np, validation_scores, threshold),
    }
    summary = {
        "protocol": "detector-generated OBB crop -> frozen DINOv2 global embedding -> balanced logistic GOOD/DEFECT head",
        "device": str(encoder.device),
        "train_report": str(train_report),
        "validation_report": str(validation_report),
        "balanced_train_count": int(len(train_artifact["labels"])),
        "classifier_artifact": str((output_classifier_dir / "classifier.pt").resolve()),
        "policy_calibration": policy,
        "results": results,
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, default=Path("reports/experiments/screw_detector_crop_train_rnd1.json"))
    parser.add_argument("--validation-report", type=Path, default=Path("reports/experiments/screw_detector_crop_calibration_rnd1.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/products/screw/screw-detector-crop-logreg-rnd1"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/experiments/screw_detector_crop_logreg_rnd1"))
    parser.add_argument("--encoder-artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--max-good-fp", type=float, default=0.20)
    args = parser.parse_args()
    if args.C <= 0 or args.max_iter < 1 or not 0 < args.target_defect_recall <= 1 or not 0 <= args.max_good_fp <= 1:
        parser.error("invalid classifier/calibration settings")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
