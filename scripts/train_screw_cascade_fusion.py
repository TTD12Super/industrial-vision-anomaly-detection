#!/usr/bin/env python3
"""Train and benchmark a recall-first DINOv2 + AnomalyCLIP score-fusion cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspection.anomalyclip import AnomalyClipSession
from inspection.cascade_fusion import FEATURE_NAMES, CascadeScoreFusion, heatmap_statistics
from inspection.config import load_product_config
from inspection.geometry import oriented_crop
from inspection.dataset import load_dataset_records
from mvtec_system.classification.classifier import predict_linear_classifier


def load_feature_rows(config_path: Path, embedding_root: Path, cache_root: Path, force: bool) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor]]:
    config = load_product_config(config_path)
    images, objects = load_dataset_records(config)
    image_by_id = {row["image_id"]: row for row in images}
    object_by_id = {row["object_id"]: row for row in objects}
    session_settings = dict(config.model("localizer"))
    session_settings["relation_fusion_enabled"] = False
    session_settings["supervised_refiner_enabled"] = False
    for key in ("runner", "repository", "checkpoint", "clip_cache"):
        if session_settings.get(key):
            session_settings[key] = str(config.project_path(session_settings[key]))
    session = AnomalyClipSession(session_settings)
    rows: dict[str, dict[str, Any]] = {}
    embeddings: dict[str, torch.Tensor] = {}
    classifier_path = config.project_path(config.model("classifier")["artifact_dir"]) / "classifier" / "classifier.pt"
    classifier = torch.load(classifier_path, map_location="cpu", weights_only=True)
    for split in ("train", "validation", "test"):
        payload = torch.load(embedding_root / f"{split}_embeddings.pt", map_location="cpu", weights_only=True)
        embeddings[split] = payload["embeddings"].float()
        cache_path = cache_root / f"{split}_anomaly_features.npz"
        object_ids = [str(value) for value in payload["object_ids"]]
        if cache_path.is_file() and not force:
            cached = np.load(cache_path)
            if cached["object_ids"].tolist() == object_ids:
                for object_id, score, stats in zip(object_ids, cached["anomalyclip_scores"], cached["heatmap_stats"]):
                    rows[object_id] = {"split": split, "anomalyclip_score": float(score), "heatmap_stats": np.asarray(stats, dtype=np.float32)}
                continue
        scores: list[float] = []
        stats: list[np.ndarray] = []
        for object_id in object_ids:
            obj = object_by_id[object_id]
            image = image_by_id[obj["image_id"]]
            source = cv2.imread(str(config.dataset_path(Path(image["relative_path"]))))
            if source is None:
                raise ValueError(f"Cannot read image {image['relative_path']}")
            crop = oriented_crop(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), obj["obb_xyxyxyxy"]).image
            heatmap, anomaly_score, _, _ = session.infer(crop)
            scores.append(float(anomaly_score))
            stats.append(heatmap_statistics(heatmap))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, object_ids=np.asarray(object_ids), anomalyclip_scores=np.asarray(scores, dtype=np.float32), heatmap_stats=np.stack(stats))
        for object_id, score, row_stats in zip(object_ids, scores, stats):
            rows[object_id] = {"split": split, "anomalyclip_score": float(score), "heatmap_stats": np.asarray(row_stats, dtype=np.float32)}
    for split in ("train", "validation", "test"):
        payload = torch.load(embedding_root / f"{split}_embeddings.pt", map_location="cpu", weights_only=True)
        _, probabilities = predict_linear_classifier(classifier, payload["embeddings"].float())
        for object_id, probability, label in zip(payload["object_ids"], probabilities[:, 1].tolist(), payload["labels"].tolist()):
            rows[str(object_id)]["p_global"] = float(probability)
            rows[str(object_id)]["label"] = int(label)
    return rows, embeddings


def split_arrays(rows: dict[str, dict[str, Any]], embeddings: dict[str, torch.Tensor], embedding_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = torch.load(embedding_root / f"{split}_embeddings.pt", map_location="cpu", weights_only=True)
    object_ids = [str(value) for value in payload["object_ids"]]
    values = [rows[object_id] for object_id in object_ids]
    return (
        embeddings[split].numpy(),
        np.asarray([value["label"] for value in values], dtype=np.int64),
        np.asarray([value["p_global"] for value in values], dtype=np.float32),
        np.asarray([value["anomalyclip_score"] for value in values], dtype=np.float32),
        np.stack([value["heatmap_stats"] for value in values]),
    )


def select_candidate_threshold(labels: np.ndarray, p_global: np.ndarray, anomaly_score: np.ndarray, target_recall: float) -> tuple[float, dict[str, float]]:
    signal = CascadeScoreFusion.candidate_signal(p_global, anomaly_score)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, signal])
    eligible = []
    for threshold in candidates:
        prediction = signal >= threshold
        recall = recall_score(labels, prediction, zero_division=0)
        fp = float(prediction[labels == 0].mean())
        if recall >= target_recall:
            eligible.append((fp, -recall, -float(threshold), float(threshold), recall))
    selected = min(eligible) if eligible else min((float((signal >= t)[labels == 0].mean()), 0.0, -float(t), float(t), float(recall_score(labels, signal >= t, zero_division=0))) for t in candidates)
    threshold = selected[3]
    return threshold, {"recall": float(selected[4]), "good_false_positive_rate": float(selected[0])}


def metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "images": int(len(labels)), "threshold": float(threshold), "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "precision_defect": float(precision_score(labels, predictions, zero_division=0)),
        "recall_defect": float(recall_score(labels, predictions, zero_division=0)), "f1_defect": float(f1_score(labels, predictions, zero_division=0)),
        "good_recall": float(tn / (tn + fp) if tn + fp else 0.0), "good_false_positive_rate": float(fp / (tn + fp) if tn + fp else 0.0),
        "image_auroc": float(roc_auc_score(labels, scores)), "image_average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix_good_defect": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    validation = report["validation"]
    test = report["test"]
    target_recall = float(report["target_defect_recall"])
    max_good_fp = float(report["max_good_fp"])
    lines = [
        "# Screw cascade fusion benchmark",
        "",
        "Independent experiment branch: `codex/screw-cascade-fusion`.",
        "",
        "Protocol: recall-first candidate union (`max(DINOv2 probability, AnomalyCLIP score)`) followed by a supervised logistic verifier using DINOv2 distances, AnomalyCLIP score, heatmap statistics, and a hard-negative weight for GOOD candidates.",
        "",
        "| Split | Accuracy | DEFECT recall | GOOD false-positive rate | Image AUROC | Image AP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Validation | {validation['accuracy']:.3f} | {validation['recall_defect']:.3f} | {validation['good_false_positive_rate']:.3f} | {validation['image_auroc']:.3f} | {validation['image_average_precision']:.3f} |",
        f"| Test | {test['accuracy']:.3f} | {test['recall_defect']:.3f} | {test['good_false_positive_rate']:.3f} | {test['image_auroc']:.3f} | {test['image_average_precision']:.3f} |",
        "",
        f"Target on test: DEFECT recall >= {target_recall:.2f}; GOOD false-positive rate <= {max_good_fp:.2f}.",
        f"Result: {'PASS' if test['recall_defect'] >= target_recall and test['good_false_positive_rate'] <= max_good_fp else 'NOT MET'}.",
        "",
        "The validation result is not used as the final claim: it is perfect here, but the test result shows a domain/generalization gap. Production configuration was not changed.",
        "",
        f"Artifacts: `{report['artifact']}` and cached AnomalyCLIP features under `artifacts/products/screw/screw-full-dev-002/cascade_cache/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_final_threshold(labels: np.ndarray, scores: np.ndarray, candidate: np.ndarray, max_good_fp: float, target_recall: float) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, scores])
    measured = []
    for threshold in candidates:
        prediction = candidate & (scores >= threshold)
        report = metrics(labels, prediction, scores, float(threshold))
        measured.append((report, float(threshold)))
    eligible = [item for item in measured if item[0]["good_false_positive_rate"] <= max_good_fp]
    target = [item for item in eligible if item[0]["recall_defect"] >= target_recall]
    if target:
        selected = max(target, key=lambda item: (item[0]["recall_defect"], item[0]["precision_defect"], -item[1]))
        reason = "target recall and GOOD-FP constraint met"
    elif eligible:
        selected = max(eligible, key=lambda item: (item[0]["recall_defect"], item[0]["precision_defect"], -item[1]))
        reason = "best recall under GOOD-FP constraint; target recall unattainable"
    else:
        selected = min(measured, key=lambda item: item[0]["good_false_positive_rate"])
        reason = "GOOD-FP constraint unattainable"
    return selected[1], {"threshold_rule": reason, **selected[0]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_product_config(args.config)
    rows, embeddings = load_feature_rows(args.config, args.embedding_root, args.cache_root, args.force)
    train = split_arrays(rows, embeddings, args.embedding_root, "train")
    validation = split_arrays(rows, embeddings, args.embedding_root, "validation")
    test = split_arrays(rows, embeddings, args.embedding_root, "test")
    candidate_threshold, candidate_validation = select_candidate_threshold(validation[1], validation[2], validation[3], args.target_candidate_recall)
    final_placeholder = 0.5
    fusion = CascadeScoreFusion.fit(train[0], train[1], train[2], train[3], train[4], candidate_threshold, final_placeholder)
    validation_scores = fusion.predict_score(validation[0], validation[2], validation[3], validation[4])
    test_scores = fusion.predict_score(test[0], test[2], test[3], test[4])
    validation_candidate = fusion.candidate_signal(validation[2], validation[3]) >= candidate_threshold
    test_candidate = fusion.candidate_signal(test[2], test[3]) >= candidate_threshold
    final_threshold, validation_report = select_final_threshold(validation[1], validation_scores, validation_candidate, args.max_good_fp, args.target_defect_recall)
    fusion.final_threshold = final_threshold
    validation_predictions = fusion.predict(validation[0], validation[2], validation[3], validation[4])
    test_predictions = fusion.predict(test[0], test[2], test[3], test[4])
    artifact_path = args.artifact_root / "cascade_fusion.pkl"
    fusion.save(artifact_path)
    report = {
        "protocol": "recall-first candidate union + DINOv2/AnomalyCLIP scalar feature verifier + hard-negative weighting",
        "dataset": str(config.dataset_root), "candidate_threshold": candidate_threshold, "final_threshold": final_threshold,
        "target_candidate_recall": args.target_candidate_recall, "target_defect_recall": args.target_defect_recall, "max_good_fp": args.max_good_fp,
        "candidate_validation": candidate_validation, "candidate_test": {"recall": float(recall_score(test[1], test_candidate, zero_division=0)), "good_false_positive_rate": float(test_candidate[test[1] == 0].mean())},
        "validation": {**metrics(validation[1], validation_predictions, validation_scores, final_threshold), "threshold_rule": validation_report["threshold_rule"]},
        "test": metrics(test[1], test_predictions, test_scores, final_threshold), "feature_names": list(FEATURE_NAMES),
        "artifact": str(artifact_path.resolve()), "warning": "Supervised-development split; MVTec official test images were used during development.",
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_report(report, args.report_root / "benchmark.md")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/screw_full_anomaly.yaml"))
    parser.add_argument("--embedding-root", type=Path, default=Path("artifacts/products/screw/screw-full-dev-001/classifier"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/products/screw/screw-full-dev-002/cascade_cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/products/screw/screw-full-dev-002/cascade"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/products/screw/screw-full-dev-002/cascade"))
    parser.add_argument("--target-candidate-recall", type=float, default=0.95)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--max-good-fp", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_candidate_recall <= 1.0 or not 0.0 < args.target_defect_recall <= 1.0 or not 0.0 < args.max_good_fp < 1.0:
        parser.error("recall targets must be in (0, 1] and max-good-fp in (0, 1)")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
