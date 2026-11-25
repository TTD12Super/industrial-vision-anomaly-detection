"""Validation-only binary/review policy calibration from saved DINOv2 embeddings."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import recall_score
import torch

from mvtec_system.classification.classifier import predict_linear_classifier

from .config import ProductConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold(labels: np.ndarray, probabilities: np.ndarray, target_recall: float) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    eligible = [float(value) for value in candidates if recall_score(labels, probabilities >= value, zero_division=0) >= target_recall]
    if not eligible:
        raise ValueError(f"Target validation DEFECT recall {target_recall} is unattainable")
    return max(eligible)


def calibrate_saved_dinov2(config: ProductConfig) -> dict[str, Any]:
    """Reuse frozen validation embeddings so policy calibration cannot touch test data."""

    classifier = config.model("classifier")
    source = classifier.get("calibration_source")
    if not isinstance(source, dict):
        raise ValueError("classifier.calibration_source is required")
    embeddings_path = config.project_path(source["validation_embeddings"])
    manifest_value = source.get("split_manifest")
    manifest_path = config.project_path(manifest_value) if manifest_value else None
    artifact_dir = config.project_path(classifier["artifact_dir"])
    cached = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    embeddings = cached.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("Validation embedding cache is malformed")
    if isinstance(cached.get("labels"), torch.Tensor):
        labels = cached["labels"].numpy().astype(np.int64)
        if len(labels) != len(embeddings):
            raise ValueError("Validation labels do not match cached embeddings")
    else:
        paths = cached.get("paths")
        if not isinstance(paths, list) or len(paths) != len(embeddings):
            raise ValueError("Validation embedding cache is malformed")
        if manifest_path is None:
            raise ValueError("calibration_source.split_manifest is required for embedding caches without labels")
        labels_by_path = {
            row["image_path"]: row["label"]
            for row in csv.DictReader(manifest_path.open(encoding="utf-8"))
            if row["split"] == "validation"
        }
        if set(paths) != set(labels_by_path):
            raise ValueError("Validation embedding cache and split manifest do not describe identical images")
        labels = np.asarray([1 if labels_by_path[path] == "DEFECT" else 0 for path in paths], dtype=np.int64)
    if set(labels) != {0, 1}:
        raise ValueError("Validation data must contain GOOD and DEFECT labels")
    classifier_data = torch.load(artifact_dir / "classifier" / "classifier.pt", map_location="cpu", weights_only=True)
    _, probabilities = predict_linear_classifier(classifier_data, embeddings)
    p_defect = probabilities[:, 1].numpy()
    target_recall = float(config.policy.get("target_defect_recall", 0.90))
    defect_threshold = _threshold(labels, p_defect, target_recall)
    missed_defect_scores = p_defect[(labels == 1) & (p_defect < defect_threshold)]
    review_low_threshold = float(missed_defect_scores.min()) if len(missed_defect_scores) else max(0.0, defect_threshold - 1e-6)
    reviewed = (p_defect >= review_low_threshold) & (p_defect < defect_threshold)
    output = {
        "schema_version": "1.0",
        "source_split": "validation",
        "defect_threshold": float(defect_threshold),
        "review_low_threshold": review_low_threshold,
        "target_defect_recall": target_recall,
        "validation_images": int(len(labels)),
        "validation_defect_recall": float(recall_score(labels, p_defect >= defect_threshold, zero_division=0)),
        "validation_defects_captured_by_review": int(((labels == 1) & reviewed).sum()),
        "validation_good_sent_to_review": int(((labels == 0) & reviewed).sum()),
        "validation_false_negatives_outside_review": int(((labels == 1) & (p_defect < review_low_threshold)).sum()),
        "provenance": {
            "classifier_sha256": _sha256(artifact_dir / "classifier" / "classifier.pt"),
            "validation_embeddings_sha256": _sha256(embeddings_path),
            "split_manifest_sha256": _sha256(manifest_path) if manifest_path is not None else None,
        },
    }
    destination = config.project_path(config.policy["calibration_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output
