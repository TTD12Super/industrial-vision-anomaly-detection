"""Validation metrics for the product-category classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from .classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
)


def evaluate_product_classifier(
    classifier: dict[str, Any],
    artifact: dict[str, Any],
    class_mapping: dict[str, int],
) -> dict[str, Any]:
    """Calculate classification metrics for one held-out embedding split."""

    labels = artifact.get("labels")
    embeddings = artifact.get("embeddings")
    if not isinstance(labels, torch.Tensor) or not isinstance(embeddings, torch.Tensor):
        raise ValueError("Evaluation artifact must contain tensor embeddings and labels.")
    if labels.ndim != 1 or embeddings.ndim != 2 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Evaluation embedding and label shapes do not match.")
    logits, probabilities = predict_linear_classifier(classifier, embeddings)
    predictions = logits.argmax(dim=1)
    class_ids = sorted(class_mapping.values())
    class_names = [
        category for category, index in sorted(class_mapping.items(), key=lambda item: item[1])
    ]
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    precision = precision_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    top_indices = probabilities.topk(min(3, probabilities.shape[1]), dim=1).indices
    top_k_accuracy = top_indices.eq(labels.cpu().unsqueeze(1)).any(dim=1).float().mean().item()

    per_class = {}
    for index, category in zip(range(len(class_names)), class_names):
        per_class[category] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int((y_true == class_ids[index]).sum()),
        }
    return {
        "split": artifact.get("split", "unknown"),
        "sample_count": int(labels.shape[0]),
        "class_mapping": class_mapping,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "top_3_accuracy": float(top_k_accuracy),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        "per_class": per_class,
    }


def save_classification_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write metrics as a readable JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def select_linear_classifier(
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    configs: list[LinearClassifierConfig],
) -> tuple[dict[str, Any], LinearClassifierConfig, dict[str, Any], list[dict[str, Any]]]:
    """Select a classifier using validation macro-F1, never the official test set."""

    if not configs:
        raise ValueError("At least one classifier configuration is required.")
    candidates = []
    best = None
    for config in configs:
        classifier = fit_linear_classifier(train_artifact, class_mapping, config)
        report = evaluate_product_classifier(classifier, validation_artifact, class_mapping)
        metrics = report["metrics"]
        score = (
            metrics["macro_f1"],
            metrics["accuracy"],
            metrics["top_3_accuracy"],
        )
        candidate = {"config": config.__dict__, "metrics": metrics}
        candidates.append(candidate)
        if best is None or score > best["score"]:
            best = {
                "classifier": classifier,
                "config": config,
                "report": report,
                "score": score,
            }
    best["report"]["model_selection"] = {
        "selection_metric": "macro_f1",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_config": best["config"].__dict__,
    }
    return best["classifier"], best["config"], best["report"], candidates
