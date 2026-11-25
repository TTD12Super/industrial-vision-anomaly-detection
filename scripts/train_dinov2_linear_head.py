#!/usr/bin/env python3
"""Train a two-class neural head on frozen DINOv2 screw embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


GOOD = 0
DEFECT = 1


class BinaryHead(nn.Module):
    """Small trainable GOOD/DEFECT head attached to DINOv2 global features."""

    def __init__(self, feature_dimension: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Dropout(dropout),
            nn.Linear(feature_dimension, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {"train": {}, "validation": {}, "test": {}}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in labels:
                raise ValueError(f"Unexpected split in manifest: {split}")
            label = GOOD if row["label"] == "GOOD" else DEFECT if row["label"] == "DEFECT" else -1
            if label < 0:
                raise ValueError(f"Unexpected label in manifest: {row['label']}")
            labels[split][row["image_path"]] = label
    return labels


def load_split(path: Path, label_map: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    embeddings = payload["embeddings"].float()
    paths = [str(path) for path in payload["paths"]]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(paths):
        raise ValueError(f"Invalid embedding artifact: {path}")
    if any(image_path not in label_map for image_path in paths):
        raise ValueError(f"Embedding paths do not match manifest: {path}")
    labels = torch.tensor([label_map[image_path] for image_path in paths], dtype=torch.long)
    if set(labels.tolist()) != {GOOD, DEFECT}:
        raise ValueError(f"Split must contain GOOD and DEFECT: {path}")
    return embeddings, labels


def probabilities(model: nn.Module, embeddings: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(embeddings), dim=1)[:, DEFECT].detach().cpu().numpy()


def choose_threshold(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> tuple[float, str]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    eligible = [float(value) for value in candidates if recall_score(labels, scores >= value, zero_division=0) >= target_recall]
    if eligible:
        return max(eligible), f"highest threshold meeting defect recall >= {target_recall:.2f}"
    ranked = []
    for value in candidates:
        prediction = scores >= value
        ranked.append((balanced_accuracy_score(labels, prediction), recall_score(labels, prediction, zero_division=0), -float(value), float(value)))
    return max(ranked)[3], "best validation balanced accuracy; target recall was unattainable"


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    prediction = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[GOOD, DEFECT]).ravel()
    return {
        "images": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision_defect": float(precision_score(labels, prediction, zero_division=0)),
        "recall_defect": float(recall_score(labels, prediction, zero_division=0)),
        "f1_defect": float(f1_score(labels, prediction, zero_division=0)),
        "good_recall": float(tn / (tn + fp) if tn + fp else 0.0),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix_good_defect": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    embedding_root = args.embedding_root.resolve()
    manifest_path = args.manifest.resolve()
    artifact_root = args.artifact_root.resolve()
    report_root = args.report_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    train_x, train_y = load_split(embedding_root / "train.pt", manifest["train"])
    validation_x, validation_y = load_split(embedding_root / "validation.pt", manifest["validation"])
    test_x, test_y = load_split(embedding_root / "test.pt", manifest["test"])
    feature_dimension = int(train_x.shape[1])
    device = torch.device(args.device)
    model = BinaryHead(feature_dimension, args.dropout).to(device)
    train_x, train_y = train_x.to(device), train_y.to(device)
    validation_x, validation_y = validation_x.to(device), validation_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    class_counts = torch.bincount(train_y, minlength=2).float()
    class_weights = (class_counts.sum() / (2.0 * class_counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        validation_scores = probabilities(model, validation_x)
        validation_labels = validation_y.detach().cpu().numpy()
        validation_auroc = roc_auc_score(validation_labels, validation_scores)
        validation_balanced = balanced_accuracy_score(validation_labels, validation_scores >= 0.5)
        history.append({"epoch": epoch, "loss": float(loss.item()), "validation_auroc": float(validation_auroc), "validation_balanced_accuracy_at_0_5": float(validation_balanced)})
        score = (float(validation_auroc), float(validation_balanced))
        best_tuple = (-float("inf"), -float("inf")) if best_state is None else best_score
        if best_state is None or score > best_tuple:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state)
    validation_labels = validation_y.detach().cpu().numpy()
    test_labels = test_y.detach().cpu().numpy()
    validation_scores = probabilities(model, validation_x)
    test_scores = probabilities(model, test_x)
    threshold, threshold_rule = choose_threshold(validation_labels, validation_scores, args.target_defect_recall)
    results = {
        "validation": metrics(validation_labels, validation_scores, threshold),
        "test": metrics(test_labels, test_scores, threshold),
    }

    checkpoint = {
        "state_dict": best_state,
        "feature_dimension": feature_dimension,
        "dropout": args.dropout,
        "class_mapping": {"GOOD": GOOD, "DEFECT": DEFECT},
        "backbone": "dinov2_vitb14_frozen",
    }
    torch.save(checkpoint, artifact_root / "head.pt")
    (artifact_root / "training_config.json").write_text(json.dumps({"epochs_requested": args.epochs, "epochs_trained": len(history), "best_epoch": best_epoch, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "dropout": args.dropout, "class_weights": class_weights.detach().cpu().tolist(), "target_defect_recall": args.target_defect_recall, "threshold": threshold, "threshold_rule": threshold_rule}, indent=2) + "\n", encoding="utf-8")
    summary = {"dataset": "MVTec AD/screw", "protocol": "frozen DINOv2 global embedding + trainable LayerNorm/Dropout/Linear(768,2) head", "device": str(device), "counts": {"train": {"GOOD": int((train_y == GOOD).sum()), "DEFECT": int((train_y == DEFECT).sum())}, "validation": {"GOOD": int((validation_y == GOOD).sum()), "DEFECT": int((validation_y == DEFECT).sum())}, "test": {"GOOD": int((test_y == GOOD).sum()), "DEFECT": int((test_y == DEFECT).sum())}}, "best_epoch": best_epoch, "epochs_trained": len(history), "threshold": threshold, "threshold_rule": threshold_rule, "results": results, "checkpoint": str((artifact_root / "head.pt").resolve()), "history": history}
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# DINOv2 GOOD/DEFECT trainable head benchmark", "", "Frozen DINOv2 backbone with a trainable `LayerNorm → Dropout → Linear(768, 2)` head.", "", f"- Best epoch: `{best_epoch}` / `{len(history)}`", f"- Threshold: `{threshold:.6f}` ({threshold_rule})", "", "| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, report in results.items():
        lines.append(f"| {split} | {report['accuracy']:.4f} | {report['balanced_accuracy']:.4f} | {report['precision_defect']:.4f} | {report['recall_defect']:.4f} | {report['f1_defect']:.4f} | {report['image_auroc']:.4f} | {report['image_average_precision']:.4f} |")
    lines += ["", f"- Checkpoint: `{(artifact_root / 'head.pt').resolve()}`", f"- Manifest: `{manifest_path}`", "- The DINOv2 backbone was not updated in this run."]
    (report_root / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=Path("artifacts/production/dinov2_binary_screw/embeddings"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/dinov2_binary_screw/split_manifest.csv"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production/dinov2_binary_screw_linear_head"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/dinov2_binary_screw_linear_head"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0 or not 0 < args.target_defect_recall <= 1:
        parser.error("epochs/patience must be positive; learning rate positive; target recall in (0, 1]")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
