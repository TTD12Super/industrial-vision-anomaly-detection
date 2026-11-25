#!/usr/bin/env python3
"""Fine-tune the last DINOv2 block with a global + patch-aware binary head."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
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
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvtec_system.core.dinov2 import DINOv2Config
from inspection.classifier_sweep import select_operating_point


GOOD = 0
DEFECT = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_rows(manifest: Path) -> dict[str, list[dict[str, str]]]:
    rows = {"train": [], "validation": [], "test": []}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in rows:
                raise ValueError(f"Unexpected split: {row['split']}")
            if row["label"] not in {"GOOD", "DEFECT"}:
                raise ValueError(f"Unexpected label: {row['label']}")
            rows[row["split"]].append(row)
    for split, items in rows.items():
        if not items or {row["label"] for row in items} != {"GOOD", "DEFECT"}:
            raise ValueError(f"{split} must contain GOOD and DEFECT")
    return rows


class ScrewDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], config: DINOv2Config) -> None:
        self.rows = rows
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB").resize(
                (self.config.image_size[1], self.config.image_size[0]), Image.Resampling.LANCZOS
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        mean = torch.tensor(self.config.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.config.std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, GOOD if row["label"] == "GOOD" else DEFECT


class GlobalPatchHead(nn.Module):
    """Fuse global logits with the strongest local patch logits."""

    def __init__(self, feature_dimension: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.top_k = top_k
        self.global_head = nn.Sequential(nn.LayerNorm(feature_dimension), nn.Dropout(dropout), nn.Linear(feature_dimension, 2))
        self.patch_head = nn.Sequential(nn.LayerNorm(feature_dimension), nn.Linear(feature_dimension, 2))
        self.fusion = nn.Linear(4, 2)

    def forward(self, global_token: torch.Tensor, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_logits = self.global_head(global_token)
        patch_logits = self.patch_head(patch_tokens)
        k = min(self.top_k, patch_logits.shape[1])
        indices = patch_logits[:, :, DEFECT].topk(k=k, dim=1).indices
        top_patch_logits = torch.gather(patch_logits, 1, indices.unsqueeze(-1).expand(-1, -1, 2)).mean(dim=1)
        combined = self.fusion(torch.cat((global_logits, top_patch_logits), dim=1))
        return combined, patch_logits[:, :, DEFECT]


class DINOv2LastBlockClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: GlobalPatchHead, padding: tuple[int, int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.padding = padding

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_bottom, pad_right = self.padding
        outputs = self.backbone.forward_features(F.pad(images, (0, pad_right, 0, pad_bottom), mode="replicate"))
        global_token = outputs["x_norm_clstoken"]
        patch_tokens = outputs["x_norm_patchtokens"]
        return self.head(global_token, patch_tokens)


def set_trainable_layers(backbone: nn.Module) -> list[nn.Parameter]:
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    blocks = getattr(backbone, "blocks", None)
    if not blocks:
        raise AttributeError("DINOv2 backbone does not expose transformer blocks")
    for parameter in blocks[-1].parameters():
        parameter.requires_grad = True
    final_norm = getattr(backbone, "norm", None)
    if final_norm is not None:
        for parameter in final_norm.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in backbone.parameters() if parameter.requires_grad]


def load_backbone(config: DINOv2Config, artifact_root: Path, device: torch.device) -> nn.Module:
    hub_dir = artifact_root / "torch_hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    backbone = torch.hub.load(config.repository, config.model_name, pretrained=True, trust_repo=True)
    return backbone.to(device)


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, _ = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
    return total_loss / max(1, total)


def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits, _ = model(images.to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, DEFECT].cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


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


def _gate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    prediction = scores >= threshold
    defect = labels == DEFECT
    good = labels == GOOD
    defect_count = int(np.sum(defect))
    good_count = int(np.sum(good))
    defect_true_positive = int(np.sum(prediction & defect))
    good_false_alarm_count = int(np.sum(prediction & good))
    return {
        "recall": float(defect_true_positive / max(1, defect_count)),
        "good_fp": float(good_false_alarm_count / max(1, good_count)),
        "defect_true_positive": defect_true_positive,
        "defect_count": defect_count,
        "good_false_alarm_count": good_false_alarm_count,
        "good_count": good_count,
    }


def threshold_sweep(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    diagnostic_labels: np.ndarray,
    diagnostic_scores: np.ndarray,
) -> list[dict[str, object]]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], validation_scores)))
    return [
        {
            "threshold": float(threshold),
            "validation": _gate_metrics(validation_labels, validation_scores, float(threshold)),
            "test_diagnostic": _gate_metrics(diagnostic_labels, diagnostic_scores, float(threshold)),
        }
        for threshold in candidates
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    config = DINOv2Config(image_size=(256, 256), feature_layers=())
    rows = read_rows(args.manifest.resolve())
    device = torch.device(args.device)
    datasets = {split: ScrewDataset(items, config) for split, items in rows.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    backbone = load_backbone(config, args.encoder_root.resolve(), device)
    trainable_backbone = set_trainable_layers(backbone)
    backbone.eval()
    head = GlobalPatchHead(int(backbone.embed_dim), args.top_k, args.dropout).to(device)
    model = DINOv2LastBlockClassifier(backbone, head, padding=(10, 10))
    class_counts = torch.tensor([sum(row["label"] == "GOOD" for row in rows["train"]), sum(row["label"] == "DEFECT" for row in rows["train"])], dtype=torch.float32)
    class_weights = (class_counts.sum() / (2.0 * class_counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone, "lr": args.backbone_learning_rate},
        {"params": head.parameters(), "lr": args.head_learning_rate},
    ], weight_decay=args.weight_decay)

    best_state = None
    best_score = (-float("inf"), -float("inf"))
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], loss_fn, optimizer, device)
        validation_labels, validation_scores = infer(model, loaders["validation"], device)
        validation_auroc = roc_auc_score(validation_labels, validation_scores)
        validation_balanced = balanced_accuracy_score(validation_labels, validation_scores >= 0.5)
        score = (float(validation_auroc), float(validation_balanced))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_auroc": float(validation_auroc), "validation_balanced_accuracy_at_0_5": float(validation_balanced)})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"epoch {epoch}/{args.epochs} loss={train_loss:.4f} val_auroc={validation_auroc:.4f}", flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint produced")
    model.load_state_dict(best_state)
    validation_labels, validation_scores = infer(model, loaders["validation"], device)
    test_labels, test_scores = infer(model, loaders["test"], device)
    points = threshold_sweep(validation_labels, validation_scores, test_labels, test_scores)
    operating_point = select_operating_point(points, args.target_defect_recall, args.max_good_fp)
    if operating_point["selected"] is not None:
        threshold = float(operating_point["selected"]["threshold"])
        threshold_rule = operating_point["selection_rule"]
        runtime_threshold_gated = True
    else:
        threshold, threshold_rule = choose_threshold(validation_labels, validation_scores, args.target_defect_recall)
        threshold_rule = f"EXPLORATORY_FALLBACK_NOT_GATED: {threshold_rule}"
        runtime_threshold_gated = False
    results = {"validation": metrics(validation_labels, validation_scores, threshold), "test": metrics(test_labels, test_scores, threshold)}

    artifact_root = args.artifact_root.resolve(); report_root = args.report_root.resolve(); artifact_root.mkdir(parents=True, exist_ok=True); report_root.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": "dinov2_vitb14_last_block_finetuned", "feature_dimension": int(backbone.embed_dim), "top_k": args.top_k, "dropout": args.dropout, "class_mapping": {"GOOD": GOOD, "DEFECT": DEFECT}}, artifact_root / "model.pt")
    threshold_report = {
        "schema_version": "1.0",
        "model": str((artifact_root / "model.pt").resolve()),
        "manifest": str(args.manifest.resolve()),
        "selected_validation_gate": operating_point["selected"],
        "validation_gate_exists": operating_point["selected"] is not None,
        "selection": operating_point,
        "points": points,
        "runtime_threshold": threshold,
        "runtime_threshold_gated": runtime_threshold_gated,
    }
    (report_root / "threshold_sweep.json").write_text(json.dumps(threshold_report, indent=2) + "\n", encoding="utf-8")
    summary = {"dataset": "MVTec AD/screw", "protocol": "DINOv2 last Transformer block + global/patch-aware GOOD/DEFECT head", "device": str(device), "counts": {split: {"GOOD": sum(row["label"] == "GOOD" for row in items), "DEFECT": sum(row["label"] == "DEFECT" for row in items), "total": len(items)} for split, items in rows.items()}, "best_epoch": best_epoch, "epochs_trained": len(history), "threshold": threshold, "threshold_rule": threshold_rule, "runtime_threshold_gated": runtime_threshold_gated, "target_defect_recall": args.target_defect_recall, "max_good_fp": args.max_good_fp, "results": results, "checkpoint": str((artifact_root / "model.pt").resolve()), "threshold_sweep": str((report_root / "threshold_sweep.json").resolve()), "history": history}
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# DINOv2 last-block + patch-aware GOOD/DEFECT benchmark", "", "DINOv2 ViT-B/14 with the last Transformer block unfrozen. The classifier fuses global-token logits with top-k patch-token logits.", "", f"- Best epoch: `{best_epoch}` / `{len(history)}`", f"- Threshold: `{threshold:.6f}` ({threshold_rule})", "", "| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, report in results.items():
        lines.append(f"| {split} | {report['accuracy']:.4f} | {report['balanced_accuracy']:.4f} | {report['precision_defect']:.4f} | {report['recall_defect']:.4f} | {report['f1_defect']:.4f} | {report['image_auroc']:.4f} | {report['image_average_precision']:.4f} |")
    lines += ["", f"- Checkpoint: `{(artifact_root / 'model.pt').resolve()}`", "- YOLO was not retrained in this experiment."]
    (report_root / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/dinov2_binary_screw/split_manifest.csv"))
    parser.add_argument("--encoder-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production/dinov2_lastblock_patch_head"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/dinov2_lastblock_patch_head"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--max-good-fp", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.top_k < 1:
        parser.error("epochs, patience, batch-size and top-k must be positive")
    if not 0.0 <= args.target_defect_recall <= 1.0 or not 0.0 <= args.max_good_fp <= 1.0:
        parser.error("target-defect-recall and max-good-fp must be in [0, 1]")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
