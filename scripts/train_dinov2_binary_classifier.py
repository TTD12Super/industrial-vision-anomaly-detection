"""Train and benchmark frozen DINOv2 GOOD/DEFECT classification for one MVTec category."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvtec_system.classification.classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device


def image_tensor(path: Path, config: DINOv2Config) -> torch.Tensor:
    """Read either processed 256px images or raw MVTec images."""

    with Image.open(path) as source:
        image = source.convert("RGB")
        image = image.resize((config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def split_rows(root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    """Use train/good for training and split official test images into dev/test."""

    rows: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    good_train = files(root / "train" / "good")
    good_test = files(root / "test" / "good")
    if not good_train or not good_test:
        raise ValueError(f"Expected train/good and test/good under {root}")

    rng = random.Random(seed)
    rng.shuffle(good_test)
    good_validation = good_test[: len(good_test) // 2]
    good_holdout = good_test[len(good_test) // 2 :]
    rows["train"].extend({"image_path": str(path), "label": "GOOD", "source_class": "good"} for path in good_train)
    rows["validation"].extend({"image_path": str(path), "label": "GOOD", "source_class": "good"} for path in good_validation)
    rows["test"].extend({"image_path": str(path), "label": "GOOD", "source_class": "good"} for path in good_holdout)

    defect_classes = sorted(
        path.name for path in (root / "test").iterdir() if path.is_dir() and path.name != "good"
    )
    if not defect_classes:
        raise ValueError(f"No defect classes found under {root / 'test'}")
    for defect_class in defect_classes:
        defect_files = files(root / "test" / defect_class)
        rng.shuffle(defect_files)
        train_count = max(1, round(len(defect_files) * 0.70))
        validation_count = max(1, round(len(defect_files) * 0.15))
        if train_count + validation_count >= len(defect_files):
            validation_count = max(1, len(defect_files) - train_count - 1)
        selections = (
            ("train", defect_files[:train_count]),
            ("validation", defect_files[train_count : train_count + validation_count]),
            ("test", defect_files[train_count + validation_count :]),
        )
        for split, selected in selections:
            rows[split].extend(
                {"image_path": str(path), "label": "DEFECT", "source_class": defect_class}
                for path in selected
            )

    for split in rows:
        rows[split].sort(key=lambda row: row["image_path"])
        if not rows[split] or {row["label"] for row in rows[split]} != {"GOOD", "DEFECT"}:
            raise ValueError(f"{split} split must contain both GOOD and DEFECT images")
    return rows


def write_manifest(rows: dict[str, list[dict[str, str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "label", "source_class", "split"))
        writer.writeheader()
        for split, items in rows.items():
            for row in items:
                writer.writerow({**row, "split": split})


def embeddings(
    items: list[dict[str, str]],
    encoder: SharedDINOv2,
    batch_size: int,
    cache_path: Path,
    force: bool,
) -> tuple[torch.Tensor, float, bool]:
    expected_paths = [row["image_path"] for row in items]
    if cache_path.is_file() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("paths") == expected_paths:
            return cached["embeddings"].float(), 0.0, True

    start = time.perf_counter()
    outputs = []
    for offset in range(0, len(items), batch_size):
        batch_paths = [Path(row["image_path"]) for row in items[offset : offset + batch_size]]
        batch = torch.stack([image_tensor(path, encoder.config) for path in batch_paths])
        outputs.append(encoder.forward(batch).global_embedding.cpu())
    result = torch.cat(outputs).float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": expected_paths, "embeddings": result}, cache_path)
    return result, time.perf_counter() - start, False


def balance_training(embeddings_: torch.Tensor, labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    good = embeddings_[labels == 0]
    defect = embeddings_[labels == 1]
    if not len(good) or not len(defect):
        raise ValueError("Training split must contain both GOOD and DEFECT images")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(defect), (len(good),), generator=generator)
    balanced = torch.cat((good, defect[indices]))
    balanced_labels = torch.cat((torch.zeros(len(good), dtype=torch.long), torch.ones(len(good), dtype=torch.long)))
    order = torch.randperm(len(balanced), generator=generator)
    return balanced[order], balanced_labels[order]


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, target_recall: float) -> tuple[float, str]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    eligible = [float(value) for value in candidates if recall_score(labels, probabilities >= value, zero_division=0) >= target_recall]
    if eligible:
        return max(eligible), f"highest threshold meeting defect recall >= {target_recall:.2f}"
    ranked = []
    for value in candidates:
        predictions = probabilities >= value
        ranked.append((balanced_accuracy_score(labels, predictions), recall_score(labels, predictions, zero_division=0), -float(value), float(value)))
    return max(ranked)[3], "best validation balanced accuracy; target recall was unattainable"


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "images": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_defect": float(precision_score(labels, predictions, zero_division=0)),
        "recall_defect": float(recall_score(labels, predictions, zero_division=0)),
        "f1_defect": float(f1_score(labels, predictions, zero_division=0)),
        "good_recall": float(tn / (tn + fp) if tn + fp else 0.0),
        "image_auroc": float(roc_auc_score(labels, probabilities)),
        "image_average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix_good_defect": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "predicted_counts": {"good": int((predictions == 0).sum()), "defect": int((predictions == 1).sum())},
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    classifier = summary["classifier"]
    validation = classifier["validation"]
    test = classifier["test"]
    lines = [
        "# DINOv2 GOOD/DEFECT classifier benchmark",
        "",
        "## Protocol",
        "",
        "Frozen DINOv2 global embeddings + balanced logistic-regression binary head.",
        "The original MVTec AD train/good images are training data. Official test images are split into validation and held-out test, so this is a supervised development benchmark, not the official MVTec AD protocol.",
        "",
        f"- Dataset: `{summary['dataset_root']}`",
        f"- Device: `{summary['device']}`",
        f"- DINOv2 embeddings: `{summary['embedding_cache_reused_splits']}/3` split caches reused; recomputation time `{summary['embedding_seconds']:.2f}s`",
        f"- Threshold rule: {classifier['threshold_rule']}",
        "",
        "## Split",
        "",
        "| Split | GOOD | DEFECT | Total |",
        "|---|---:|---:|---:|",
    ]
    for split, counts in summary["counts"].items():
        lines.append(f"| {split} | {counts['GOOD']} | {counts['DEFECT']} | {counts['total']} |")
    lines += [
        "",
        "## Metrics",
        "",
        "| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in (("Validation", validation), ("Held-out test", test)):
        lines.append(
            f"| {name} | {report['accuracy']:.4f} | {report['balanced_accuracy']:.4f} | {report['precision_defect']:.4f} | {report['recall_defect']:.4f} | {report['f1_defect']:.4f} | {report['image_auroc']:.4f} | {report['image_average_precision']:.4f} |"
        )
    lines += [
        "",
        f"- Selected threshold: `{classifier['threshold']:.6f}`",
        f"- Held-out confusion matrix `[[TN, FP], [FN, TP]]`: `{test['confusion_matrix_good_defect']}`",
        f"- Classifier artifact: `{summary['classifier_artifact']}`",
        f"- Split manifest: `{summary['split_manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.dataset_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    report_root = Path(args.report_root).resolve()
    rows = split_rows(root, args.seed)
    report_root.mkdir(parents=True, exist_ok=True)
    split_manifest = report_root / "split_manifest.csv"
    write_manifest(rows, split_manifest)

    device = args.device or str(select_device())
    config = DINOv2Config(image_size=(256, 256), feature_layers=())
    encoder = SharedDINOv2(config, artifact_root=args.encoder_root, device=device)
    total_seconds = 0.0
    cache_reused_splits = 0
    all_embeddings: dict[str, torch.Tensor] = {}
    for split, items in rows.items():
        all_embeddings[split], elapsed, reused = embeddings(
            items,
            encoder,
            args.batch_size,
            artifact_root / "embeddings" / f"{split}.pt",
            args.force,
        )
        total_seconds += elapsed
        cache_reused_splits += int(reused)

    labels = {
        split: torch.tensor([0 if row["label"] == "GOOD" else 1 for row in items], dtype=torch.long)
        for split, items in rows.items()
    }
    train_embeddings, train_labels = balance_training(all_embeddings["train"], labels["train"], args.seed)
    classifier = fit_linear_classifier(
        {"embeddings": train_embeddings, "labels": train_labels},
        {"GOOD": 0, "DEFECT": 1},
        LinearClassifierConfig(),
    )
    classifier_root = artifact_root / "classifier"
    save_linear_classifier(
        classifier_root,
        classifier,
        {"GOOD": 0, "DEFECT": 1},
        LinearClassifierConfig(),
        {"artifact_version": "v1", "source": "dinov2_frozen_binary_screw"},
    )
    _, validation_probabilities = predict_linear_classifier(classifier, all_embeddings["validation"])
    threshold, threshold_rule = choose_threshold(
        labels["validation"].numpy(), validation_probabilities[:, 1].numpy(), args.target_defect_recall
    )
    _, test_probabilities = predict_linear_classifier(classifier, all_embeddings["test"])
    reports = {
        "validation": metrics(labels["validation"].numpy(), validation_probabilities[:, 1].numpy(), threshold),
        "test": metrics(labels["test"].numpy(), test_probabilities[:, 1].numpy(), threshold),
    }
    predictions = []
    for row, probability in zip(rows["test"], test_probabilities[:, 1].tolist()):
        predictions.append(
            {
                "image": row["image_path"],
                "true_label": row["label"],
                "source_class": row["source_class"],
                "probability_defect": float(probability),
                "decision": "DEFECT" if probability >= threshold else "GOOD",
            }
        )
    (report_root / "test_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")

    counts = {
        split: {
            "GOOD": sum(row["label"] == "GOOD" for row in items),
            "DEFECT": sum(row["label"] == "DEFECT" for row in items),
            "total": len(items),
        }
        for split, items in rows.items()
    }
    summary: dict[str, object] = {
        "dataset": "MVTec AD/screw",
        "dataset_root": str(root),
        "device": device,
        "encoder": encoder.metadata(),
        "counts": counts,
        "embedding_images": sum(counts[split]["total"] for split in counts),
        "embedding_seconds": total_seconds,
        "embedding_cache_reused_splits": cache_reused_splits,
        "classifier": {
            "type": "frozen DINOv2 global embedding + logistic regression",
            "training_images_before_balance": len(rows["train"]),
            "training_images_after_balance": len(train_labels),
            "threshold": threshold,
            "threshold_rule": threshold_rule,
            **reports,
        },
        "classifier_artifact": str((classifier_root / "classifier.pt").resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "test_predictions": str((report_root / "test_predictions.json").resolve()),
        "benchmark_note": "Official test images were partitioned for supervised development; do not compare this directly with the original one-class MVTec AD benchmark.",
    }
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_root / "benchmark.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mvtec_ad_screw_compare_processed/screw"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production/dinov2_binary_screw"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/dinov2_binary_screw"))
    parser.add_argument("--encoder-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--force", action="store_true", help="recompute cached DINOv2 embeddings")
    args = parser.parse_args()
    if not 0.0 < args.target_defect_recall <= 1.0:
        parser.error("--target-defect-recall must be in (0, 1]")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
