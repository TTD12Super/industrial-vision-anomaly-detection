"""Train DINOv2 GOOD/DEFECT on M2 and evaluate on three screw datasets.

Direction 1 protocol:
  - fit on M2 Train only;
  - choose the defect threshold on an M2 Train validation slice;
  - evaluate once on untouched M2 Test, MVTec screw Test, and CAD-SD Test.
"""

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


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
GOOD = 0
DEFECT = 1


def image_tensor(path: Path, config: DINOv2Config) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def rows_for_folder(root: Path, split_name: str) -> list[dict[str, str]]:
    """Read BODY/HEAD/{Train,Test}/{OK,NG} without depending on exact casing."""

    rows: list[dict[str, str]] = []
    for component in sorted(path for path in root.iterdir() if path.is_dir()):
        split_dir = next((p for p in component.iterdir() if p.is_dir() and p.name.lower() == split_name.lower()), None)
        if split_dir is None:
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            name = class_dir.name.lower()
            if name not in {"ok", "ng", "good", "bad", "defect"}:
                continue
            label = "GOOD" if name in {"ok", "good"} else "DEFECT"
            for path in image_files(class_dir):
                rows.append(
                    {
                        "image_path": str(path.resolve()),
                        "label": label,
                        "source_class": f"{component.name}/{class_dir.name}",
                        "component": component.name,
                    }
                )
    if not rows:
        raise ValueError(f"No M2 images found under {root} for split {split_name!r}")
    return rows


def m2_rows(root: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    train_rows = rows_for_folder(root, "Train")
    test_rows = rows_for_folder(root, "Test")
    rng = random.Random(seed)
    by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in train_rows:
        by_group.setdefault((row["component"], row["label"]), []).append(row)

    rows = {"train": [], "validation": [], "test": test_rows}
    for group_rows in by_group.values():
        rng.shuffle(group_rows)
        validation_count = max(1, round(len(group_rows) * 0.15))
        rows["validation"].extend(group_rows[:validation_count])
        rows["train"].extend(group_rows[validation_count:])
    for split in rows:
        rows[split].sort(key=lambda row: row["image_path"])
        labels = {row["label"] for row in rows[split]}
        if labels != {"GOOD", "DEFECT"}:
            raise ValueError(f"M2 {split} split must contain GOOD and DEFECT, got {labels}")
    return rows


def external_rows(mvtec_root: Path, cad_root: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}

    mvtec: list[dict[str, str]] = []
    for class_dir in sorted(path for path in (mvtec_root / "test").iterdir() if path.is_dir()):
        label = "GOOD" if class_dir.name == "good" else "DEFECT"
        for path in image_files(class_dir):
            mvtec.append({"image_path": str(path.resolve()), "label": label, "source_class": class_dir.name})
    rows["MVTec_test"] = mvtec

    cad: list[dict[str, str]] = []
    for class_dir in sorted(path for path in cad_root.iterdir() if path.is_dir()):
        label = "GOOD" if class_dir.name.lower().startswith("0_good") else "DEFECT"
        for path in image_files(class_dir):
            cad.append({"image_path": str(path.resolve()), "label": label, "source_class": class_dir.name})
    rows["CAD_SD_test"] = cad
    for name, items in rows.items():
        if not items or {row["label"] for row in items} != {"GOOD", "DEFECT"}:
            raise ValueError(f"{name} must contain GOOD and DEFECT images")
    return rows


def write_manifest(rows: dict[str, list[dict[str, str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("dataset_split", "image_path", "label", "source_class", "component")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_split, items in rows.items():
            for row in items:
                writer.writerow({field: row.get(field, "") for field in fields} | {"dataset_split": dataset_split})


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
        if offset and offset % (batch_size * 100) == 0:
            print(f"  embeddings {offset}/{len(items)}", flush=True)
    result = torch.cat(outputs).float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": expected_paths, "embeddings": result}, cache_path)
    return result, time.perf_counter() - start, False


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
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[GOOD, DEFECT]).ravel()
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
        "predicted_counts": {"good": int((predictions == GOOD).sum()), "defect": int((predictions == DEFECT).sum())},
    }


def counts(items: list[dict[str, str]]) -> dict[str, int]:
    return {
        "GOOD": sum(row["label"] == "GOOD" for row in items),
        "DEFECT": sum(row["label"] == "DEFECT" for row in items),
        "total": len(items),
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    classifier = summary["classifier"]
    lines = [
        "# Direction 1 — DINOv2 M2 screw benchmark",
        "",
        "## Protocol",
        "",
        "Frozen DINOv2 global embeddings + balanced logistic-regression GOOD/DEFECT head.",
        "The classifier is fit only on M2 Train. The threshold is selected only on an M2 Train validation slice; M2 Test, MVTec Test, and CAD-SD Test remain external evaluations.",
        "",
        f"- Device: `{summary['device']}`",
        f"- Threshold: `{classifier['threshold']:.6f}` ({classifier['threshold_rule']})",
        f"- Embedding time: `{summary['embedding_seconds']:.2f}s`",
        "",
        "## Dataset counts",
        "",
        "| Dataset/split | GOOD | DEFECT | Total |",
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["counts"].items():
        lines.append(f"| {name} | {values['GOOD']} | {values['DEFECT']} | {values['total']} |")
    lines += [
        "",
        "## Results",
        "",
        "| Dataset/split | Accuracy | Balanced acc. | GOOD recall | DEFECT precision | DEFECT recall | F1 | AUROC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in summary["results"].items():
        lines.append(
            f"| {name} | {report['accuracy']:.4f} | {report['balanced_accuracy']:.4f} | {report['good_recall']:.4f} | {report['precision_defect']:.4f} | {report['recall_defect']:.4f} | {report['f1_defect']:.4f} | {report['image_auroc']:.4f} | {report['image_average_precision']:.4f} |"
        )
    lines += [
        "",
        "Confusion matrices use `[[GOOD→GOOD, GOOD→DEFECT], [DEFECT→GOOD, DEFECT→DEFECT]]`.",
        "",
        f"- Classifier artifact: `{summary['classifier_artifact']}`",
        f"- Manifest: `{summary['manifest']}`",
        "- No GitHub push was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    m2_root = Path(args.m2_root).resolve()
    mvtec_root = Path(args.mvtec_root).resolve()
    cad_root = Path(args.cad_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    report_root = Path(args.report_root).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    m2 = m2_rows(m2_root, args.seed)
    external = external_rows(mvtec_root, cad_root)
    all_rows = {**m2, **external}
    write_manifest(all_rows, report_root / "manifest.csv")

    device = args.device or str(select_device())
    config = DINOv2Config(image_size=(256, 256), feature_layers=())
    encoder = SharedDINOv2(config, artifact_root=args.encoder_root, device=device)
    embedding_seconds = 0.0
    cache_reused = 0
    features: dict[str, torch.Tensor] = {}
    for name, items in all_rows.items():
        print(f"Extracting {name}: {len(items)} images", flush=True)
        features[name], elapsed, reused = embeddings(
            items, encoder, args.batch_size, artifact_root / "embeddings_m2" / f"{name}.pt", args.force
        )
        embedding_seconds += elapsed
        cache_reused += int(reused)

    train_labels = torch.tensor([GOOD if row["label"] == "GOOD" else DEFECT for row in m2["train"]], dtype=torch.long)
    train_embeddings = features["train"]
    classifier = fit_linear_classifier(
        {"embeddings": train_embeddings, "labels": train_labels},
        {"GOOD": GOOD, "DEFECT": DEFECT},
        LinearClassifierConfig(),
    )
    classifier_root = artifact_root / "classifier_m2"
    save_linear_classifier(
        classifier_root,
        classifier,
        {"GOOD": GOOD, "DEFECT": DEFECT},
        LinearClassifierConfig(),
        {"artifact_version": "v1", "source": "dinov2_frozen_binary_screw_m2"},
    )

    validation_labels = np.array([GOOD if row["label"] == "GOOD" else DEFECT for row in m2["validation"]])
    _, validation_probabilities = predict_linear_classifier(classifier, features["validation"])
    threshold, threshold_rule = choose_threshold(validation_labels, validation_probabilities[:, DEFECT].numpy(), args.target_defect_recall)

    results: dict[str, dict[str, object]] = {}
    for name, items in all_rows.items():
        labels = np.array([GOOD if row["label"] == "GOOD" else DEFECT for row in items])
        _, probabilities = predict_linear_classifier(classifier, features[name])
        results[name] = metrics(labels, probabilities[:, DEFECT].numpy(), threshold)

    summary: dict[str, object] = {
        "protocol": "train M2 Train; calibrate M2 Train validation; evaluate M2 Test + MVTec Test + CAD-SD Test",
        "m2_root": str(m2_root),
        "mvtec_root": str(mvtec_root),
        "cad_root": str(cad_root),
        "device": device,
        "encoder": encoder.metadata(),
        "embedding_seconds": embedding_seconds,
        "embedding_cache_reused_splits": cache_reused,
        "counts": {name: counts(items) for name, items in all_rows.items()},
        "classifier": {
            "type": "frozen DINOv2 global embedding + logistic regression",
            "training_images": len(m2["train"]),
            "validation_images": len(m2["validation"]),
            "threshold": threshold,
            "threshold_rule": threshold_rule,
        },
        "results": results,
        "classifier_artifact": str((classifier_root / "classifier.pt").resolve()),
        "manifest": str((report_root / "manifest.csv").resolve()),
        "note": "This is the requested Direction 1 experiment; it does not use MVTec or CAD-SD for training or threshold selection.",
    }
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_root / "benchmark.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=Path("datasets/external/m2_screw/extracted"))
    parser.add_argument("--mvtec-root", type=Path, default=Path("datasets/mvtec_ad_screw_compare_processed/screw"))
    parser.add_argument("--cad-root", type=Path, default=Path("datasets/external/cad_sd/data/unpacked/Co-occurrence Anomaly Detection Screw Dataset/Test"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production/dinov2_binary_screw_m2"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/dinov2_binary_screw_m2"))
    parser.add_argument("--encoder-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-defect-recall", type=float, default=0.90)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_defect_recall <= 1.0:
        parser.error("--target-defect-recall must be in (0, 1]")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
