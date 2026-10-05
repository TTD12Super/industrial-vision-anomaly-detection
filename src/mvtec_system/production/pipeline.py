"""One-command automation for an arbitrary MVTec-style dataset."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import torch

from ..anomaly.feature_bank import FeatureBankResult, load_or_build_feature_bank
from ..anomaly.knn import nearest_neighbor_scores
from ..anomaly.maps import make_anomaly_maps, plot_anomaly_examples
from ..anomaly.thresholds import (
    binary_anomaly_masks,
    calibrate_category_image_thresholds,
    calibrate_category_pixel_thresholds,
    classify_image_score,
)
from ..classification.calibration import calibrate_open_set_thresholds, save_calibration
from ..classification.classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    save_linear_classifier,
)
from ..classification.embeddings import (
    build_global_embedding_artifact,
    category_mapping,
    make_embedding_metadata,
    read_classification_manifest,
    save_embedding_artifacts,
)
from ..classification.product_test_evaluation import discover_mvtec_test_images
from ..core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor
from ..data.preprocessing import (
    category_directories,
    create_classification_manifest,
    discover_dataset_root,
    image_files,
    prepare_mvtec_dataset,
    validate_raw_dataset,
)
from ..evaluation.visualization import plot_dataset_distribution, save_figure


@dataclass(frozen=True)
class ProductionConfig:
    """All run settings; dataset and category names are discovered at runtime."""

    datasets_root: Path = Path("datasets")
    dataset_root: Path | None = None
    processed_root: Path | None = None
    report_root: Path = Path("reports/production")
    artifact_root: Path = Path("artifacts/production")
    image_size: int = 256
    validation_fraction: float = 0.30
    seed: int = 42
    batch_size: int = 8
    k: int = 1
    target_acceptance_rate: float = 0.99
    classification_confidence_threshold: float = 0.50
    device: str | None = None
    force: bool = False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    size = max(8.0, len(labels) * 0.55)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="True category",
        title=title,
    )
    axis.tick_params(axis="x", rotation=45)
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
                fontsize=7,
            )
    save_figure(figure, path)


def _score_with_bank(
    image_path: Path,
    encoder: SharedDINOv2,
    bank: torch.Tensor,
    k: int,
) -> tuple[float, np.ndarray]:
    """Run one frozen forward pass and exact kNN against one category bank."""

    with torch.inference_mode():
        features = encoder.forward(torch.stack([image_tensor(image_path, encoder.config)]))
    patches = features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1])
    patch_scores = nearest_neighbor_scores(patches, bank, k=k).reshape(1, -1)
    score = float(patch_scores.max().item())
    anomaly_map = make_anomaly_maps(
        patch_scores,
        encoder.patch_grid,
        encoder.padded_size,
        encoder.config.image_size,
    )[0]
    return score, anomaly_map


def _calibrate_anomaly_thresholds(
    rows: list[dict[str, str]],
    processed_root: Path,
    banks: dict[str, FeatureBankResult],
    encoder: SharedDINOv2,
    config: ProductionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_scores: dict[str, list[float]] = {}
    pixel_scores: dict[str, list[float]] = {}
    for category in sorted(banks):
        selected = [row for row in rows if row["category"] == category and row["split"] == "validation"]
        if not selected:
            selected = [row for row in rows if row["category"] == category and row["split"] == "train"][:1]
        scores: list[float] = []
        pixels: list[np.ndarray] = []
        for row in selected:
            score, anomaly_map = _score_with_bank(
                processed_root / row["image_path"], encoder, banks[category].features, config.k
            )
            scores.append(score)
            pixels.append(anomaly_map.reshape(-1))
        image_scores[category] = scores
        pixel_scores[category] = np.concatenate(pixels).tolist()
    return (
        calibrate_category_image_thresholds(image_scores, config.target_acceptance_rate),
        calibrate_category_pixel_thresholds(pixel_scores, config.target_acceptance_rate),
    )


def _safe_ranking(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    if len(set(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _anomaly_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("anomaly_evaluated")]
    scores = [float(row["anomaly_score"]) for row in evaluated]
    labels = [int(row["is_defect"]) for row in evaluated]
    predictions = [int(row.get("is_defect", False)) for row in evaluated]
    report: dict[str, Any] = {
        "test_images": len(rows),
        "anomaly_evaluated_images": len(evaluated),
        "anomaly_skipped_images": len(rows) - len(evaluated),
        "image": {"ranking": _safe_ranking(scores, labels)},
    }
    if evaluated:
        report["image"]["thresholded"] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }
    pixel_scores: list[float] = []
    pixel_labels: list[int] = []
    pixel_predictions: list[int] = []
    for row in evaluated:
        truth = np.asarray(row["mask"], dtype=np.uint8).reshape(-1)
        score_map = np.asarray(row["anomaly_map"], dtype=np.float32).reshape(-1)
        prediction = np.asarray(row["binary_mask"], dtype=np.uint8).reshape(-1)
        pixel_scores.extend(score_map.tolist())
        pixel_labels.extend((truth > 0).astype(np.int64).tolist())
        pixel_predictions.extend(prediction.astype(np.int64).tolist())
    report["pixel"] = {"ranking": _safe_ranking(pixel_scores, pixel_labels)}
    if pixel_predictions:
        report["pixel"]["thresholded"] = {
            "precision": float(precision_score(pixel_labels, pixel_predictions, zero_division=0)),
            "recall": float(recall_score(pixel_labels, pixel_predictions, zero_division=0)),
            "f1": float(f1_score(pixel_labels, pixel_predictions, zero_division=0)),
        }
    return report


def _classification_report(rows: list[dict[str, Any]], categories: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known = {category: index for index, category in enumerate(categories)}
    true = [known[row["true_category"]] for row in rows]
    accepted = [row for row in rows if not row["rejected"]]
    predictions = [known[row["category"]] if not row["rejected"] else -1 for row in rows]
    correct = sum(not row["rejected"] and row["category"] == row["true_category"] for row in rows)
    rejected = sum(row["rejected"] for row in rows)
    accepted_correct = sum(row["category"] == row["true_category"] for row in accepted)
    per_category = []
    for category in categories:
        subset = [row for row in rows if row["true_category"] == category]
        category_rejected = sum(row["rejected"] for row in subset)
        category_correct = sum(not row["rejected"] and row["category"] == category for row in subset)
        category_incorrect = len(subset) - category_rejected - category_correct
        per_category.append(
            {
                "category": category,
                "support": len(subset),
                "correct": category_correct,
                "incorrect": category_incorrect,
                "rejected": category_rejected,
                "accuracy": category_correct / len(subset) if subset else 0.0,
                "rejection_rate": category_rejected / len(subset) if subset else 0.0,
            }
        )
    summary = {
        "total_test_images": len(rows),
        "accepted_known_predictions": len(accepted),
        "unknown_rejected_predictions": rejected,
        "rejection_rate": rejected / len(rows) if rows else 0.0,
        "top1_accuracy_all": correct / len(rows) if rows else 0.0,
        "top1_accuracy_accepted": accepted_correct / len(accepted) if accepted else 0.0,
        "macro_precision": float(precision_score(true, predictions, labels=list(range(len(categories))), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, predictions, labels=list(range(len(categories))), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, predictions, labels=list(range(len(categories))), average="macro", zero_division=0)),
        "per_category": per_category,
    }
    for split_name, split_rows in (
        ("good_test", [row for row in rows if not row["is_defect"]]),
        ("defect_test", [row for row in rows if row["is_defect"]]),
    ):
        split_accepted = [row for row in split_rows if not row["rejected"]]
        split_correct = sum(
            not row["rejected"] and row["category"] == row["true_category"]
            for row in split_rows
        )
        summary[split_name] = {
            "total_images": len(split_rows),
            "accepted_known_predictions": len(split_accepted),
            "unknown_rejected_predictions": len(split_rows) - len(split_accepted),
            "accuracy": split_correct / len(split_rows) if split_rows else 0.0,
            "accepted_accuracy": (
                sum(row["category"] == row["true_category"] for row in split_accepted)
                / len(split_accepted)
                if split_accepted
                else 0.0
            ),
            "rejection_rate": (
                (len(split_rows) - len(split_accepted)) / len(split_rows)
                if split_rows
                else 0.0
            ),
        }
    return summary, per_category


def _load_mask(item: dict[str, Any], image_size: int) -> np.ndarray:
    if not item["is_defect"]:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    image = Path(item["image_path"])
    mask_path = image.parents[2] / "ground_truth" / item["defect_type"] / f"{image.stem}_mask.png"
    if not mask_path.is_file():
        return np.zeros((image_size, image_size), dtype=np.uint8)
    with Image.open(mask_path) as mask:
        return np.asarray(mask.convert("L").resize((image_size, image_size))) > 0


def _markdown_report(summary: dict[str, Any], report_dir: Path) -> str:
    classification = summary["classification"]
    anomaly = summary["anomaly"]
    lines = [
        f"# Production run report: `{summary['dataset_name']}`",
        "",
        "This report was generated automatically after the one-command pipeline completed.",
        "",
        "## Pipeline",
        "",
        "`raw validation → preprocessing → train/validation split → feature bank → classification → kNN defect detection → heatmaps → evaluation`",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Processed root: `{summary['processed_root']}`",
        f"- Categories: **{len(summary['categories'])}**",
        f"- Classification: **{'enabled' if summary['classification_enabled'] else 'disabled (one category)'}**",
        f"- Product confidence threshold: **{summary['configuration']['classification_confidence_threshold']:.2f}**",
        f"- kNN neighbors: **{summary['configuration']['k']}**",
        "",
        "## Product classification",
        "",
        f"- Official test images: **{classification['total_test_images']}**",
        f"- Top-1 accuracy over all images: **{classification['top1_accuracy_all']:.2%}**",
        f"- Top-1 accuracy over accepted images: **{classification['top1_accuracy_accepted']:.2%}**",
        f"- UNKNOWN/rejected: **{classification['unknown_rejected_predictions']} ({classification['rejection_rate']:.2%})**",
        f"- Macro precision / recall / F1: **{classification['macro_precision']:.4f} / {classification['macro_recall']:.4f} / {classification['macro_f1']:.4f}**",
        f"- Good-image accuracy: **{classification['good_test']['accuracy']:.2%}**; defect-image accuracy: **{classification['defect_test']['accuracy']:.2%}**",
        f"- Accuracy drop on defect images: **{classification['good_test']['accuracy'] - classification['defect_test']['accuracy']:.2%}**",
        "",
        "## Defect detection",
        "",
        f"- Images with anomaly inference: **{anomaly['anomaly_evaluated_images']}**",
        f"- Images skipped because routing rejected them: **{anomaly['anomaly_skipped_images']}**",
        f"- Image AUROC / AP: **{anomaly['image']['ranking']['auroc']} / {anomaly['image']['ranking']['average_precision']}**",
        f"- Pixel AUROC / AP: **{anomaly['pixel']['ranking']['auroc']} / {anomaly['pixel']['ranking']['average_precision']}**",
        "",
        "## Generated files",
        "",
    ]
    for name in summary["output_files"].values():
        lines.append(f"- `{report_dir / name}`")
    return "\n".join(lines) + "\n"


def run_production(config: ProductionConfig | None = None) -> dict[str, Any]:
    """Run the complete dataset-driven pipeline and write the final report."""

    config = config or ProductionConfig()
    raw_root = discover_dataset_root(config.datasets_root, config.dataset_root)
    processed_root = (config.processed_root or raw_root.parent / f"{raw_root.name}_processed").resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (config.report_root / raw_root.name / run_name).resolve()
    artifact_dir = (config.artifact_root / raw_root.name).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    preparation = prepare_mvtec_dataset(raw_root, processed_root, config.image_size, config.force)
    raw_validation = validate_raw_dataset(raw_root)
    _write_json(report_dir / "dataset_validation.json", preparation)
    plot_dataset_distribution(raw_validation, report_dir / "dataset_distribution.png")

    manifest_info = create_classification_manifest(
        processed_root,
        artifact_dir / "classification_split.csv",
        config.validation_fraction,
        config.seed,
    )
    rows = read_classification_manifest(manifest_info["manifest_path"], processed_root)
    mapping = category_mapping(rows)
    categories = [category for category, _ in sorted(mapping.items(), key=lambda item: item[1])]
    encoder = SharedDINOv2(
        DINOv2Config(image_size=(config.image_size, config.image_size)),
        artifact_root=artifact_dir,
        device=config.device,
    )

    classifier = None
    reference_artifact = None
    calibration: dict[str, Any] = {}
    classification_report: dict[str, Any] | None = None
    if len(categories) > 1:
        train_artifact = build_global_embedding_artifact(rows, "train", processed_root, encoder, mapping, config.batch_size)
        validation_artifact = build_global_embedding_artifact(rows, "validation", processed_root, encoder, mapping, config.batch_size)
        embedding_metadata = make_embedding_metadata(rows, mapping, manifest_info["manifest_path"], processed_root, encoder)
        save_embedding_artifacts(artifact_dir / "embeddings", train_artifact, validation_artifact, mapping, embedding_metadata)
        classifier_config = LinearClassifierConfig()
        classifier = fit_linear_classifier(train_artifact, mapping, classifier_config)
        save_linear_classifier(artifact_dir / "classifier", classifier, mapping, classifier_config, embedding_metadata)
        reference_artifact = train_artifact
        calibration = calibrate_open_set_thresholds(
            classifier,
            reference_artifact,
            validation_artifact,
            mapping,
            classification_confidence_threshold=config.classification_confidence_threshold,
        )
        save_calibration(artifact_dir / "calibration.json", calibration)
    else:
        train_artifact = None

    train_paths = {
        category: [processed_root / row["image_path"] for row in rows if row["category"] == category and row["split"] == "train"]
        for category in categories
    }
    banks: dict[str, FeatureBankResult] = {}
    for category in categories:
        paths = train_paths[category] or [processed_root / row["image_path"] for row in rows if row["category"] == category]
        bank = load_or_build_feature_bank(
            artifact_dir / "feature_banks",
            category,
            paths,
            processed_root,
            encoder,
            batch_size=config.batch_size,
            rebuild=config.force,
        )
        bank.metadata.setdefault("category", category)
        bank.path.parent.mkdir(parents=True, exist_ok=True)
        (bank.path.parent / "bank_metadata.json").write_text(json.dumps(bank.metadata, indent=2) + "\n", encoding="utf-8")
        banks[category] = bank

    image_thresholds, pixel_thresholds = _calibrate_anomaly_thresholds(rows, processed_root, banks, encoder, config)
    _write_json(artifact_dir / "image_thresholds.json", image_thresholds)
    _write_json(artifact_dir / "pixel_thresholds.json", pixel_thresholds)
    test_records = discover_mvtec_test_images(processed_root)
    inference_rows: list[dict[str, Any]] = []
    for item in test_records:
        from ..anomaly.pipeline import infer_image

        result = infer_image(
            item["image_path"],
            encoder,
            classifier,
            reference_artifact,
            mapping,
            calibration,
            artifact_dir / "feature_banks",
            image_thresholds,
            pixel_thresholds,
            k=config.k,
            classification_confidence_threshold=config.classification_confidence_threshold,
        )
        result.update(item)
        result["mask"] = _load_mask(item, config.image_size)
        result["anomaly_evaluated"] = result.get("anomaly_score") is not None
        inference_rows.append(result)

    classification_report, per_category = _classification_report(inference_rows, categories)
    anomaly_report = _anomaly_report(inference_rows)
    _write_json(report_dir / "classification_test_summary.json", classification_report)
    _write_json(report_dir / "anomaly_metrics.json", anomaly_report)
    _write_csv(
        report_dir / "per_category_metrics.csv",
        per_category,
        ["category", "support", "correct", "incorrect", "rejected", "accuracy", "rejection_rate"],
    )
    error_rows = [
        {
            "image_path": item["image_path"],
            "true_category": item["true_category"],
            "predicted_category": item.get("predicted_category"),
            "routed_category": item.get("category"),
            "confidence": item.get("confidence"),
            "routing_status": item.get("status"),
            "defect_type": item["defect_type"],
            "is_defect": item["is_defect"],
            "global_distance": item.get("nearest_global_distance"),
            "rejection_reason": item.get("reason", "") if item.get("rejected") else "",
        }
        for item in inference_rows
        if item.get("rejected") or item.get("category") != item["true_category"]
    ]
    _write_csv(
        report_dir / "error_analysis.csv",
        error_rows,
        ["image_path", "true_category", "predicted_category", "routed_category", "confidence", "routing_status", "defect_type", "is_defect", "global_distance", "rejection_reason"],
    )
    true_ids = [mapping[item["true_category"]] for item in inference_rows]
    pred_ids = [mapping[item["category"]] for item in inference_rows if not item.get("rejected")]
    accepted_true = [mapping[item["true_category"]] for item in inference_rows if not item.get("rejected")]
    _plot_confusion(
        confusion_matrix(accepted_true, pred_ids, labels=list(range(len(categories)))) if accepted_true else np.zeros((len(categories), len(categories)), dtype=int),
        categories,
        report_dir / "confusion_matrix.png",
        "Accepted product-category predictions",
    )
    unknown_pred = [mapping[item["category"]] if not item.get("rejected") else len(categories) for item in inference_rows]
    _plot_confusion(
        confusion_matrix(true_ids, unknown_pred, labels=list(range(len(categories) + 1))),
        categories + ["UNKNOWN"],
        report_dir / "confusion_matrix_with_unknown.png",
        "Product-category predictions including UNKNOWN",
    )

    for category in categories:
        examples = [item for item in inference_rows if item.get("category") == category and item.get("anomaly_evaluated")]
        if not examples:
            continue
        good_examples = [item for item in examples if not item["is_defect"]]
        defect_examples = [item for item in examples if item["is_defect"]]
        selected = (good_examples[:2] + defect_examples[:2])[:4]
        images = [np.asarray(Image.open(item["image_path"]).convert("RGB")) for item in selected]
        masks = [item["mask"] for item in selected]
        maps = np.stack([item["anomaly_map"] for item in selected])
        figure = plot_anomaly_examples(
            category,
            images,
            masks,
            maps,
            [float(item["anomaly_score"]) for item in selected],
            [bool(item["is_defect"]) for item in selected],
            config.k,
            examples_per_class=max(1, min(2, len(selected))),
        )
        save_figure(figure, report_dir / "heatmaps" / f"{category}.png")

    summary = {
        "report_version": "production-v1",
        "dataset_name": raw_root.name,
        "dataset_root": str(raw_root),
        "processed_root": str(processed_root),
        "report_dir": str(report_dir),
        "categories": categories,
        "classification_enabled": len(categories) > 1,
        "configuration": {
            "image_size": config.image_size,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "k": config.k,
            "classification_confidence_threshold": config.classification_confidence_threshold,
            "target_acceptance_rate": config.target_acceptance_rate,
        },
        "preparation": preparation,
        "manifest": manifest_info,
        "classification": classification_report,
        "anomaly": anomaly_report,
        "output_files": {
            "final_report": "final_report.md",
            "summary": "final_summary.json",
            "classification_summary": "classification_test_summary.json",
            "anomaly_metrics": "anomaly_metrics.json",
            "per_category": "per_category_metrics.csv",
            "errors": "error_analysis.csv",
            "confusion_matrix": "confusion_matrix.png",
            "confusion_matrix_with_unknown": "confusion_matrix_with_unknown.png",
            "dataset_distribution": "dataset_distribution.png",
            "heatmaps": "heatmaps/",
        },
    }
    _write_json(report_dir / "final_summary.json", summary)
    (report_dir / "final_report.md").write_text(_markdown_report(summary, report_dir), encoding="utf-8")
    return summary
