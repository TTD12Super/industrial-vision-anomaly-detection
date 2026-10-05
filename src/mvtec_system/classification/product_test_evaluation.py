"""Full official MVTec test-split evaluation for product-category routing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .calibration import nearest_global_distances
from .classifier import predict_linear_classifier
from ..core.dinov2 import SharedDINOv2, image_tensor
from .router import DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD, routing_decision


REPORT_VERSION = "v1"
UNKNOWN_LABEL = "unknown_product_category"


def product_category_from_mvtec_path(image_path: str | Path) -> str:
    """Derive the product category from ``<category>/test/<defect>/image``."""

    path = Path(image_path)
    parts = path.parts
    try:
        test_index = parts.index("test")
    except ValueError as error:
        raise ValueError(f"MVTec test path does not contain a test directory: {path}") from error
    if test_index == 0:
        raise ValueError(f"MVTec test path has no category before test: {path}")
    return parts[test_index - 1]


def discover_mvtec_test_images(dataset_root: str | Path) -> list[dict[str, Any]]:
    """List every image in every category's official processed test split."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed MVTec root does not exist: {root}")
    records: list[dict[str, Any]] = []
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        test_dir = category_dir / "test"
        if not test_dir.is_dir():
            continue
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            for image_path in sorted(
                path
                for path in defect_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ):
                records.append(
                    {
                        "image_path": str(image_path),
                        "true_category": product_category_from_mvtec_path(image_path),
                        "defect_type": defect_type,
                        "is_defect": defect_type != "good",
                    }
                )
    if not records:
        raise ValueError(f"No processed MVTec test images found under {root}")
    return records


def _batched(values: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _classify_test_images(
    records: list[dict[str, Any]],
    encoder: SharedDINOv2,
    classifier: dict[str, Any],
    reference_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    calibration: dict[str, Any],
    batch_size: int,
    classification_confidence_threshold: float | None,
) -> list[dict[str, Any]]:
    """Encode test images and apply the exact product routing rule without banks."""

    global_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in _batched(records, batch_size):
            images = torch.stack(
                [image_tensor(item["image_path"], encoder.config) for item in batch]
            )
            features = encoder.forward(images)
            global_batches.append(features.global_embedding.detach().cpu())
    global_embeddings = torch.cat(global_batches, dim=0)
    reference_embeddings = reference_artifact.get("embeddings")
    if not isinstance(reference_embeddings, torch.Tensor):
        raise ValueError("Reference artifact must contain tensor embeddings.")
    logits, probabilities = predict_linear_classifier(classifier, global_embeddings)
    distances = nearest_global_distances(global_embeddings, reference_embeddings)
    inverse_mapping = {index: category for category, index in class_mapping.items()}
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        predicted_index = int(logits[index].argmax().item())
        predicted_category = inverse_mapping[predicted_index]
        confidence = float(probabilities[index].max().item())
        distance = float(distances[index].item())
        decision = routing_decision(
            predicted_category,
            confidence,
            distance,
            calibration,
            classification_confidence_threshold,
        )
        predictions.append(
            {
                **item,
                "predicted_category": predicted_category,
                "routed_category": decision["category"],
                "confidence": confidence,
                "global_distance": distance,
                "global_distance_threshold": decision["global_distance_threshold"],
                "global_distance_ood": bool(decision["global_distance_ood"]),
                "routing_status": decision["status"],
                "rejected": bool(decision["rejected"]),
                "rejection_reason": decision["reason"] if decision["rejected"] else "",
            }
        )
    return predictions


def _split_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    rejected = sum(item["rejected"] for item in records)
    global_distance_ood = sum(item["global_distance_ood"] for item in records)
    correct = sum(
        not item["rejected"] and item["routed_category"] == item["true_category"]
        for item in records
    )
    accepted = total - rejected
    return {
        "total_images": total,
        "correct_predictions": correct,
        "accepted_known_predictions": accepted,
        "unknown_rejected_predictions": rejected,
        "accuracy": correct / total if total else 0.0,
        "rejection_rate": rejected / total if total else 0.0,
        "accepted_accuracy": correct / accepted if accepted else 0.0,
        "global_distance_ood_count": global_distance_ood,
        "global_distance_ood_rate": global_distance_ood / total if total else 0.0,
    }


def _per_category_metrics(
    records: list[dict[str, Any]], categories: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for category in categories:
        category_records = [item for item in records if item["true_category"] == category]
        support = len(category_records)
        rejected = sum(item["rejected"] for item in category_records)
        correct = sum(
            not item["rejected"] and item["routed_category"] == category
            for item in category_records
        )
        accepted_incorrect = support - rejected - correct
        rows.append(
            {
                "category": category,
                "support": support,
                "correct": correct,
                "incorrect": accepted_incorrect,
                "total_errors": accepted_incorrect + rejected,
                "rejected": rejected,
                "accuracy": correct / support if support else 0.0,
                "rejection_rate": rejected / support if support else 0.0,
            }
        )
    return rows


def _error_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "image_path": item["image_path"],
            "true_category": item["true_category"],
            "predicted_category": item["predicted_category"],
            "routed_category": item["routed_category"],
            "confidence": item["confidence"],
            "routing_status": item["routing_status"],
            "defect_type": item["defect_type"],
            "is_defect": item["is_defect"],
            "global_distance": item["global_distance"],
            "global_distance_threshold": item["global_distance_threshold"],
            "global_distance_ood": item["global_distance_ood"],
            "rejection_reason": item["rejection_reason"],
        }
        for item in records
        if item["rejected"] or item["routed_category"] != item["true_category"]
    ]


def _global_distance_defect_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export accepted images whose global embedding is far from normal references."""

    return [
        {
            "image_path": item["image_path"],
            "true_category": item["true_category"],
            "predicted_category": item["predicted_category"],
            "confidence": item["confidence"],
            "global_distance": item["global_distance"],
            "global_distance_threshold": item["global_distance_threshold"],
            "global_distance_ood": item["global_distance_ood"],
            "defect_type": item["defect_type"],
            "is_defect": item["is_defect"],
            "routing_status": item["routing_status"],
        }
        for item in records
        if item["global_distance_ood"]
    ]


def _save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    figure_size = max(8.0, len(labels) * 0.55)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
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
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=7,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(summary: dict[str, Any], output_dir: Path) -> str:
    overall = summary["overall"]
    good = summary["good_test"]
    defect = summary["defect_test"]
    lines = [
        "# Product-category recognition on the full MVTec test split",
        "",
        "This report evaluates the frozen classifier and frozen routing thresholds. "
        "The official test split was used only for evaluation; no weights, banks, "
        "backbone, or anomaly thresholds were changed.",
        "",
        "## Routing",
        "",
        f"- Minimum configurable confidence threshold: `{summary['routing']['minimum_confidence_threshold']:.2f}`",
        "- Existing calibrated confidence rejection: `disabled`",
        f"- Global-distance defect threshold (90th percentile): `{summary['routing']['global_distance_threshold']}`",
        "- Rule: confidence below the minimum returns `unknown_product_category`; accepted predictions continue to bank/KNN routing. Global distance above the 90th-percentile threshold marks a defect suspicion but does not reject the product category.",
        "",
        "## Overall",
        "",
        f"- Test images: **{overall['total_images']}**",
        f"- Accepted known predictions: **{overall['accepted_known_predictions']}**",
        f"- Unknown/rejected predictions: **{overall['unknown_rejected_predictions']} ({overall['rejection_rate']:.2%})**",
        f"- Global-distance defect candidates: **{summary['routing']['global_distance_ood_count']} ({summary['routing']['global_distance_ood_rate']:.2%})**",
        f"- Top-1 accuracy over all test images: **{overall['accuracy']:.2%}**",
        f"- Top-1 accuracy over accepted predictions: **{overall['accepted_accuracy']:.2%}**",
        f"- Macro precision: **{overall['macro_precision']:.4f}**",
        f"- Macro recall: **{overall['macro_recall']:.4f}**",
        f"- Macro F1: **{overall['macro_f1']:.4f}**",
        "",
        "## Defect robustness",
        "",
        "Does the classifier still identify the correct product category when the product contains a defect?",
        "",
        f"- Good test accuracy: **{good['accuracy']:.2%}**; rejection rate: **{good['rejection_rate']:.2%}**",
        f"- Defect test accuracy: **{defect['accuracy']:.2%}**; rejection rate: **{defect['rejection_rate']:.2%}**",
        f"- Global-distance defect flag: good **{good['global_distance_ood_count']} ({good['global_distance_ood_rate']:.2%})**, defect **{defect['global_distance_ood_count']} ({defect['global_distance_ood_rate']:.2%})**",
        f"- Accuracy change (good minus defect): **{summary['defect_robustness']['accuracy_drop_good_minus_defect']:.2%}**",
        "",
        "Categories with the largest defect-image accuracy drops:",
        "",
    ]
    for row in summary["defect_robustness"]["largest_accuracy_drops"]:
        lines.append(
            f"- `{row['category']}`: good `{row['good_accuracy']:.2%}`, defect `{row['defect_accuracy']:.2%}`, drop `{row['accuracy_drop_good_minus_defect']:.2%}`"
        )
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            f"- Misclassified or rejected images saved in `{output_dir / 'error_analysis.csv'}`.",
            f"- Global-distance defect candidates saved in `{output_dir / 'global_distance_defect_candidates.csv'}`.",
            f"- Accepted confusion matrix: `{output_dir / 'confusion_matrix.png'}`.",
            f"- Confusion matrix with UNKNOWN: `{output_dir / 'confusion_matrix_with_unknown.png'}`.",
            "",
            "Categories with the most total classification failures (incorrect + UNKNOWN):",
            "",
        ]
    )
    for row in summary["categories_with_most_errors"]:
        lines.append(
            f"- `{row['category']}`: {row['total_errors']} total failures "
            f"({row['incorrect']} incorrect, {row['rejected']} UNKNOWN)"
        )
    lines.extend(["", "Categories with the most UNKNOWN rejections:", ""])
    for row in summary["categories_with_most_unknown_rejections"]:
        lines.append(f"- `{row['category']}`: {row['rejected']} rejected images")
    return "\n".join(lines) + "\n"


def evaluate_product_classification_test_split(
    dataset_root: str | Path,
    output_dir: str | Path,
    encoder: SharedDINOv2,
    classifier: dict[str, Any],
    reference_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    calibration: dict[str, Any],
    batch_size: int = 16,
    classification_confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate all official MVTec test images and write the versioned report."""

    records = discover_mvtec_test_images(dataset_root)
    categories = [
        category for category, _ in sorted(class_mapping.items(), key=lambda item: item[1])
    ]
    discovered_categories = sorted({item["true_category"] for item in records})
    if discovered_categories != sorted(categories):
        raise ValueError(
            "Dataset categories do not match classifier mapping: "
            f"dataset={discovered_categories}, mapping={sorted(categories)}"
        )
    predictions = _classify_test_images(
        records,
        encoder,
        classifier,
        reference_artifact,
        class_mapping,
        calibration,
        batch_size,
        classification_confidence_threshold,
    )
    category_rows = _per_category_metrics(predictions, categories)
    all_true = [class_mapping[item["true_category"]] for item in predictions]
    all_pred = [
        class_mapping[item["routed_category"]] if not item["rejected"] else -1
        for item in predictions
    ]
    class_ids = list(range(len(categories)))
    overall_split = _split_metrics(predictions)
    overall_split.update(
        {
            "top1_accuracy_all": overall_split["accuracy"],
            "top1_accuracy_accepted": overall_split["accepted_accuracy"],
            "macro_precision": float(
                precision_score(all_true, all_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(all_true, all_pred, labels=class_ids, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(all_true, all_pred, labels=class_ids, average="macro", zero_division=0)
            ),
        }
    )
    good = _split_metrics([item for item in predictions if not item["is_defect"]])
    defect = _split_metrics([item for item in predictions if item["is_defect"]])
    error_rows = _error_rows(predictions)
    global_distance_defect_rows = _global_distance_defect_rows(predictions)
    defect_drops = []
    for category in categories:
        good_row = _split_metrics(
            [item for item in predictions if not item["is_defect"] and item["true_category"] == category]
        )
        defect_row = _split_metrics(
            [item for item in predictions if item["is_defect"] and item["true_category"] == category]
        )
        defect_drops.append(
            {
                "category": category,
                "good_accuracy": good_row["accuracy"],
                "defect_accuracy": defect_row["accuracy"],
                "accuracy_drop_good_minus_defect": good_row["accuracy"] - defect_row["accuracy"],
            }
        )
    defect_drops.sort(key=lambda row: (-row["accuracy_drop_good_minus_defect"], row["category"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_records = [item for item in predictions if not item["rejected"]]
    accepted_true = [class_mapping[item["true_category"]] for item in accepted_records]
    accepted_pred = [class_mapping[item["routed_category"]] for item in accepted_records]
    accepted_matrix = confusion_matrix(accepted_true, accepted_pred, labels=class_ids)
    unknown_labels = categories + [UNKNOWN_LABEL]
    unknown_pred = [
        class_mapping[item["routed_category"]] if not item["rejected"] else len(categories)
        for item in predictions
    ]
    unknown_matrix = confusion_matrix(all_true, unknown_pred, labels=list(range(len(categories) + 1)))
    _save_confusion_matrix(
        accepted_matrix,
        categories,
        output / "confusion_matrix.png",
        "Accepted product-category predictions",
    )
    _save_confusion_matrix(
        unknown_matrix,
        unknown_labels,
        output / "confusion_matrix_with_unknown.png",
        "Product-category predictions including UNKNOWN",
    )
    _write_csv(
        output / "per_category_metrics.csv",
        category_rows,
        [
            "category",
            "support",
            "correct",
            "incorrect",
            "total_errors",
            "rejected",
            "accuracy",
            "rejection_rate",
        ],
    )
    _write_csv(
        output / "error_analysis.csv",
        error_rows,
        [
            "image_path",
            "true_category",
            "predicted_category",
            "routed_category",
            "confidence",
            "routing_status",
            "defect_type",
            "is_defect",
            "global_distance",
            "global_distance_threshold",
            "global_distance_ood",
            "rejection_reason",
        ],
    )
    _write_csv(
        output / "global_distance_defect_candidates.csv",
        global_distance_defect_rows,
        [
            "image_path",
            "true_category",
            "predicted_category",
            "confidence",
            "global_distance",
            "global_distance_threshold",
            "global_distance_ood",
            "defect_type",
            "is_defect",
            "routing_status",
        ],
    )
    minimum_threshold = (
        float(
            calibration.get(
                "classification_confidence_threshold",
                DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
            )
        )
        if classification_confidence_threshold is None
        else float(classification_confidence_threshold)
    )
    summary = {
        "report_version": REPORT_VERSION,
        "dataset_root": str(Path(dataset_root)),
        "evaluation_only": True,
        "classifier_and_backbone_unchanged": True,
        "total_test_images": len(predictions),
        "accepted_known_predictions": overall_split["accepted_known_predictions"],
        "unknown_rejected_predictions": overall_split["unknown_rejected_predictions"],
        "rejection_rate": overall_split["rejection_rate"],
        "top1_accuracy_all": overall_split["top1_accuracy_all"],
        "top1_accuracy_accepted": overall_split["top1_accuracy_accepted"],
        "macro_precision": overall_split["macro_precision"],
        "macro_recall": overall_split["macro_recall"],
        "macro_f1": overall_split["macro_f1"],
        "routing": {
            "minimum_confidence_threshold": minimum_threshold,
            "calibrated_confidence_threshold": calibration.get("confidence_threshold"),
            "global_distance_threshold": calibration.get(
                "global_distance_defect_threshold",
                calibration.get("nearest_global_distance_threshold"),
            ),
            "global_distance_percentile": calibration.get(
                "global_distance_defect_percentile", 0.90
            ),
            "calibrated_confidence_rejection_enabled": False,
            "distance_rejection_enabled": False,
            "global_distance_marks_defect": True,
            "global_distance_ood_count": sum(
                item["global_distance_ood"] for item in predictions
            ),
            "global_distance_ood_rate": (
                sum(item["global_distance_ood"] for item in predictions) / len(predictions)
                if predictions
                else 0.0
            ),
            "banks_loaded": False,
            "knn_run": False,
        },
        "overall": overall_split,
        "good_test": good,
        "defect_test": defect,
        "per_category": category_rows,
        "defect_robustness": {
            "question": "Does the classifier still identify the correct product category when the product contains a defect?",
            "good_accuracy": good["accuracy"],
            "defect_accuracy": defect["accuracy"],
            "accuracy_drop_good_minus_defect": good["accuracy"] - defect["accuracy"],
            "largest_accuracy_drops": defect_drops[:5],
        },
        "categories_with_most_errors": sorted(
            category_rows, key=lambda row: (-row["total_errors"], row["category"])
        )[:5],
        "categories_with_most_unknown_rejections": sorted(
            category_rows, key=lambda row: (-row["rejected"], row["category"])
        )[:5],
        "output_files": {
            "summary": "classification_test_summary.json",
            "per_category": "per_category_metrics.csv",
            "errors": "error_analysis.csv",
            "global_distance_defect_candidates": "global_distance_defect_candidates.csv",
            "confusion_matrix": "confusion_matrix.png",
            "confusion_matrix_with_unknown": "confusion_matrix_with_unknown.png",
            "report": "product_classification_test_report.md",
        },
    }
    (output / "classification_test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "product_classification_test_report.md").write_text(
        _markdown_report(summary, output), encoding="utf-8"
    )
    return summary
