"""Evaluate ten cached PatchCore heatmap mask methods and keep the best."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import cv2
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from mvtec_system.anomaly.detectors import PatchCoreDetector
from mvtec_system.anomaly.multi_layer import MultiLayerConfig
from mvtec_system.anomaly.refinement import refine_binary_mask
from mvtec_system.classification.product_test_evaluation import discover_mvtec_test_images
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor
from mvtec_system.evaluation.performance import synchronize


def _mask(item: dict, image_size: int) -> np.ndarray:
    if not item["is_defect"]:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    image = Path(item["image_path"])
    mask_path = image.parents[2] / "ground_truth" / item["defect_type"] / f"{image.stem}_mask.png"
    if not mask_path.is_file():
        return np.zeros((image_size, image_size), dtype=np.uint8)
    from PIL import Image

    with Image.open(mask_path) as mask:
        return np.asarray(mask.convert("L").resize((image_size, image_size))) > 0


def _count(mask: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    values = np.asarray(mask, dtype=bool)
    labels = np.asarray(truth, dtype=bool)
    return {
        "true_positive": int(np.logical_and(values, labels).sum()),
        "false_positive": int(np.logical_and(values, ~labels).sum()),
        "true_negative": int(np.logical_and(~values, ~labels).sum()),
        "false_negative": int(np.logical_and(~values, labels).sum()),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _adaptive_mask(
    heatmap: np.ndarray,
    small_threshold: float,
    large_threshold: float,
    large_area: int,
) -> np.ndarray:
    candidates = np.asarray(heatmap > small_threshold, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )
    result = np.zeros_like(candidates, dtype=bool)
    for component in range(1, component_count):
        region = labels == component
        threshold = large_threshold if stats[component, cv2.CC_STAT_AREA] >= large_area else small_threshold
        result[region] = np.asarray(heatmap[region] > threshold, dtype=bool)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/mvtec_ad_processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/production/mvtec_ad/classification_split.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/production/heatmap"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument("--good-quantile", type=float, default=0.995)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.test_stride < 1 or not 0.0 < args.good_quantile < 1.0:
        raise SystemExit("test-stride must be positive and good-quantile must be in (0, 1)")
    processed_root = args.processed_root.resolve()
    artifact_root = (args.artifact_root / processed_root.name.removesuffix("_processed")).resolve()
    test_records = discover_mvtec_test_images(processed_root)[:: args.test_stride]
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        validation_records = [
            {
                "image_path": str(processed_root / row["image_path"]),
                "true_category": row["category"],
                "is_defect": False,
            }
            for row in csv.DictReader(handle)
            if row["split"] == "validation"
        ]
    if not validation_records or not test_records:
        raise SystemExit("Validation or test subset is empty")

    first_category = test_records[0]["true_category"]
    payload = torch.load(
        artifact_root / "detectors" / "patchcore" / first_category / "coreset.pt",
        map_location="cpu",
        weights_only=True,
    )
    detector_config = MultiLayerConfig(**payload["config"])
    device = torch.device(args.device)
    encoder = SharedDINOv2(
        DINOv2Config(image_size=(args.image_size, args.image_size), feature_layers=detector_config.layers),
        artifact_root=artifact_root,
        device=device,
    )
    categories = sorted({item["true_category"] for item in test_records + validation_records})
    detectors = {
        category: PatchCoreDetector(detector_config, k=1).load(
            artifact_root / "detectors" / "patchcore" / category
        )
        for category in categories
    }

    def infer(records: list[dict]) -> list[tuple[dict, np.ndarray]]:
        outputs = []
        for index, item in enumerate(records, start=1):
            with torch.inference_mode():
                features = encoder.forward(
                    torch.stack([image_tensor(item["image_path"], encoder.config)])
                )
            outputs.append((item, detectors[item["true_category"]].predict(features).anomaly_map))
            if index % 50 == 0 or index == len(records):
                print(f"processed {index}/{len(records)}", flush=True)
        return outputs

    validation_maps = infer(validation_records)
    thresholds = {
        category: float(
            np.quantile(
                np.concatenate([heatmap.reshape(-1) for item, heatmap in validation_maps if item["true_category"] == category]),
                args.good_quantile,
            )
        )
        for category in categories
    }
    synchronize(device)
    started = time.perf_counter()
    test_maps = infer(test_records)
    synchronize(device)
    test_seconds = time.perf_counter() - started

    count_keys = ("true_positive", "false_positive", "true_negative", "false_negative")
    method_names = (
        "fixed_0.50",
        "fixed_0.60",
        "fixed_0.70",
        "fixed_0.80",
        "fixed_0.70_open3_area9",
        "fixed_0.70_area25",
        "category_q995",
        "category_q995_open3_area9",
        "adaptive_0.40_0.80_area256",
        "top1pct_open3_area9",
    )
    method_counts = {name: {key: 0 for key in count_keys} for name in method_names}
    raw_scores: list[float] = []
    labels: list[int] = []
    raw_pixel_scores: list[float] = []
    pixel_labels: list[int] = []
    for item, heatmap in test_maps:
        truth = _mask(item, args.image_size)
        masks = {
            "fixed_0.50": heatmap > 0.5,
            "fixed_0.60": heatmap > 0.6,
            "fixed_0.70": heatmap > 0.7,
            "fixed_0.80": heatmap > 0.8,
            "fixed_0.70_open3_area9": refine_binary_mask(heatmap > 0.7, 9, 3),
            "fixed_0.70_area25": refine_binary_mask(heatmap > 0.7, 25, 1),
            "category_q995": heatmap > thresholds[item["true_category"]],
            "category_q995_open3_area9": refine_binary_mask(
                heatmap > thresholds[item["true_category"]], 9, 3
            ),
            "adaptive_0.40_0.80_area256": _adaptive_mask(
                heatmap,
                0.4,
                0.8,
                256,
            ),
            "top1pct_open3_area9": refine_binary_mask(
                heatmap > np.quantile(heatmap, 0.99), 9, 3
            ),
        }
        for name in method_names:
            counts = _count(masks[name], truth)
            for key in count_keys:
                method_counts[name][key] += counts[key]
        raw_scores.append(float(heatmap.max()))
        labels.append(int(item["is_defect"]))
        raw_pixel_scores.extend(heatmap.reshape(-1).tolist())
        pixel_labels.extend((truth > 0).astype(np.int64).reshape(-1).tolist())

    result = {
        "report_version": "heatmap-refinement-comparison-v1",
        "test_images": len(test_records),
        "test_stride": args.test_stride,
        "validation_images": len(validation_records),
        "device": str(device),
        "methods": {name: _metrics(method_counts[name]) for name in method_names},
        "raw_heatmap": {
            "pixel_auroc": float(roc_auc_score(pixel_labels, raw_pixel_scores)),
            "pixel_average_precision": float(average_precision_score(pixel_labels, raw_pixel_scores)),
            "image_auroc": float(roc_auc_score(labels, raw_scores)),
            "image_average_precision": float(average_precision_score(labels, raw_scores)),
        },
        "test_latency": {
            "total_seconds": test_seconds,
            "mean_seconds": test_seconds / len(test_records),
        },
        "thresholds_by_category": thresholds,
        "selection": {
            "primary_metric": "f1",
            "tie_breaker": "iou",
            "best_method": max(
                method_names,
                key=lambda name: (
                    _metrics(method_counts[name])["f1"],
                    _metrics(method_counts[name])["iou"],
                ),
            ),
            "warning": "Best method is selected on the 1/5 test subset for exploration; validate on a held-out split before production.",
        },
        "protocol": "All methods reuse cached PatchCore n=3 maps; category thresholds use GOOD validation pixels only; morphology uses 3x3 opening and connected-component filtering.",
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.output_root.resolve() / "refinement" / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    methods = result["methods"]
    best_name = result["selection"]["best_method"]
    (report_dir / "comparison.md").write_text(
        "\n".join(
            [
                "# PatchCore heatmap refinement comparison",
                "",
                f"- Test images: **{len(test_records)}**; validation GOOD images: **{len(validation_records)}**",
                "",
                "| Method | Precision | Recall | F1 | IoU | FPR |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {name} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['iou']:.6f} | {metrics['false_positive_rate']:.6f} |"
                    for name, metrics in methods.items()
                ],
                "",
                f"- Best method by F1/IoU: **{best_name}**",
                f"- Threshold range: **{min(thresholds.values()):.6f} – {max(thresholds.values()):.6f}**",
                "- Best method selection is exploratory because it uses this 1/5 test subset.",
                "- PaDiM: **not run**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report_dir": str(report_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
