"""Compare fixed and validation-calibrated cached PatchCore masks."""

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
    parser.add_argument("--fixed-threshold", type=float, default=0.7)
    parser.add_argument("--small-threshold", type=float, default=0.4)
    parser.add_argument("--large-threshold", type=float, default=0.8)
    parser.add_argument("--large-area", type=int, default=256)
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

    fixed_counts = {key: 0 for key in ("true_positive", "false_positive", "true_negative", "false_negative")}
    adaptive_counts = dict(fixed_counts)
    calibrated_counts = dict(fixed_counts)
    raw_scores: list[float] = []
    labels: list[int] = []
    raw_pixel_scores: list[float] = []
    pixel_labels: list[int] = []
    for item, heatmap in test_maps:
        truth = _mask(item, args.image_size)
        fixed = _count(heatmap > args.fixed_threshold, truth)
        calibrated = _count(
            refine_binary_mask(heatmap > thresholds[item["true_category"]]), truth
        )
        adaptive = _count(
            _adaptive_mask(
                heatmap,
                args.small_threshold,
                args.large_threshold,
                args.large_area,
            ),
            truth,
        )
        for key in fixed_counts:
            fixed_counts[key] += fixed[key]
            adaptive_counts[key] += adaptive[key]
            calibrated_counts[key] += calibrated[key]
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
        "fixed": {
            "threshold": args.fixed_threshold,
            "mask": _metrics(fixed_counts),
        },
        "adaptive": {
            "small_threshold": args.small_threshold,
            "large_threshold": args.large_threshold,
            "large_area": args.large_area,
            "mask": _metrics(adaptive_counts),
        },
        "calibrated": {
            "good_quantile": args.good_quantile,
            "thresholds_by_category": thresholds,
            "mask": _metrics(calibrated_counts),
        },
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
        "protocol": "Both masks use cached PatchCore n=3; calibrated thresholds use GOOD validation pixels only; calibrated mask applies 3x3 opening and min component area 9.",
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.output_root.resolve() / "refinement" / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fixed = result["fixed"]["mask"]
    adaptive = result["adaptive"]["mask"]
    calibrated = result["calibrated"]["mask"]
    (report_dir / "comparison.md").write_text(
        "\n".join(
            [
                "# PatchCore heatmap refinement comparison",
                "",
                f"- Test images: **{len(test_records)}**; validation GOOD images: **{len(validation_records)}**",
                "",
                "| Mask | Precision | Recall | F1 | IoU | FPR |",
                "|---|---:|---:|---:|---:|---:|",
                f"| Fixed > {args.fixed_threshold:.2f} | {fixed['precision']:.6f} | {fixed['recall']:.6f} | {fixed['f1']:.6f} | {fixed['iou']:.6f} | {fixed['false_positive_rate']:.6f} |",
                f"| Adaptive small > {args.small_threshold:.2f}, large > {args.large_threshold:.2f} | {adaptive['precision']:.6f} | {adaptive['recall']:.6f} | {adaptive['f1']:.6f} | {adaptive['iou']:.6f} | {adaptive['false_positive_rate']:.6f} |",
                f"| Calibrated q={args.good_quantile:.3f} + refine | {calibrated['precision']:.6f} | {calibrated['recall']:.6f} | {calibrated['f1']:.6f} | {calibrated['iou']:.6f} | {calibrated['false_positive_rate']:.6f} |",
                "",
                f"- Adaptive F1 delta vs fixed: **{adaptive['f1'] - fixed['f1']:+.6f}**",
                f"- Adaptive IoU delta vs fixed: **{adaptive['iou'] - fixed['iou']:+.6f}**",
                f"- F1 delta: **{calibrated['f1'] - fixed['f1']:+.6f}**",
                f"- IoU delta: **{calibrated['iou'] - fixed['iou']:+.6f}**",
                f"- Threshold range: **{min(thresholds.values()):.6f} – {max(thresholds.values()):.6f}**",
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
