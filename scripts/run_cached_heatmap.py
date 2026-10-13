"""Evaluate cached PatchCore heatmaps on a deterministic test subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from mvtec_system.anomaly.detectors import PatchCoreDetector
from mvtec_system.classification.product_test_evaluation import discover_mvtec_test_images
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor
from mvtec_system.evaluation.performance import synchronize
from mvtec_system.anomaly.multi_layer import MultiLayerConfig


def _ground_truth_mask(item: dict, image_size: int) -> np.ndarray:
    if not item["is_defect"]:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    image = Path(item["image_path"])
    mask_path = image.parents[2] / "ground_truth" / item["defect_type"] / f"{image.stem}_mask.png"
    if not mask_path.is_file():
        return np.zeros((image_size, image_size), dtype=np.uint8)
    from PIL import Image

    with Image.open(mask_path) as mask:
        return np.asarray(mask.convert("L").resize((image_size, image_size))) > 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/mvtec_ad_processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/production/heatmap"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.test_stride < 1:
        raise SystemExit("--test-stride must be positive")
    processed_root = args.processed_root.resolve()
    artifact_root = (args.artifact_root / processed_root.name.removesuffix("_processed")).resolve()
    records = discover_mvtec_test_images(processed_root)[:: args.test_stride]
    if not records:
        raise SystemExit("No test images selected")

    payload = torch.load(
        artifact_root / "detectors" / "patchcore" / records[0]["true_category"] / "coreset.pt",
        map_location="cpu",
        weights_only=True,
    )
    detector_config = MultiLayerConfig(**payload["config"])
    device = torch.device(args.device)
    encoder = SharedDINOv2(
        DINOv2Config(
            image_size=(args.image_size, args.image_size),
            feature_layers=detector_config.layers,
        ),
        artifact_root=artifact_root,
        device=device,
    )
    detectors = {}
    for category in sorted({item["true_category"] for item in records}):
        detectors[category] = PatchCoreDetector(detector_config, k=1).load(
            artifact_root / "detectors" / "patchcore" / category
        )

    labels: list[int] = []
    image_scores: list[float] = []
    heatmap_scores: list[float] = []
    pixel_labels: list[int] = []
    latencies: list[float] = []
    for index, item in enumerate(records, start=1):
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            features = encoder.forward(
                torch.stack([image_tensor(item["image_path"], encoder.config)])
            )
        result = detectors[item["true_category"]].predict(features)
        synchronize(device)
        latencies.append(time.perf_counter() - started)
        truth = _ground_truth_mask(item, args.image_size)
        labels.append(int(item["is_defect"]))
        image_scores.append(float(result.image_score))
        heatmap_scores.extend(result.anomaly_map.reshape(-1).tolist())
        pixel_labels.extend((truth > 0).astype(np.int64).reshape(-1).tolist())
        if index % 25 == 0 or index == len(records):
            print(f"processed {index}/{len(records)}", flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.output_root.resolve() / "cached" / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "report_version": "cached-patchcore-heatmap-v1",
        "mode": "patchcore_heatmap_only",
        "padim_run": False,
        "processed_root": str(processed_root),
        "artifact_root": str(artifact_root),
        "test_stride": args.test_stride,
        "test_images": len(records),
        "device": str(device),
        "image_ranking": {
            "auroc": float(roc_auc_score(labels, image_scores)),
            "average_precision": float(average_precision_score(labels, image_scores)),
        },
        "heatmap_ranking": {
            "auroc": float(roc_auc_score(pixel_labels, heatmap_scores)),
            "average_precision": float(average_precision_score(pixel_labels, heatmap_scores)),
        },
        "latency": {
            "mean_seconds": float(np.mean(latencies)),
            "median_seconds": float(np.median(latencies)),
            "p95_seconds": float(np.percentile(latencies, 95)),
            "images_per_second": float(len(latencies) / sum(latencies)),
        },
        "protocol": "Cached PatchCore n=3; category selected from the MVTec test path; no classifier or PaDiM run.",
    }
    (report_dir / "heatmap_only.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report_dir / "heatmap_only.md").write_text(
        "\n".join(
            [
                "# Cached PatchCore heatmap-only evaluation",
                "",
                f"- Test images: **{summary['test_images']}** (every {args.test_stride}th sorted image)",
                f"- Heatmap pixel AUROC / AP: **{summary['heatmap_ranking']['auroc']:.6f} / {summary['heatmap_ranking']['average_precision']:.6f}**",
                f"- Image-score AUROC / AP: **{summary['image_ranking']['auroc']:.6f} / {summary['image_ranking']['average_precision']:.6f}**",
                f"- Mean / median / p95 latency: **{summary['latency']['mean_seconds']:.4f} / {summary['latency']['median_seconds']:.4f} / {summary['latency']['p95_seconds']:.4f} s/image**",
                "- PaDiM and binary-mask evaluation: **not run**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report_dir": str(report_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
