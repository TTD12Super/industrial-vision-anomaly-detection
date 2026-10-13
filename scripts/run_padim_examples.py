"""Run the persisted PaDiM detectors on a few defect images and save visual results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from mvtec_system.anomaly.detectors import PaDiMDetector
from mvtec_system.anomaly.multi_layer import MultiLayerConfig
from mvtec_system.anomaly.refinement import refine_binary_mask
from mvtec_system.classification.product_test_evaluation import discover_mvtec_test_images
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor


def _ground_truth(item: dict, image_size: int) -> np.ndarray:
    image = Path(item["image_path"])
    mask_path = image.parents[2] / "ground_truth" / item["defect_type"] / f"{image.stem}_mask.png"
    with Image.open(mask_path) as mask:
        return np.asarray(mask.convert("L").resize((image_size, image_size))) > 0


def _metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "predicted_positive_fraction": float(predicted.mean()),
        "ground_truth_positive_fraction": float(truth.mean()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/mvtec_ad_processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/production/padim_examples"))
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "capsule", "carpet", "transistor", "zipper"])
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    processed_root = args.processed_root.resolve()
    artifact_root = args.artifact_root.resolve()
    records_by_category: dict[str, dict] = {}
    for item in discover_mvtec_test_images(processed_root):
        if item["true_category"] in args.categories and item["is_defect"]:
            records_by_category.setdefault(item["true_category"], item)
    missing = [category for category in args.categories if category not in records_by_category]
    if missing:
        raise SystemExit(f"No defect sample found for: {', '.join(missing)}")

    first_payload = torch.load(
        artifact_root / "detectors" / "padim" / args.categories[0] / "statistics.pt",
        map_location="cpu",
        weights_only=True,
    )
    detector_config = MultiLayerConfig(**first_payload["config"])
    image_size = tuple(first_payload["image_size"])
    encoder = SharedDINOv2(
        DINOv2Config(image_size=image_size, feature_layers=detector_config.layers),
        artifact_root=artifact_root,
        device=torch.device(args.device),
    )
    thresholds_payload = json.loads(
        (artifact_root / "thresholds" / "exact_first" / "pixel_thresholds.json").read_text(encoding="utf-8")
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    figure, axes = plt.subplots(len(args.categories), 5, figsize=(17, 3.4 * len(args.categories)), squeeze=False)
    for row, category in enumerate(args.categories):
        item = records_by_category[category]
        detector = PaDiMDetector(detector_config).load(
            artifact_root / "detectors" / "padim" / category
        )
        with Image.open(item["image_path"]) as source:
            image = np.asarray(source.convert("RGB"))
        truth = _ground_truth(item, image_size[0])
        with torch.inference_mode():
            features = encoder.forward(
                torch.stack([image_tensor(item["image_path"], encoder.config)])
            )
        anomaly_map = detector.predict(features).anomaly_map
        threshold = float(thresholds_payload["categories"][category]["threshold"])
        binary_mask = refine_binary_mask(anomaly_map >= threshold, 9, 3)
        result = {
            "category": category,
            "defect_type": item["defect_type"],
            "image_path": item["image_path"],
            "padim_threshold": threshold,
            "padim_image_score": float(anomaly_map.max()),
            **_metrics(binary_mask, truth),
        }
        results.append(result)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{category}\n{item['defect_type']}")
        axes[row, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(anomaly_map, cmap="inferno")
        axes[row, 2].set_title(f"PaDiM map\nmax={anomaly_map.max():.3f}")
        axes[row, 3].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"Binary mask\nthreshold={threshold:.3f}")
        axes[row, 4].imshow(image)
        axes[row, 4].imshow(binary_mask, cmap="spring", alpha=0.48, vmin=0, vmax=1)
        axes[row, 4].set_title(f"PaDiM overlay\nF1={result['f1']:.3f}, IoU={result['iou']:.3f}")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("PaDiM examples: map → thresholded binary mask → overlay")
    figure.tight_layout()
    figure.savefig(output_dir / "padim_examples.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    (output_dir / "results.json").write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "image": str(output_dir / "padim_examples.png")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
