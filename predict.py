"""Command-line inference for the unified MVTec pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mvtec_system import SharedDINOv2, infer_image


def build_parser() -> argparse.ArgumentParser:
    """Build the small inference CLI."""

    parser = argparse.ArgumentParser(description="Classify and inspect one MVTec image.")
    parser.add_argument("--image", required=True, type=Path, help="RGB 256x256 image path")
    parser.add_argument(
        "--classifier-root",
        type=Path,
        default=Path("artifacts/product_classifier/v1"),
    )
    parser.add_argument(
        "--dinov2-artifact-root",
        type=Path,
        default=Path("artifacts/dinov2_knn"),
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=Path("artifacts/dinov2_knn/final_run"),
    )
    parser.add_argument("--image-thresholds", type=Path, default=None)
    parser.add_argument("--pixel-thresholds", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/inference"))
    parser.add_argument("--device", default=None, help="cpu, cuda, mps, or leave empty for auto")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_inputs(args: argparse.Namespace) -> tuple[Any, ...]:
    root = args.classifier_root
    classifier = torch.load(root / "classifier.pt", map_location="cpu", weights_only=True)
    if not isinstance(classifier, dict):
        raise ValueError("classifier.pt must contain a dictionary.")
    mapping_record = _read_json(root / "class_mapping.json")
    class_mapping = mapping_record.get("category_to_index")
    if not isinstance(class_mapping, dict):
        raise ValueError("class_mapping.json is missing category_to_index.")
    class_mapping = {str(category): int(index) for category, index in class_mapping.items()}
    reference_artifact = torch.load(
        root / "train_embeddings.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(reference_artifact, dict):
        raise ValueError("train_embeddings.pt must contain a dictionary.")
    calibration = _read_json(root / "classification_calibration.json")
    image_thresholds = (
        _read_json(args.image_thresholds) if args.image_thresholds is not None else None
    )
    pixel_thresholds = (
        _read_json(args.pixel_thresholds) if args.pixel_thresholds is not None else None
    )
    encoder = SharedDINOv2(artifact_root=args.dinov2_artifact_root, device=args.device)
    return (
        encoder,
        classifier,
        reference_artifact,
        class_mapping,
        calibration,
        image_thresholds,
        pixel_thresholds,
    )


def _save_heatmap(anomaly_map: Any, output_dir: Path, image_path: Path) -> Path:
    """Save one continuous anomaly heatmap and return its path."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{image_path.stem}_heatmap.png"
    plt.imsave(path, anomaly_map, cmap="inferno")
    plt.close()
    return path


def make_output_payload(
    result: dict[str, Any],
    calibration: dict[str, Any],
    heatmap_path: Path | None,
) -> dict[str, Any]:
    """Keep CLI JSON readable while replacing arrays with useful paths/shapes."""

    payload = {
        key: value
        for key, value in result.items()
        if key not in {"patch_scores", "anomaly_map", "binary_mask"}
    }
    payload["classification_confidence"] = result.get("confidence")
    payload["nearest_global_distance"] = result.get("nearest_global_distance")
    payload["reject_thresholds"] = {
        "confidence": calibration.get("confidence_threshold"),
        "nearest_global_distance": calibration.get("nearest_global_distance_threshold"),
    }
    payload["heatmap_path"] = str(heatmap_path) if heatmap_path is not None else None
    anomaly_map = result.get("anomaly_map")
    binary_mask = result.get("binary_mask")
    payload["anomaly_map_shape"] = list(anomaly_map.shape) if anomaly_map is not None else None
    payload["binary_mask_shape"] = list(binary_mask.shape) if binary_mask is not None else None
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run CLI inference and print a JSON result."""

    args = build_parser().parse_args(argv)
    (
        encoder,
        classifier,
        reference_artifact,
        class_mapping,
        calibration,
        image_thresholds,
        pixel_thresholds,
    ) = _load_inputs(args)
    result = infer_image(
        args.image,
        encoder,
        classifier,
        reference_artifact,
        class_mapping,
        calibration,
        args.bank_root,
        image_thresholds,
        pixel_thresholds,
    )
    heatmap_path = None
    if result.get("anomaly_map") is not None:
        heatmap_path = _save_heatmap(result["anomaly_map"], args.output_dir, args.image)
    print(json.dumps(make_output_payload(result, calibration, heatmap_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
