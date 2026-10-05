"""Run the full official MVTec product-category test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mvtec_system import SharedDINOv2, evaluate_product_classification_test_split


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate product-category routing on all official MVTec test images."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mvtec_ad_processed"))
    parser.add_argument(
        "--classifier-root", type=Path, default=Path("artifacts/product_classifier/v1")
    )
    parser.add_argument(
        "--dinov2-artifact-root", type=Path, default=Path("artifacts/dinov2_knn")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/product_classification_test")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps, or automatic")
    parser.add_argument(
        "--classification-confidence-threshold",
        type=float,
        default=None,
        help="minimum product confidence; default is 0.50",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.classifier_root
    classifier = torch.load(root / "classifier.pt", map_location="cpu", weights_only=True)
    reference_artifact = torch.load(
        root / "train_embeddings.pt", map_location="cpu", weights_only=True
    )
    mapping_record = _read_json(root / "class_mapping.json")
    class_mapping = mapping_record.get("category_to_index")
    if not isinstance(class_mapping, dict):
        raise ValueError("class_mapping.json is missing category_to_index")
    class_mapping = {str(category): int(index) for category, index in class_mapping.items()}
    calibration = _read_json(root / "classification_calibration.json")
    encoder = SharedDINOv2(artifact_root=args.dinov2_artifact_root, device=args.device)
    summary = evaluate_product_classification_test_split(
        args.dataset_root,
        args.output_dir,
        encoder,
        classifier,
        reference_artifact,
        class_mapping,
        calibration,
        batch_size=args.batch_size,
        classification_confidence_threshold=args.classification_confidence_threshold,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
