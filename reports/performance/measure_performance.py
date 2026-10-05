"""Measure one real MVTec inference path on selected torch devices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mvtec_system import (
    SharedDINOv2,
    available_backends,
    file_size_mb,
    image_tensor,
    infer_image,
    measure_callable,
    memory_usage_mb,
    nearest_global_distances,
    nearest_neighbor_scores,
    predict_linear_classifier,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _measure_device(
    device_name: str,
    image_path: Path,
    classifier_root: Path,
    dinov2_root: Path,
    bank_root: Path,
) -> dict[str, Any]:
    device = torch.device(device_name)
    classifier = torch.load(classifier_root / "classifier.pt", map_location="cpu", weights_only=True)
    reference = torch.load(
        classifier_root / "train_embeddings.pt", map_location="cpu", weights_only=True
    )
    mapping_record = _read_json(classifier_root / "class_mapping.json")
    mapping = {str(k): int(v) for k, v in mapping_record["category_to_index"].items()}
    calibration = _read_json(classifier_root / "classification_calibration.json")
    encoder = SharedDINOv2(artifact_root=dinov2_root, device=device)
    images = image_tensor(image_path, encoder.config).unsqueeze(0)
    features, dino_seconds = measure_callable(lambda: encoder.forward(images), device)
    _, classification_seconds = measure_callable(
        lambda: predict_linear_classifier(classifier, features.global_embedding), device
    )
    _, routing_distance_seconds = measure_callable(
        lambda: nearest_global_distances(
            features.global_embedding,
            reference["embeddings"],
        ),
        device,
    )
    category = next(name for name, index in mapping.items() if index == int(
        predict_linear_classifier(classifier, features.global_embedding)[0].argmax(dim=1).item()
    ))
    bank_path = bank_root / category / "feature_bank.pt"
    bank, bank_seconds = measure_callable(
        lambda: torch.load(bank_path, map_location="cpu", weights_only=True)["features"],
    )
    queries = features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1])
    _, knn_seconds = measure_callable(
        lambda: nearest_neighbor_scores(queries, bank), device
    )
    _, total_seconds = measure_callable(
        lambda: infer_image(
            image_path,
            encoder,
            classifier,
            reference,
            mapping,
            calibration,
            bank_root,
        ),
        device,
    )
    return {
        "device": device_name,
        "status": "ok",
        "image": str(image_path),
        "category_for_bank": category,
        "dino_seconds": dino_seconds,
        "classification_seconds": classification_seconds,
        "routing_distance_seconds": routing_distance_seconds,
        "bank_loading_seconds": bank_seconds,
        "knn_seconds": knn_seconds,
        "total_inference_seconds": total_seconds,
        "ram_high_water_mb": memory_usage_mb(),
        "feature_bank_size_mb": file_size_mb(bank_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("datasets/mvtec_ad_processed/bottle/test/good/000.png"),
    )
    parser.add_argument("--classifier-root", type=Path, default=Path("artifacts/product_classifier/v1"))
    parser.add_argument("--dinov2-root", type=Path, default=Path("artifacts/dinov2_knn"))
    parser.add_argument("--bank-root", type=Path, default=Path("artifacts/dinov2_knn/final_run"))
    parser.add_argument("--devices", default="cpu", help="Comma-separated cpu,mps,cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/performance/performance.json"),
    )
    args = parser.parse_args()
    backend_status = available_backends()
    measurements = []
    for name in [value.strip() for value in args.devices.split(",") if value.strip()]:
        if name not in backend_status:
            measurements.append({"device": name, "status": "unsupported"})
            continue
        if not backend_status[name]:
            measurements.append({"device": name, "status": "unavailable"})
            continue
        try:
            measurements.append(
                _measure_device(name, args.image, args.classifier_root, args.dinov2_root, args.bank_root)
            )
        except Exception as error:
            measurements.append({"device": name, "status": "error", "error": str(error)})
    report = {
        "backends": backend_status,
        "measurements": measurements,
        "note": "Single-image wall-clock measurements; no optimization or threshold tuning performed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
