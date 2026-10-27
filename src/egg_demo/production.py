"""Build a combined MVTec + egg routing and exact-kNN artifact set."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from mvtec_system import (
    DINOv2Config,
    LinearClassifierConfig,
    SharedDINOv2,
    build_global_embedding_artifact,
    calibrate_category_image_thresholds,
    calibrate_category_pixel_thresholds,
    calibrate_open_set_thresholds,
    evaluate_product_classifier,
    fit_linear_classifier,
    image_scores_from_patch_scores,
    image_tensor,
    load_or_build_feature_bank,
    make_anomaly_maps,
    make_embedding_metadata,
    nearest_neighbor_scores,
    save_calibration,
    save_classification_report,
    save_embedding_artifacts,
    save_linear_classifier,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _quality_paths(root: Path, split: str) -> list[Path]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        paths = [
            root / row["image_path"]
            for row in csv.DictReader(handle)
            if row["split"] == split and row["label"] == "good"
        ]
    if not paths:
        raise ValueError(f"No GOOD egg images found for split: {split}")
    return paths


def _relative_paths(root: Path, paths: list[Path]) -> list[str]:
    return [path.resolve().relative_to(root.resolve()).as_posix() for path in paths]


def _combined_artifact(base: dict[str, Any], egg: dict[str, Any], split: str) -> dict[str, Any]:
    base_paths = [str(Path("datasets/mvtec_ad_processed") / path) for path in base["image_paths"]]
    egg_paths = [str(Path("datasets/processed/egg_quality") / path) for path in egg["image_paths"]]
    return {
        "embeddings": torch.cat([base["embeddings"], egg["embeddings"]]).float(),
        "labels": torch.cat([base["labels"], egg["labels"]]).long(),
        "image_paths": base_paths + egg_paths,
        "categories": list(base["categories"]) + list(egg["categories"]),
        "split": split,
    }


def _write_manifest(path: Path, train: dict[str, Any], validation: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for artifact, split in ((train, "train"), (validation, "validation")):
        rows.extend(
            {
                "image_path": image_path,
                "category": category,
                "split": split,
            }
            for image_path, category in zip(artifact["image_paths"], artifact["categories"])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "category", "split"))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _link_existing_banks(base_root: Path, output_root: Path) -> None:
    source_root = base_root / "feature_banks"
    target_root = output_root / "feature_banks"
    target_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_root.iterdir()):
        target = target_root / source.name
        if target.is_symlink():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("feature_bank.pt", "bank_metadata.json"):
            source_file = source / filename
            target_file = target / filename
            if target_file.exists():
                continue
            try:
                os.link(source_file, target_file)
            except OSError:
                shutil.copy2(source_file, target_file)


def _build_egg_thresholds(
    validation_paths: list[Path],
    bank: torch.Tensor,
    encoder: SharedDINOv2,
    target_acceptance_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_scores = []
    pixel_scores = []
    for start in range(0, len(validation_paths), 8):
        paths = validation_paths[start : start + 8]
        images = torch.stack([image_tensor(path, encoder.config) for path in paths])
        features = encoder.forward(images)
        patch_scores = nearest_neighbor_scores(
            features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1]), bank
        ).reshape(len(paths), -1)
        maps = make_anomaly_maps(
            patch_scores,
            encoder.patch_grid,
            encoder.padded_size,
            encoder.config.image_size,
        )
        image_scores.append(image_scores_from_patch_scores(patch_scores))
        pixel_scores.append(torch.as_tensor(maps, dtype=torch.float32).reshape(-1))
    image_thresholds = calibrate_category_image_thresholds(
        {"egg": torch.cat(image_scores)}, target_acceptance_rate
    )
    pixel_thresholds = calibrate_category_pixel_thresholds(
        {"egg": torch.cat(pixel_scores)}, target_acceptance_rate
    )
    return image_thresholds, pixel_thresholds


def build_egg_production(
    quality_root: str | Path,
    base_artifact_root: str | Path,
    output_root: str | Path,
    encoder_artifact_root: str | Path,
    batch_size: int = 8,
    target_acceptance_rate: float = 0.99,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build the egg bank and a 16-class MVTec + egg classifier artifact."""

    quality_root = Path(quality_root)
    base_root = Path(base_artifact_root)
    output_root = Path(output_root)
    encoder = SharedDINOv2(
        DINOv2Config(image_size=(256, 256), feature_layers=()),
        artifact_root=encoder_artifact_root,
        device=device,
    )
    train_paths = _quality_paths(quality_root, "train")
    validation_paths = _quality_paths(quality_root, "validation")

    _link_existing_banks(base_root, output_root)
    bank = load_or_build_feature_bank(
        output_root / "feature_banks",
        "egg",
        train_paths,
        quality_root,
        encoder,
        batch_size=batch_size,
    )
    (bank.path.parent / "bank_metadata.json").write_text(
        json.dumps(bank.metadata, indent=2) + "\n", encoding="utf-8"
    )

    base_mapping = _read_json(base_root / "classifier" / "class_mapping.json")["category_to_index"]
    class_mapping = {str(category): int(index) for category, index in base_mapping.items()}
    if "egg" in class_mapping:
        raise ValueError("Base classifier already contains egg")
    class_mapping["egg"] = len(class_mapping)

    egg_rows = [
        {"image_path": path, "category": "egg", "split": split}
        for split, paths in (("train", train_paths), ("validation", validation_paths))
        for path in _relative_paths(quality_root, paths)
    ]
    egg_train = build_global_embedding_artifact(
        [row for row in egg_rows if row["split"] == "train"],
        "train", quality_root, encoder, class_mapping, batch_size
    )
    egg_validation = build_global_embedding_artifact(
        [row for row in egg_rows if row["split"] == "validation"],
        "validation", quality_root, encoder, class_mapping, batch_size
    )
    base_train = torch.load(base_root / "embeddings" / "train_embeddings.pt", map_location="cpu", weights_only=True)
    base_validation = torch.load(base_root / "embeddings" / "validation_embeddings.pt", map_location="cpu", weights_only=True)
    train_artifact = _combined_artifact(base_train, egg_train, "train")
    validation_artifact = _combined_artifact(base_validation, egg_validation, "validation")
    manifest_path = output_root / "combined_classification_manifest.csv"
    rows = _write_manifest(manifest_path, train_artifact, validation_artifact)
    embedding_metadata = make_embedding_metadata(
        rows, class_mapping, manifest_path, Path("."), encoder
    )
    save_embedding_artifacts(
        output_root / "embeddings", train_artifact, validation_artifact,
        class_mapping, embedding_metadata
    )

    classifier_config = LinearClassifierConfig()
    classifier = fit_linear_classifier(train_artifact, class_mapping, classifier_config)
    save_linear_classifier(
        output_root / "classifier", classifier, class_mapping,
        classifier_config, embedding_metadata
    )
    classifier_root = output_root / "classifier"
    for split in ("train", "validation"):
        link = classifier_root / f"{split}_embeddings.pt"
        if not link.exists():
            link.symlink_to((output_root / "embeddings" / f"{split}_embeddings.pt").resolve())

    calibration = calibrate_open_set_thresholds(
        classifier, train_artifact, validation_artifact, class_mapping,
        classification_confidence_threshold=0.5,
    )
    save_calibration(output_root / "calibration.json", calibration)
    save_calibration(classifier_root / "classification_calibration.json", calibration)
    report = evaluate_product_classifier(classifier, validation_artifact, class_mapping)
    save_classification_report(output_root / "classification_report.json", report)

    image_thresholds, pixel_thresholds = _build_egg_thresholds(
        validation_paths, bank.features, encoder, target_acceptance_rate
    )
    base_image = _read_json(base_root / "thresholds" / "exact_first" / "image_thresholds.json")
    base_pixel = _read_json(base_root / "thresholds" / "exact_first" / "pixel_thresholds.json")
    base_image["categories"]["egg"] = image_thresholds["categories"]["egg"]
    base_pixel["categories"]["egg"] = pixel_thresholds["categories"]["egg"]
    threshold_root = output_root / "thresholds" / "exact_first"
    threshold_root.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (threshold_root / "image_thresholds.json", base_image),
        (threshold_root / "pixel_thresholds.json", base_pixel),
        (output_root / "image_thresholds.json", base_image),
        (output_root / "pixel_thresholds.json", base_pixel),
    ):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return {
        "output_root": str(output_root),
        "class_count": len(class_mapping),
        "classes": class_mapping,
        "egg_train_images": len(train_paths),
        "egg_validation_images": len(validation_paths),
        "egg_feature_bank": str(bank.path),
        "feature_bank_reused": bank.reused,
        "classification_accuracy": report["metrics"]["accuracy"],
        "classification_macro_f1": report["metrics"]["macro_f1"],
        "egg_image_threshold": image_thresholds["categories"]["egg"]["threshold"],
        "egg_pixel_threshold": pixel_thresholds["categories"]["egg"]["threshold"],
    }
