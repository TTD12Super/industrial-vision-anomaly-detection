"""Generic detector and DINOv2 classifier training from canonical product manifests."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml

from mvtec_system.classification.classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device

from .config import ProductConfig
from .dataset import load_dataset_records
from .geometry import oriented_crop


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to overwrite detector input: {destination}")
    destination.symlink_to(source.resolve())


def materialize_yolo_obb_dataset(config: ProductConfig, output_root: str | Path) -> dict[str, Any]:
    """Create a symlinked YOLO-OBB view of any valid product manifest."""

    images, objects = load_dataset_records(config)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated detector dataset: {output_root}")
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in objects:
        by_image[row["image_id"]].append(row)
    counts = {"train": 0, "val": 0, "test": 0}
    for image in images:
        split = {"validation": "val"}.get(image["split"], image["split"])
        source = config.dataset_path(Path(image["relative_path"]))
        suffix = source.suffix.lower() or ".png"
        destination = output_root / "images" / split / f"{image['image_id']}{suffix}"
        _symlink(source, destination)
        lines: list[str] = []
        for object_row in by_image[image["image_id"]]:
            values: list[float] = [0.0]
            for x, y in object_row["obb_xyxyxyxy"]:
                values.extend([float(x) / image["width"], float(y) / image["height"]])
            lines.append(" ".join(f"{value:.7f}" for value in values))
        label = output_root / "labels" / split / f"{image['image_id']}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines) + ("\n" if lines else "")
        if label.exists() and label.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite changed detector label: {label}")
        if not label.exists():
            label.write_text(content, encoding="utf-8")
        counts[split] += 1
    data_yaml = output_root / "data.yaml"
    data = {"path": str(output_root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: config.product_key}}
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    summary = {"product_key": config.product_key, "root": str(output_root.resolve()), "data_yaml": str(data_yaml.resolve()), "images": counts}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def train_yolo_obb(
    config: ProductConfig,
    output_root: str | Path,
    model: str,
    epochs: int,
    image_size: int,
    device: str | None,
) -> dict[str, Any]:
    """Train YOLO-OBB only after the generic manifest has been materialized."""

    if epochs < 1 or image_size < 32:
        raise ValueError("epochs must be positive and image_size must be at least 32")
    summary = materialize_yolo_obb_dataset(config, output_root)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("ultralytics is required to train YOLO-OBB") from error
    model_instance = YOLO(model)
    options: dict[str, Any] = {
        "data": summary["data_yaml"],
        "task": "obb",
        "epochs": epochs,
        "imgsz": image_size,
        "project": str(Path(output_root) / "training"),
        "name": "run",
        "degrees": 20.0,
        "fliplr": 0.5,
    }
    if device:
        options["device"] = device
    results = model_instance.train(**options)
    return {**summary, "training_result": str(results)}


def _tensor(crop: np.ndarray, config: DINOv2Config) -> torch.Tensor:
    image = Image.fromarray(crop).resize((config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(config.mean).view(3, 1, 1)
    std = torch.tensor(config.std).view(3, 1, 1)
    return (pixels - mean) / std


def train_dinov2_logreg(
    config: ProductConfig,
    output_dir: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train a frozen DINOv2 GOOD/DEFECT head from oriented product object crops."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    images, objects = load_dataset_records(config)
    image_by_id = {row["image_id"]: row for row in images}
    rows = []
    for object_row in objects:
        image = image_by_id[object_row["image_id"]]
        rows.append({**object_row, "split": image["split"], "image": image})
    if any(not any(row["split"] == split for row in rows) for split in ("train", "validation", "test")):
        raise ValueError("Every object split train, validation and test is required")
    encoder_config = DINOv2Config(image_size=(256, 256))
    encoder_root = config.model("classifier").get("encoder_artifact_root", "artifacts/dinov2_knn")
    encoder = SharedDINOv2(encoder_config, artifact_root=config.project_path(encoder_root), device=device or str(select_device()))
    split_embeddings: dict[str, torch.Tensor] = {}
    split_labels: dict[str, torch.Tensor] = {}
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        encoded: list[torch.Tensor] = []
        for offset in range(0, len(selected), batch_size):
            tensors = []
            for row in selected[offset : offset + batch_size]:
                with Image.open(config.dataset_path(Path(row["image"]["relative_path"]))) as source:
                    crop = oriented_crop(np.asarray(source.convert("RGB")), row["obb_xyxyxyxy"]).image
                tensors.append(_tensor(crop, encoder_config))
            encoded.append(encoder.forward(torch.stack(tensors)).global_embedding.cpu())
        split_embeddings[split] = torch.cat(encoded)
        split_labels[split] = torch.tensor([0 if row["quality_label"] == "GOOD" else 1 for row in selected], dtype=torch.long)
        split_ids[split] = [row["object_id"] for row in selected]
        if set(split_labels[split].tolist()) != {0, 1}:
            raise ValueError(f"{split} must contain GOOD and DEFECT objects")
    random_generator = torch.Generator().manual_seed(seed)
    train = split_embeddings["train"]
    labels = split_labels["train"]
    good_indices = torch.where(labels == 0)[0]
    defect_indices = torch.where(labels == 1)[0]
    target = max(len(good_indices), len(defect_indices))
    balanced_indices = torch.cat(
        [
            good_indices[torch.randint(len(good_indices), (target,), generator=random_generator)],
            defect_indices[torch.randint(len(defect_indices), (target,), generator=random_generator)],
        ]
    )
    classifier = fit_linear_classifier(
        {"embeddings": train[balanced_indices], "labels": labels[balanced_indices]},
        {"GOOD": 0, "DEFECT": 1},
        LinearClassifierConfig(random_state=seed),
    )
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite classifier output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_linear_classifier(output_dir / "classifier", classifier, {"GOOD": 0, "DEFECT": 1}, LinearClassifierConfig(random_state=seed), {"artifact_version": "v1", "encoder": encoder.metadata()})
    metrics: dict[str, Any] = {"product_key": config.product_key, "seed": seed, "encoder": encoder.metadata(), "splits": {}}
    for split in ("train", "validation", "test"):
        _, probabilities = predict_linear_classifier(classifier, split_embeddings[split])
        torch.save({"object_ids": split_ids[split], "embeddings": split_embeddings[split], "labels": split_labels[split]}, output_dir / f"{split}_embeddings.pt")
        metrics["splits"][split] = {"objects": len(split_ids[split]), "good": int((split_labels[split] == 0).sum()), "defect": int((split_labels[split] == 1).sum()), "mean_p_defect": float(probabilities[:, 1].mean())}
    (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
