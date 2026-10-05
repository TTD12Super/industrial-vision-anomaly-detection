"""Manifest loading and cached global embeddings for product classification."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .dinov2 import SharedDINOv2, image_tensor


MANIFEST_FIELDS = ["image_path", "category", "split"]
SPLITS = ("train", "validation")


def read_classification_manifest(
    manifest_path: str | Path,
    processed_root: str | Path,
) -> list[dict[str, str]]:
    """Read and validate the metadata-only classification split."""

    manifest_path = Path(manifest_path)
    processed_root = Path(processed_root).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(f"Unexpected manifest columns: {reader.fieldnames}")
        rows = list(reader)

    if not rows:
        raise ValueError("Classification manifest is empty.")

    for row in rows:
        relative_path = Path(row["image_path"])
        if relative_path.is_absolute() or row["split"] not in SPLITS:
            raise ValueError(f"Invalid manifest row: {row}")
        if relative_path.parts[-4:-1] != (row["category"], "train", "good"):
            raise ValueError(f"Image is not a train/good path: {row['image_path']}")
        image_path = (processed_root / relative_path).resolve()
        if not image_path.is_relative_to(processed_root):
            raise ValueError(f"Manifest path leaves processed root: {row['image_path']}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    return rows


def category_mapping(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    categories = sorted({row["category"] for row in rows})
    if not categories:
        raise ValueError("No categories found in manifest.")
    return {category: index for index, category in enumerate(categories)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def build_global_embedding_artifact(
    rows: list[dict[str, str]],
    split: str,
    processed_root: str | Path,
    encoder: SharedDINOv2,
    mapping: dict[str, int],
    batch_size: int = 8,
) -> dict[str, Any]:
    """Extract and return CPU global embeddings for one manifest split."""

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    processed_root = Path(processed_root)
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No rows found for split: {split}")

    embeddings = []
    labels = []
    image_paths = []
    categories = []
    for start in range(0, len(selected), batch_size):
        batch_rows = selected[start : start + batch_size]
        batch = torch.stack(
            [image_tensor(processed_root / row["image_path"], encoder.config) for row in batch_rows]
        )
        features = encoder.forward(batch).global_embedding.detach().cpu()
        if features.ndim != 2 or features.shape[0] != len(batch_rows):
            raise ValueError(f"Unexpected global embedding shape: {tuple(features.shape)}")
        if not torch.isfinite(features).all().item():
            raise ValueError("Global embeddings contain non-finite values.")
        embeddings.append(features)
        labels.extend(mapping[row["category"]] for row in batch_rows)
        image_paths.extend(row["image_path"] for row in batch_rows)
        categories.extend(row["category"] for row in batch_rows)

    return {
        "embeddings": torch.cat(embeddings, dim=0).float(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_paths": image_paths,
        "categories": categories,
        "split": split,
    }


def make_embedding_metadata(
    rows: list[dict[str, str]],
    mapping: dict[str, int],
    manifest_path: str | Path,
    processed_root: str | Path,
    encoder: SharedDINOv2,
) -> dict[str, Any]:
    """Create serializable metadata for compatibility checks and reuse."""

    metadata = encoder.metadata()
    split_metadata = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        split_metadata[split] = {
            "count": len(split_rows),
            "source_image_paths": [row["image_path"] for row in split_rows],
            "source_categories": [row["category"] for row in split_rows],
        }
    return {
        "artifact_version": "v1",
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": sha256_file(manifest_path),
        "processed_root": str(Path(processed_root)),
        "categories": list(mapping),
        "class_mapping": mapping,
        "global_feature_key": encoder.config.global_feature_key,
        "global_l2_normalized": False,
        "embedding_dtype": "float32",
        "splits": split_metadata,
        "encoder": metadata,
    }


def save_embedding_artifacts(
    output_dir: str | Path,
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    mapping: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    """Save the two split tensors and their compatibility metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_artifact, output_dir / "train_embeddings.pt")
    torch.save(validation_artifact, output_dir / "validation_embeddings.pt")
    (output_dir / "class_mapping.json").write_text(
        json.dumps(
            {
                "categories": list(mapping),
                "category_to_index": mapping,
                "index_to_category": {str(index): category for category, index in mapping.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def load_compatible_embedding_artifacts(
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return cached splits only when every compatibility record matches."""

    output_dir = Path(output_dir)
    required = (
        output_dir / "train_embeddings.pt",
        output_dir / "validation_embeddings.pt",
        output_dir / "class_mapping.json",
        output_dir / "embedding_metadata.json",
    )
    if not all(path.is_file() for path in required):
        return None

    try:
        stored_metadata = json.loads(required[3].read_text(encoding="utf-8"))
        stored_mapping = json.loads(required[2].read_text(encoding="utf-8"))
        if stored_metadata != metadata or stored_mapping.get("category_to_index") != metadata["class_mapping"]:
            return None
        artifacts = {
            "train": torch.load(required[0], map_location="cpu", weights_only=True),
            "validation": torch.load(required[1], map_location="cpu", weights_only=True),
        }
    except (OSError, EOFError, ImportError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None

    for split, artifact in artifacts.items():
        expected_paths = metadata["splits"][split]["source_image_paths"]
        expected_categories = metadata["splits"][split]["source_categories"]
        expected_labels = [metadata["class_mapping"][category] for category in expected_categories]
        if not isinstance(artifact, dict):
            return None
        if artifact.get("split") != split or artifact.get("image_paths") != expected_paths:
            return None
        embeddings = artifact.get("embeddings")
        labels = artifact.get("labels")
        categories = artifact.get("categories")
        if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_paths):
            return None
        if embeddings.shape[1] != metadata["encoder"]["embedding_dimension"]:
            return None
        if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all().item():
            return None
        if labels.ndim != 1 or labels.shape[0] != len(expected_paths):
            return None
        if not isinstance(categories, list) or len(categories) != len(expected_paths):
            return None
        if labels.dtype != torch.int64:
            return None
        if categories != expected_categories or labels.tolist() != expected_labels:
            return None

    return artifacts["train"], artifacts["validation"]
