"""Reusable normal patch feature-bank build, save, and reuse logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .dinov2 import SharedDINOv2, image_tensor


@dataclass(frozen=True)
class FeatureBankResult:
    """A feature bank and whether it came from an existing compatible file."""

    features: torch.Tensor
    metadata: dict[str, Any]
    reused: bool
    path: Path


def feature_bank_path(bank_root: str | Path, category: str) -> Path:
    """Return the versioned feature-bank file for one category."""

    root = Path(bank_root).resolve()
    path = (root / category / "feature_bank.pt").resolve()
    if not path.parent.is_relative_to(root):
        raise ValueError("Category would leave the feature-bank root.")
    return path


def make_bank_metadata(
    category: str,
    train_paths: list[Path],
    processed_root: str | Path,
    encoder: SharedDINOv2,
) -> dict[str, Any]:
    """Create metadata matching the existing notebook bank schema."""

    processed_root = Path(processed_root).resolve()
    relative_paths = []
    for path in train_paths:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(processed_root):
            raise ValueError(f"Training path leaves processed root: {path}")
        relative = resolved.relative_to(processed_root)
        if relative.parts[-4:-1] != (category, "train", "good"):
            raise ValueError(f"Training path is not category/train/good: {path}")
        relative_paths.append(relative.as_posix())
    encoder_metadata = encoder.metadata()
    return {
        "category": category,
        "model_name": encoder_metadata["model_name"],
        "model_revision": encoder_metadata["model_revision"],
        "feature_key": encoder_metadata["patch_feature_key"],
        "image_size": encoder_metadata["image_size"],
        "padding": [encoder_metadata["padding"][1], encoder_metadata["padding"][0]],
        "mean": encoder_metadata["mean"],
        "std": encoder_metadata["std"],
        "l2_normalization": encoder_metadata["l2_normalize_patches"],
        "embedding_dimension": encoder_metadata["embedding_dimension"],
        "patch_grid": encoder_metadata["patch_grid"],
        "sampling": "none; all train/good patches",
        "train_paths": relative_paths,
    }


def build_feature_bank(
    category: str,
    train_paths: list[Path],
    processed_root: str | Path,
    encoder: SharedDINOv2,
    bank_device: str | torch.device = "cpu",
    batch_size: int = 8,
    max_bank_gib: float = 4.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extract normalized patch features for all normal train images."""

    if not train_paths:
        raise ValueError("train_paths must not be empty.")
    if batch_size < 1 or max_bank_gib <= 0:
        raise ValueError("batch_size must be positive and max_bank_gib must be positive.")
    processed_root = Path(processed_root)
    bank_device = torch.device(bank_device)
    metadata = make_bank_metadata(category, train_paths, processed_root, encoder)
    dimension = int(metadata["embedding_dimension"])
    patch_count = int(metadata["patch_grid"][0] * metadata["patch_grid"][1])
    number_of_features = len(train_paths) * patch_count
    estimated_gib = number_of_features * dimension * 4 / (1024**3)
    if estimated_gib > max_bank_gib:
        raise MemoryError(
            f"Feature bank needs {estimated_gib:.3f} GiB, above max_bank_gib={max_bank_gib}."
        )

    bank = torch.empty((number_of_features, dimension), dtype=torch.float32, device=bank_device)
    offset = 0
    for start in range(0, len(train_paths), batch_size):
        paths = train_paths[start : start + batch_size]
        images = torch.stack([image_tensor(path, encoder.config) for path in paths])
        features = encoder.forward(images).patch_embeddings
        if tuple(features.shape[1:]) != (patch_count, dimension):
            raise ValueError(f"Unexpected patch feature shape: {tuple(features.shape)}")
        patches = features.reshape(-1, dimension).detach().to(bank_device)
        next_offset = offset + patches.shape[0]
        bank[offset:next_offset] = patches
        offset = next_offset
    if offset != number_of_features or not torch.isfinite(bank).all().item():
        raise ValueError("The feature bank is incomplete or contains invalid features.")
    return bank, metadata


def save_feature_bank(
    path: str | Path,
    features: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    """Save CPU features and metadata in the existing bank payload format."""

    if features.ndim != 2 or not torch.isfinite(features).all().item():
        raise ValueError("Feature bank must be a finite two-dimensional tensor.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.detach().cpu(), "metadata": metadata}, path)


def load_feature_bank(
    path: str | Path,
    expected_metadata: dict[str, Any],
    bank_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load one bank only when its metadata and shape match the request."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        raise ValueError("Saved feature bank metadata does not match the current configuration.")
    features = payload.get("features")
    expected_rows = len(expected_metadata["train_paths"]) * (
        expected_metadata["patch_grid"][0] * expected_metadata["patch_grid"][1]
    )
    expected_shape = (expected_rows, expected_metadata["embedding_dimension"])
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
        raise ValueError(f"Saved feature bank shape is not {expected_shape}.")
    if features.dtype != torch.float32 or not torch.isfinite(features).all().item():
        raise ValueError("Saved feature bank contains invalid features.")
    return features.to(torch.device(bank_device))


def load_or_build_feature_bank(
    bank_root: str | Path,
    category: str,
    train_paths: list[Path],
    processed_root: str | Path,
    encoder: SharedDINOv2,
    bank_device: str | torch.device = "cpu",
    batch_size: int = 8,
    max_bank_gib: float = 4.0,
    rebuild: bool = False,
) -> FeatureBankResult:
    """Reuse a compatible bank; rebuild only when missing or explicitly allowed."""

    path = feature_bank_path(bank_root, category)
    metadata = make_bank_metadata(category, train_paths, processed_root, encoder)
    if path.is_file():
        try:
            features = load_feature_bank(path, metadata, bank_device)
            return FeatureBankResult(features, metadata, True, path)
        except (OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError):
            if not rebuild:
                raise
    features, metadata = build_feature_bank(
        category,
        train_paths,
        processed_root,
        encoder,
        bank_device,
        batch_size,
        max_bank_gib,
    )
    save_feature_bank(path, features, metadata)
    return FeatureBankResult(features, metadata, False, path)
