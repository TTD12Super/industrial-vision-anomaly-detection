"""Normal feature-bank construction, persistence, and loading.

Rules preserved from the notebook:
* the bank contains **every** patch of **every** ``train/good`` image of one
  category — no sampling, no test images, no test labels, no threshold
  calibration on the test set
* saved payload shape: ``{"features": Tensor, "metadata": dict}`` (``.pt``)
* reloading verifies provenance metadata, expected shape, and finite values
* banks are large local artifacts and must never be committed to Git
  (``artifacts/dinov2_knn/`` is already git-ignored)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .config import (
    FEATURE_KEY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INFERENCE_BATCH_SIZE,
    L2_NORMALIZE_FEATURES,
    MAX_BANK_GIB,
    MODEL_NAME,
    MODEL_REVISION,
    PROCESSED_SIZE,
    SUPPORTED_EXTENSIONS,
    get_bank_device,
    validate_category,
)
from .preprocessing import image_tensor

if TYPE_CHECKING:
    import torch

    from .feature_extractor import DINOv2FeatureExtractor

logger = logging.getLogger(__name__)


def list_train_images(train_good_dir: str | Path) -> list[Path]:
    """Sorted supported images under a category ``train/good`` folder."""
    folder = Path(train_good_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"train/good folder not found: {folder}.")
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images found under {folder}.")
    return files


def bank_metadata(
    category: str,
    *,
    embedding_dimension: int,
    patch_grid: tuple[int, int],
    padding: tuple[int, int],
    train_paths: list[str],
) -> dict:
    """Provenance metadata stored alongside a bank (notebook ``bank_metadata``)."""
    validate_category(category)
    return {
        "category": category,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "feature_key": FEATURE_KEY,
        "image_size": list(PROCESSED_SIZE),
        "padding": [int(padding[0]), int(padding[1])],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "l2_normalization": L2_NORMALIZE_FEATURES,
        "embedding_dimension": int(embedding_dimension),
        "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "sampling": "none; all train/good patches",
        "train_paths": list(train_paths),
    }


def save_feature_bank(path: str | Path, bank: torch.Tensor, metadata: dict) -> Path:
    """Persist a bank payload ``{"features": ..., "metadata": ...}`` to ``.pt``."""
    import torch as _torch

    destination = Path(path)
    if bank.ndim != 2:
        raise ValueError(f"Feature bank must be 2D, got shape {tuple(bank.shape)}.")
    if not _torch.isfinite(bank).all().item():
        raise ValueError("Refusing to save a bank containing non-finite values.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _torch.save({"features": bank.detach().cpu(), "metadata": dict(metadata)}, destination)
    logger.info("Saved feature bank (%s) to %s.", tuple(bank.shape), destination)
    return destination


def load_feature_bank(
    path: str | Path,
    expected_metadata: dict | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict]:
    """Load a bank, verifying metadata, shape, and finite values.

    Args:
        path: ``.pt`` file created by :func:`save_feature_bank`.
        expected_metadata: if given, must equal the stored metadata exactly
            (detects reuse across different settings/preprocessing).
        device: target device; defaults to the notebook bank-device rule.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature bank not found: {file_path}.")
    import torch as _torch

    try:
        payload = _torch.load(file_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Unable to load feature bank {file_path}: {exc}.") from exc
    if not isinstance(payload, dict) or "features" not in payload or "metadata" not in payload:
        raise ValueError(f"Invalid feature bank payload: {file_path}.")
    metadata = payload["metadata"]
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError(
            "Saved bank settings do not match the current configuration "
            f"(bank: {file_path})."
        )
    bank = payload["features"]
    if not isinstance(bank, _torch.Tensor) or bank.ndim != 2:
        raise ValueError(f"Feature bank has an unexpected shape: {file_path}.")
    expected_rows = len(metadata.get("train_paths", [])) * (
        metadata.get("patch_grid", [0, 0])[0] * metadata.get("patch_grid", [0, 0])[1]
    )
    if expected_rows and tuple(bank.shape) != (
        expected_rows,
        metadata.get("embedding_dimension"),
    ):
        raise ValueError(f"Saved bank has an unexpected shape: {file_path}.")
    if not _torch.isfinite(bank).all().item():
        raise ValueError(f"Saved bank contains invalid features: {file_path}.")
    target = _torch.device(str(device)) if device is not None else get_bank_device()
    logger.info("Loaded feature bank (%s) from %s.", tuple(bank.shape), file_path)
    return bank.to(target), metadata


def build_feature_bank(
    category: str,
    train_paths: list[str | Path],
    extractor: DINOv2FeatureExtractor,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
    bank_device: torch.device | str | None = None,
    save_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> tuple[torch.Tensor, dict]:
    """Build the normal bank for *category* from its ``train/good`` images.

    Args:
        category: MVTec category name.
        train_paths: image files from ``train/good`` only.
        extractor: :class:`DINOv2FeatureExtractor` instance.
        batch_size: images per forward pass.
        bank_device: device holding the bank (notebook rule by default).
        save_path: optional ``.pt`` destination.
        dataset_root: if given, stored ``train_paths`` are relativized to it.

    Returns:
        ``(bank, metadata)`` with bank shape
        ``(num_train * num_patches, embedding_dim)``.
    """
    validate_category(category)
    if not train_paths:
        raise ValueError(f"{category}: no train/good images provided.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    import torch as _torch

    target_device = (
        _torch.device(str(bank_device)) if bank_device is not None else get_bank_device()
    )
    resolved = [Path(p) for p in train_paths]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Training image not found: {path}.")

    # Extract in batches. The first batch reveals the embedding dimension
    # (as in the notebook), then the full bank is allocated and filled.
    pending = list(resolved)
    bank: torch.Tensor | None = None
    offset = 0
    embedding_dimension = 0
    patch_count = int(extractor.patch_count)
    while pending:
        chunk = pending[:batch_size]
        pending = pending[batch_size:]
        batch = _torch.stack([image_tensor(p) for p in chunk])
        batch_features = extractor.extract_batch(batch)  # (B, P, D)
        features = batch_features.reshape(-1, batch_features.shape[-1])  # (B*P, D)
        if bank is None:
            embedding_dimension = int(features.shape[1])
            number_of_features = len(resolved) * patch_count
            estimated_gib = number_of_features * embedding_dimension * 4 / (1024**3)
            logger.info(
                "%s: %s normal patches, %.3f GiB.", category, f"{number_of_features:,}", estimated_gib
            )
            if estimated_gib > MAX_BANK_GIB:
                raise MemoryError("Feature bank exceeds MAX_BANK_GIB. Review available memory.")
            bank = _torch.empty(
                (number_of_features, embedding_dimension),
                dtype=_torch.float32,
                device=target_device,
            )
        next_offset = offset + len(features)
        bank[offset:next_offset] = features.to(target_device)
        offset = next_offset

    assert bank is not None  # train_paths is non-empty, loop ran at least once
    if offset != len(bank) or not _torch.isfinite(bank).all().item():
        raise ValueError("The normal feature bank is incomplete or contains invalid values.")
    logger.info("Normal bank shape: %s.", tuple(bank.shape))

    if dataset_root is not None:
        root = Path(dataset_root)
        stored_paths = [str(Path(p).relative_to(root)) for p in resolved]
    else:
        stored_paths = [str(p) for p in resolved]
    metadata = bank_metadata(
        category,
        embedding_dimension=embedding_dimension,
        patch_grid=tuple(extractor.patch_grid),
        padding=(int(extractor.pad_right), int(extractor.pad_bottom)),
        train_paths=stored_paths,
    )
    if save_path is not None:
        save_feature_bank(save_path, bank, metadata)
    return bank, metadata
