"""Shared configuration for the DINOv2 + exact-kNN inference pipeline.

Reference (source of truth): ``notebooks/02_first_model.ipynb``.
Every constant below mirrors the notebook baseline exactly:

* backbone ``dinov2_vitb14`` at pinned revision ``7764ea0...``
* feature key ``x_norm_patchtokens`` (final-layer patch tokens, no CLS)
* 256x256 RGB input, ImageNet normalization
* L2 normalization enabled, exact kNN with k=1 (Euclidean)
* image score = maximum patch score, no Normal/Anomaly threshold

Do not change these values without an explicit follow-up request.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is optional at import time (unit tests may run without it)
    from torch import device as TorchDevice

# ---------------------------------------------------------------------------
# Model (notebook cells: "Choose the model settings", "Load and freeze DINOv2")
# ---------------------------------------------------------------------------
MODEL_NAME = "dinov2_vitb14"
MODEL_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_REPOSITORY = f"facebookresearch/dinov2:{MODEL_REVISION}"
FEATURE_KEY = "x_norm_patchtokens"

# ---------------------------------------------------------------------------
# Preprocessing (notebooks 01 + 02)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256
PROCESSED_SIZE = (256, 256)  # (height, width); notebook: 256x256 RGB PNGs

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Distance / aggregation (notebook: kNN + "maximum patch distance")
# ---------------------------------------------------------------------------
L2_NORMALIZE_FEATURES = True
K_NEIGHBORS = 1
DISTANCE_METRIC = "euclidean"  # torch.cdist(p=2); same ranking as cosine on unit vectors
IMAGE_AGGREGATION = "max"  # image score = maximum patch score

# ---------------------------------------------------------------------------
# Memory-bounded chunking (notebook: "Choose batch sizes and memory limits")
# ---------------------------------------------------------------------------
INFERENCE_BATCH_SIZE = 8
QUERY_CHUNK_SIZE = 512
BANK_CHUNK_SIZE = 16384
MAX_BANK_GIB = 4.0

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

# ---------------------------------------------------------------------------
# Categories (discovered from the processed MVTec AD folders, notebook cell
# "Discover the actual categories"; kept here as the single source of truth)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def validate_category(category: str) -> str:
    """Return the category if it is supported, otherwise raise ValueError."""
    if category not in CATEGORIES:
        raise ValueError(
            f"Unknown category: {category!r}. Supported categories: {', '.join(CATEGORIES)}."
        )
    return category


def get_device(prefer: str | None = None) -> TorchDevice | str:
    """Select the compute device, preferring CUDA -> MPS -> CPU.

    Mirrors the notebook cell "Select the computing devices". Falls back to
    the string ``"cpu"`` when torch is not installed (lets dependency-free
    unit tests import this module safely).
    """
    preference = (prefer or os.environ.get("INFERENCE_DEVICE", "")).strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if preference in ("cuda", "mps", "cpu"):
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_bank_device(device: TorchDevice | str | None = None) -> TorchDevice | str:
    """Device holding the normal feature bank during kNN search.

    The notebook keeps the bank on CUDA when CUDA is available, otherwise on
    CPU (MPS search runs on CPU because the bank lives on CPU there).
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    resolved = device if device is not None else get_device()
    if isinstance(resolved, str):
        resolved = torch.device(resolved)
    if resolved.type == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def default_bank_path(category: str, bank_dir: str | Path) -> Path:
    """Resolve ``<bank_dir>/<category>/feature_bank.pt`` for a category."""
    validate_category(category)
    return Path(bank_dir) / category / "feature_bank.pt"
