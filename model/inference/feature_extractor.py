"""Frozen DINOv2 patch-feature extraction.

Mirrors the notebook cells "Load and freeze DINOv2" through "Normalize bank
and test features identically":

* backbone loaded from the pinned ``MODEL_REPOSITORY``/``MODEL_NAME``
* ``eval()`` mode, all parameters frozen (``requires_grad=False``)
* ``forward_features`` output ``FEATURE_KEY`` (``x_norm_patchtokens``)
* patch count/shape validated against the observed patch grid
* finite-value checks, optional L2 normalization (enabled by default)

Model weights are downloaded/cached by ``torch.hub`` on first use and are
never committed to Git.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from .config import (
    FEATURE_KEY,
    L2_NORMALIZE_FEATURES,
    MODEL_NAME,
    MODEL_REPOSITORY,
    PROCESSED_SIZE,
    get_device,
)
from .preprocessing import (
    compute_patch_grid,
    compute_padding,
    image_tensor,
    pad_for_patch_embedding,
)
from .preprocessing import ImageLike

logger = logging.getLogger(__name__)


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2-normalize patch embeddings (baseline choice, applied to both sides)."""
    if L2_NORMALIZE_FEATURES:
        return F.normalize(features, p=2, dim=-1)
    return features


class DINOv2FeatureExtractor:
    """Frozen DINOv2 ViT-B/14 patch-token extractor."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        torch_hub_dir: str | None = None,
    ) -> None:
        self.device = torch.device(str(device)) if device is not None else get_device()
        if torch_hub_dir is not None:
            torch.hub.set_dir(torch_hub_dir)
        try:
            backbone = torch.hub.load(
                MODEL_REPOSITORY,
                MODEL_NAME,
                pretrained=True,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load {MODEL_NAME} from {MODEL_REPOSITORY}: {exc}. "
                "Check network access to the torch hub repository."
            ) from exc
        self.backbone = backbone.to(self.device)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        patch_size = self.backbone.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size: tuple[int, int] = (int(patch_size[0]), int(patch_size[1]))
        image_height, image_width = PROCESSED_SIZE
        pad_right, pad_bottom, padded_h, padded_w = compute_padding(
            image_height, image_width, self.patch_size[0], self.patch_size[1]
        )
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom
        self.padded_size: tuple[int, int] = (padded_h, padded_w)
        self.patch_grid: tuple[int, int] = compute_patch_grid(
            padded_h, padded_w, self.patch_size[0], self.patch_size[1]
        )
        self.patch_count = self.patch_grid[0] * self.patch_grid[1]
        self.embedding_dimension: int | None = None  # discovered on first forward
        logger.info(
            "Loaded %s on %s (patch_size=%s, grid=%s, patches=%d).",
            MODEL_NAME,
            self.device,
            self.patch_size,
            self.patch_grid,
            self.patch_count,
        )

    @property
    def is_frozen(self) -> bool:
        """True when no backbone parameter requires gradients."""
        return not any(p.requires_grad for p in self.backbone.parameters())

    @torch.inference_mode()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalized patch features for a ``(B, 3, 256, 256)`` batch.

        Returns ``(B, num_patches, embedding_dim)`` on CPU.
        """
        if images.ndim != 4 or images.shape[1:] != (3, *PROCESSED_SIZE):
            raise ValueError(
                f"Expected batch shape (B, 3, {PROCESSED_SIZE[0]}, {PROCESSED_SIZE[1]}), "
                f"got {tuple(images.shape)}."
            )
        padded = pad_for_patch_embedding(
            images.to(self.device), self.pad_right, self.pad_bottom
        )
        self.backbone.eval()
        outputs: dict[str, Any] = self.backbone.forward_features(padded)
        if FEATURE_KEY not in outputs:
            raise KeyError(f"The backbone did not return {FEATURE_KEY}.")
        features = outputs[FEATURE_KEY].float()
        if features.ndim != 3 or features.shape[1] != self.patch_count:
            raise ValueError(
                f"Unexpected patch feature shape: {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all().item():
            raise ValueError("DINOv2 produced non-finite patch features.")
        normalized = normalize_features(features)
        if self.embedding_dimension is None:
            self.embedding_dimension = int(normalized.shape[-1])
            logger.info("Observed embedding dimension: %d.", self.embedding_dimension)
        return normalized.detach().cpu()

    def extract(self, source: ImageLike | torch.Tensor) -> torch.Tensor:
        """Extract patch features for one image.

        Accepts a file path / PIL image / numpy array (preprocessed first)
        or an already-normalized ``(3, 256, 256)`` tensor.
        Returns ``(num_patches, embedding_dim)`` on CPU.
        """
        if isinstance(source, torch.Tensor):
            if source.ndim != 3:
                raise ValueError(
                    f"Expected a (3, H, W) image tensor, got {tuple(source.shape)}."
                )
            batch = source.unsqueeze(0)
        else:
            batch = image_tensor(source).unsqueeze(0)
        return self.extract_batch(batch)[0]
