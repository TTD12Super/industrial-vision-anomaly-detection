"""The shared, frozen DINOv2 encoder used by the MVTec pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DINOv2Config:
    """Settings that must stay aligned with the existing notebook baseline."""

    model_name: str = "dinov2_vitb14"
    model_revision: str = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    patch_feature_key: str = "x_norm_patchtokens"
    global_feature_key: str = "x_norm_clstoken"
    image_size: tuple[int, int] = (256, 256)  # height, width
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    l2_normalize_patches: bool = True

    @property
    def repository(self) -> str:
        return f"facebookresearch/dinov2:{self.model_revision}"


@dataclass(frozen=True)
class DINOv2Features:
    """One shared forward pass: global features for classification and patches for kNN."""

    global_embedding: torch.Tensor
    patch_embeddings: torch.Tensor


def select_device() -> torch.device:
    """Match the notebook's CUDA → MPS → CPU preference."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_tensor(path: str | Path, config: DINOv2Config | None = None) -> torch.Tensor:
    """Read one processed RGB image using the notebook's normalization contract."""

    config = config or DINOv2Config()
    path = Path(path)
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Expected an RGB image: {path}")
        expected_size = (config.image_size[1], config.image_size[0])
        if image.size != expected_size:
            raise ValueError(f"Expected {expected_size}, got {image.size}: {path}")
        pixels = np.array(image, copy=True)

    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class SharedDINOv2:
    """Frozen DINOv2 wrapper that exposes global and spatial patch embeddings."""

    def __init__(
        self,
        config: DINOv2Config | None = None,
        artifact_root: str | Path = "artifacts/dinov2_knn",
        device: str | torch.device | None = None,
        backbone: Any | None = None,
    ) -> None:
        self.config = config or DINOv2Config()
        self.artifact_root = Path(artifact_root)
        self.device = torch.device(device) if device is not None else select_device()
        self.backbone = backbone or self._load_backbone()
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.patch_size = self._read_patch_size()
        height, width = self.config.image_size
        pad_bottom = (-height) % self.patch_size[0]
        pad_right = (-width) % self.patch_size[1]
        self.padding = (pad_bottom, pad_right)
        self.padded_size = (height + pad_bottom, width + pad_right)
        self.patch_grid = (
            self.padded_size[0] // self.patch_size[0],
            self.padded_size[1] // self.patch_size[1],
        )
        self.patch_count = self.patch_grid[0] * self.patch_grid[1]

    def _load_backbone(self) -> Any:
        hub_dir = self.artifact_root / "torch_hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
        return torch.hub.load(
            self.config.repository,
            self.config.model_name,
            pretrained=True,
            trust_repo=True,
        )

    def _read_patch_size(self) -> tuple[int, int]:
        patch_size = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", None)
        if patch_size is None:
            raise AttributeError("DINOv2 backbone does not expose patch_embed.patch_size.")
        if isinstance(patch_size, int):
            return (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError(f"Unexpected patch size: {patch_size}")
        return (int(patch_size[0]), int(patch_size[1]))

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [batch, 3, height, width], got {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != self.config.image_size:
            raise ValueError(
                f"Expected image size {self.config.image_size}, got {tuple(images.shape[-2:])}"
            )
        if not torch.isfinite(images).all().item():
            raise ValueError("Input images contain non-finite values.")
        images = images.to(self.device)
        pad_bottom, pad_right = self.padding
        return F.pad(images, (0, pad_right, 0, pad_bottom), mode="replicate")

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> DINOv2Features:
        """Return both representations from one frozen DINOv2 forward pass."""

        padded_images = self._prepare_images(images)
        outputs = self.backbone.forward_features(padded_images)
        patch_embeddings = self._feature(outputs, self.config.patch_feature_key)
        global_embedding = self._feature(outputs, self.config.global_feature_key)

        if patch_embeddings.ndim != 3 or patch_embeddings.shape[1] != self.patch_count:
            raise ValueError(f"Unexpected patch feature shape: {tuple(patch_embeddings.shape)}")
        if global_embedding.ndim != 2 or global_embedding.shape[0] != images.shape[0]:
            raise ValueError(f"Unexpected global feature shape: {tuple(global_embedding.shape)}")
        if not torch.isfinite(patch_embeddings).all().item():
            raise ValueError("DINOv2 produced non-finite patch features.")
        if not torch.isfinite(global_embedding).all().item():
            raise ValueError("DINOv2 produced non-finite global features.")

        patch_embeddings = patch_embeddings.float()
        global_embedding = global_embedding.float()
        if self.config.l2_normalize_patches:
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        return DINOv2Features(global_embedding, patch_embeddings)

    @staticmethod
    def _feature(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in outputs:
            raise KeyError(f"The backbone did not return {key}.")
        return outputs[key]

    def metadata(self) -> dict[str, Any]:
        """Return serializable settings for artifact compatibility checks."""

        return {
            "model_name": self.config.model_name,
            "model_revision": self.config.model_revision,
            "patch_feature_key": self.config.patch_feature_key,
            "global_feature_key": self.config.global_feature_key,
            "image_size": list(self.config.image_size),
            "mean": list(self.config.mean),
            "std": list(self.config.std),
            "l2_normalize_patches": self.config.l2_normalize_patches,
            "patch_size": list(self.patch_size),
            "padding": list(self.padding),
            "padded_size": list(self.padded_size),
            "patch_grid": list(self.patch_grid),
            "patch_count": self.patch_count,
            "embedding_dimension": self.backbone.embed_dim,
        }
