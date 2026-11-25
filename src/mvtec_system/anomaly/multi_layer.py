"""Small, reusable feature transforms shared by PatchCore and PaDiM."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import torch
import torch.nn.functional as F

from ..core.dinov2 import DINOv2Features, SharedDINOv2, image_tensor


@dataclass(frozen=True)
class MultiLayerConfig:
    """Configurable settings for the two multi-layer detector variants."""

    # DINOv2 exposes transformer blocks with zero-based indices 0..11.
    layers: tuple[int, ...] = (5, 8, 11)
    neighborhood_size: int = 3
    coreset_ratio: float = 0.10
    coreset_max_points: int = 50_000
    reduced_dim: int = 100
    covariance_regularization: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("layers must be a sorted, non-empty tuple")
        if self.neighborhood_size < 1 or self.neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be a positive odd number")
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise ValueError("coreset_ratio must be in (0, 1]")
        if self.coreset_max_points < 1 or self.reduced_dim < 1:
            raise ValueError("coreset_max_points and reduced_dim must be positive")
        if self.covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fused_patch_features(
    features: DINOv2Features,
    layers: Iterable[int],
) -> torch.Tensor:
    """Concatenate selected layers into one normalized [B, patches, dims] tensor."""

    selected = tuple(int(layer) for layer in layers)
    available = features.layer_patch_embeddings
    if not available:
        raise ValueError("Multi-layer features are missing; configure feature_layers on DINOv2Config.")
    tensors = []
    expected_shape = None
    for layer in selected:
        tensor = available.get(layer)
        if tensor is None:
            raise KeyError(f"Requested layer {layer} is not available")
        if tensor.ndim != 3:
            raise ValueError("Layer features must have shape [batch, patches, dimension]")
        shape = tensor.shape[:2]
        if expected_shape is None:
            expected_shape = shape
        if shape != expected_shape:
            raise ValueError("Selected layer feature grids do not align")
        tensors.append(tensor.float())
    fused = torch.cat(tensors, dim=-1)
    return F.normalize(fused, dim=-1)


def local_neighborhood_aggregate(
    descriptors: torch.Tensor,
    patch_grid: tuple[int, int] | list[int],
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """Average a patch descriptor with its local spatial neighborhood."""

    if descriptors.ndim != 3:
        raise ValueError("descriptors must have shape [batch, patches, dimension]")
    height, width = (int(value) for value in patch_grid)
    if descriptors.shape[1] != height * width:
        raise ValueError("descriptor patch count does not match patch_grid")
    if neighborhood_size < 1 or neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be a positive odd number")
    batch, _, dimension = descriptors.shape
    grid = descriptors.transpose(1, 2).reshape(batch, dimension, height, width)
    padding = neighborhood_size // 2
    aggregated = F.avg_pool2d(grid, neighborhood_size, stride=1, padding=padding)
    flattened = aggregated.reshape(batch, dimension, -1).transpose(1, 2)
    return F.normalize(flattened, dim=-1)


def select_coreset(
    descriptors: torch.Tensor,
    ratio: float = 0.10,
    max_points: int = 50_000,
    seed: int = 42,
) -> torch.Tensor:
    """Select a deterministic representative normal memory subset.

    A seeded uniform subset is the scalable fallback for very large banks.  A
    capped greedy farthest-point pass is used for smaller candidate sets so
    the result remains a useful PatchCore-style representative memory.
    """

    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("descriptors must be a non-empty [points, dimension] tensor")
    if not 0.0 < ratio <= 1.0 or max_points < 1:
        raise ValueError("ratio must be in (0, 1] and max_points must be positive")
    point_count = descriptors.shape[0]
    target = min(point_count, max(1, round(point_count * ratio)), max_points)
    if target == point_count:
        return descriptors.detach().cpu().float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate_count = min(point_count, max(target * 4, 2_048))
    candidate_indices = torch.randperm(point_count, generator=generator)[:candidate_count]
    candidates = descriptors.detach().cpu().float()[candidate_indices]
    if target > 5_000 or candidate_count > 20_000:
        # ponytail: bounded uniform memory is the safe fallback; swap for a
        # scalable k-center implementation if coreset quality needs tuning.
        chosen = torch.randperm(candidate_count, generator=generator)[:target]
        return candidates[chosen]
    chosen_indices = [0]
    minimum_distance = torch.cdist(candidates[:1], candidates).squeeze(0)
    for _ in range(1, target):
        next_index = int(minimum_distance.argmax().item())
        chosen_indices.append(next_index)
        distance = torch.cdist(candidates[next_index : next_index + 1], candidates).squeeze(0)
        minimum_distance = torch.minimum(minimum_distance, distance)
    return candidates[torch.tensor(chosen_indices, dtype=torch.long)]


def extract_fused_descriptors(
    paths: list,
    encoder: SharedDINOv2,
    layers: tuple[int, ...],
    batch_size: int = 8,
    local_neighborhood_size: int | None = None,
) -> torch.Tensor:
    """Extract fused descriptors from image paths without retaining activations."""

    if not paths or batch_size < 1:
        raise ValueError("paths must be non-empty and batch_size must be positive")
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = torch.stack(
            [image_tensor(path, encoder.config) for path in paths[start : start + batch_size]]
        )
        features = encoder.forward(images)
        fused = fused_patch_features(features, layers)
        if local_neighborhood_size is not None:
            fused = local_neighborhood_aggregate(
                fused, encoder.patch_grid, local_neighborhood_size
            )
        chunks.append(fused.detach().cpu())
    return torch.cat(chunks, dim=0)


def make_projection(input_dim: int, reduced_dim: int, seed: int = 42) -> torch.Tensor:
    """Create a deterministic orthonormal random projection matrix."""

    if input_dim < 1 or reduced_dim < 1:
        raise ValueError("projection dimensions must be positive")
    dimension = min(input_dim, reduced_dim)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_matrix = torch.randn(input_dim, dimension, generator=generator)
    projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return projection.float()


def fit_padim_statistics(
    descriptors: torch.Tensor,
    reduced_dim: int = 100,
    covariance_regularization: float = 0.01,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one Gaussian mean and inverse covariance per spatial position."""

    if descriptors.ndim != 3 or descriptors.shape[0] < 2:
        raise ValueError("PaDiM requires [images, patches, dimensions] and at least two images")
    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be positive")
    images, patches, dimension = descriptors.shape
    projection = make_projection(dimension, reduced_dim, seed)
    reduced = descriptors.float() @ projection
    means = reduced.mean(dim=0)
    centered = reduced - means
    covariance = torch.einsum("npd,npe->pde", centered, centered) / max(images - 1, 1)
    identity = torch.eye(projection.shape[1], dtype=torch.float32)
    covariance = covariance + covariance_regularization * identity.unsqueeze(0)
    inverse_covariance = torch.linalg.pinv(covariance)
    return means.float(), inverse_covariance.float(), projection.float()


def padim_scores(
    descriptors: torch.Tensor,
    means: torch.Tensor,
    inverse_covariance: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Return Mahalanobis scores with shape [batch, patches]."""

    if descriptors.ndim != 3:
        raise ValueError("descriptors must have shape [batch, patches, dimensions]")
    # Artifacts are intentionally loaded on CPU; inference features may live on MPS/CUDA.
    device = descriptors.device
    projection = projection.to(device=device, dtype=torch.float32)
    means = means.to(device=device, dtype=torch.float32)
    inverse_covariance = inverse_covariance.to(device=device, dtype=torch.float32)
    reduced = descriptors.float() @ projection
    if reduced.shape[1:] != means.shape:
        raise ValueError("PaDiM statistics do not match descriptor shape")
    difference = reduced - means
    distances = torch.einsum(
        "npd,pde,npe->np", difference, inverse_covariance, difference
    ).clamp_min(0.0)
    return distances.sqrt()
