"""DINOv2 + PaDiM reference building and scoring for cropped products."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from mvtec_system.anomaly.detectors import PaDiMDetector
from mvtec_system.anomaly.multi_layer import MultiLayerConfig
from mvtec_system.anomaly.refinement import refine_binary_mask
from mvtec_system.anomaly.thresholds import calibrate_category_image_thresholds, calibrate_quantile_threshold
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor


@dataclass(frozen=True)
class AnomalyScore:
    score: float
    label: str
    threshold: float
    anomaly_map: np.ndarray
    binary_mask: np.ndarray


def _pil_tensor(image: Image.Image, config: DINOv2Config) -> torch.Tensor:
    image = image.convert("RGB").resize((config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _predict_paths(detector: PaDiMDetector, encoder: SharedDINOv2, paths: Sequence[Path]) -> list:
    results = []
    for path in paths:
        with torch.inference_mode():
            features = encoder.forward(torch.stack([image_tensor(path, encoder.config)]))
        results.append(detector.predict(features))
    return results


def build_padim_reference(
    train_paths: Sequence[str | Path],
    validation_paths: Sequence[str | Path],
    artifact_dir: str | Path,
    encoder_artifact_root: str | Path,
    image_size: int = 256,
    feature_layers: tuple[int, ...] = (5, 8, 11),
    neighborhood_size: int = 3,
    reduced_dim: int = 100,
    covariance_regularization: float = 0.01,
    target_acceptance_rate: float = 0.99,
    batch_size: int = 8,
    seed: int = 42,
) -> dict:
    """Fit PaDiM on normal training crops and calibrate on normal validation crops."""

    train_paths = [Path(path) for path in train_paths]
    validation_paths = [Path(path) for path in validation_paths]
    if len(train_paths) < 2 or not validation_paths:
        raise ValueError("PaDiM reference needs at least two train images and one validation image")
    config = MultiLayerConfig(
        layers=feature_layers,
        neighborhood_size=neighborhood_size,
        reduced_dim=reduced_dim,
        covariance_regularization=covariance_regularization,
        seed=seed,
    )
    encoder = SharedDINOv2(
        DINOv2Config(image_size=(image_size, image_size), feature_layers=feature_layers),
        artifact_root=encoder_artifact_root,
        device="cpu",
    )
    detector = PaDiMDetector(config).fit("egg", train_paths, encoder, batch_size)
    validation_results = _predict_paths(detector, encoder, validation_paths)
    image_threshold = calibrate_category_image_thresholds(
        {"egg": [result.image_score for result in validation_results]}, target_acceptance_rate
    )
    pixel_threshold = calibrate_quantile_threshold(
        np.concatenate([result.anomaly_map.reshape(-1) for result in validation_results]),
        target_acceptance_rate,
    )
    artifact_dir = Path(artifact_dir)
    detector.save(artifact_dir / "detector")
    metadata = {
        "product": "egg",
        "train_images": [str(path) for path in train_paths],
        "validation_images": [str(path) for path in validation_paths],
        "image_size": image_size,
        "detector_config": config.to_dict(),
        "image_thresholds": image_threshold,
        "pixel_threshold": pixel_threshold,
        "protocol": "normal-only PaDiM reference; thresholds calibrated from normal validation crops; no bad images used for fitting or threshold selection",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class PaDiMReference:
    """Load one persisted reference and score PIL crops."""

    def __init__(self, artifact_dir: str | Path, encoder_artifact_root: str | Path, device: str = "cpu"):
        artifact_dir = Path(artifact_dir)
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        detector_values = dict(metadata["detector_config"])
        detector_values["layers"] = tuple(detector_values["layers"])
        config = MultiLayerConfig(**detector_values)
        self.encoder = SharedDINOv2(
            DINOv2Config(image_size=(metadata["image_size"], metadata["image_size"]), feature_layers=config.layers),
            artifact_root=encoder_artifact_root,
            device=device,
        )
        self.detector = PaDiMDetector(config).load(artifact_dir / "detector")
        self.image_threshold = float(metadata["image_thresholds"]["categories"]["egg"]["threshold"])
        self.pixel_threshold = float(metadata["pixel_threshold"]["threshold"])

    def score(self, image: Image.Image, min_component_area: int = 9, opening_kernel_size: int = 3) -> AnomalyScore:
        with torch.inference_mode():
            features = self.encoder.forward(torch.stack([_pil_tensor(image, self.encoder.config)]))
        result = self.detector.predict(features)
        mask = refine_binary_mask(result.anomaly_map >= self.pixel_threshold, min_component_area, opening_kernel_size)
        return AnomalyScore(
            score=result.image_score,
            label="DEFECT" if result.image_score >= self.image_threshold else "GOOD",
            threshold=self.image_threshold,
            anomaly_map=result.anomaly_map,
            binary_mask=mask,
        )


def evaluate_image_level(
    reference: PaDiMReference,
    good_paths: Sequence[str | Path],
    bad_paths: Sequence[str | Path],
) -> dict[str, float | int | str]:
    """Evaluate Good/Bad scores only; no pixel metric is fabricated without masks."""

    scores: list[float] = []
    labels: list[int] = []
    for label, paths in ((0, good_paths), (1, bad_paths)):
        for path in paths:
            with Image.open(path) as image:
                scores.append(reference.score(image).score)
            labels.append(label)
    if len(set(labels)) < 2:
        raise ValueError("Good and bad test images are both required")
    return {
        "test_images": len(labels),
        "good_images": len(good_paths),
        "bad_images": len(bad_paths),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "pixel_metrics": "not available: quality dataset has no verified defect masks",
    }
