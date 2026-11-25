"""Small supervised patch-mask refiner for the screw pilot experiment."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def patch_inputs(
    dino_features: np.ndarray,
    heatmap: np.ndarray,
    side: int,
    feature_mode: str = "default",
) -> np.ndarray:
    """Join frozen DINO patch features with anomaly scores and optional local context."""

    features = np.asarray(dino_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != side * side:
        raise ValueError("DINO features must match the square patch grid")
    scores = cv2.resize(np.asarray(heatmap, dtype=np.float32), (side, side), interpolation=cv2.INTER_AREA)
    if feature_mode == "default":
        return np.concatenate([features, scores.reshape(-1, 1)], axis=1)
    if feature_mode != "spatial":
        raise ValueError(f"unknown patch feature mode: {feature_mode}")
    smooth = cv2.GaussianBlur(scores, (3, 3), 0)
    local_max = cv2.dilate(scores, np.ones((3, 3), np.uint8))
    local_min = cv2.erode(scores, np.ones((3, 3), np.uint8))
    gradient = scores - smooth
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    coordinates = np.stack((xx / max(side - 1, 1), yy / max(side - 1, 1)), axis=-1)
    context = np.concatenate(
        (
            scores[..., None],
            smooth[..., None],
            local_max[..., None],
            local_min[..., None],
            gradient[..., None],
            coordinates,
        ),
        axis=-1,
    ).reshape(-1, 7)
    return np.concatenate([features, context], axis=1)


class SupervisedPatchMaskRefiner:
    """Small patch classifier trained only from object-level pixel masks."""

    def __init__(self, classifier: object, feature_mode: str = "default") -> None:
        self.classifier = classifier
        self.feature_mode = feature_mode

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        model: str = "logistic",
        feature_mode: str = "default",
    ) -> "SupervisedPatchMaskRefiner":
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("training labels must contain both normal and defect patches")
        features = np.asarray(features, dtype=np.float32)
        if model == "logistic":
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        elif model == "mlp":
            rng = np.random.default_rng(42)
            positive = np.flatnonzero(labels == 1)
            negative = np.flatnonzero(labels == 0)
            if len(positive) < len(negative):
                positive = rng.choice(positive, size=len(negative), replace=True)
            indices = np.concatenate([negative, positive])
            rng.shuffle(indices)
            features, labels = features[indices], labels[indices]
            classifier = MLPClassifier(
                hidden_layer_sizes=(64,), alpha=1e-3, batch_size="auto",
                learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=5, random_state=42,
            )
        else:
            raise ValueError(f"unknown patch refiner model: {model}")
        classifier.fit(features, labels)
        return cls(classifier, feature_mode)

    @classmethod
    def fit_spatial_decoder(
        cls,
        feature_maps: list[np.ndarray],
        label_maps: list[np.ndarray],
        feature_mode: str = "default",
        epochs: int = 40,
        learning_rate: float = 1e-3,
    ) -> "SupervisedPatchMaskRefiner":
        """Fit a compact spatial decoder over complete patch grids.

        The logistic/MLP refiners see one patch at a time.  This decoder keeps
        the square grid intact and lets neighbouring patches contribute to the
        prediction, while still using frozen upstream features only.
        """

        if not feature_maps or len(feature_maps) != len(label_maps):
            raise ValueError("spatial decoder needs equally sized non-empty feature and label lists")
        first = np.asarray(feature_maps[0], dtype=np.float32)
        if first.ndim != 2:
            raise ValueError("spatial decoder features must be [patch, channel]")
        side = int(round(np.sqrt(first.shape[0])))
        if side * side != first.shape[0]:
            raise ValueError("spatial decoder features must form a square patch grid")
        tensors: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for features, labels in zip(feature_maps, label_maps):
            features = np.asarray(features, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32).reshape(-1)
            if features.shape != first.shape or labels.shape != (side * side,):
                raise ValueError("all spatial decoder samples must share the same grid and channels")
            tensors.append(features.reshape(side, side, -1).transpose(2, 0, 1))
            targets.append(labels.reshape(1, side, side))
        inputs = np.stack(tensors).astype(np.float32)
        targets_array = np.stack(targets).astype(np.float32)
        torch.manual_seed(42)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = _SpatialDecoder(inputs.shape[1]).to(device)
        input_tensor = torch.from_numpy(inputs).to(device)
        target_tensor = torch.from_numpy(targets_array).to(device)
        positives = float(targets_array.sum())
        negatives = float(targets_array.size - positives)
        if positives <= 0 or negatives <= 0:
            raise ValueError("spatial decoder labels must contain both classes")
        pos_weight = torch.tensor([min(30.0, negatives / positives)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        model.train()
        for _ in range(max(1, int(epochs))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)
            bce = loss_fn(logits, target_tensor)
            intersection = (probabilities * target_tensor).sum(dim=(1, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (probabilities.sum(dim=(1, 2, 3)) + target_tensor.sum(dim=(1, 2, 3)) + 1.0)).mean()
            loss = bce + dice
            loss.backward()
            optimizer.step()
        model = model.to("cpu").eval()
        classifier = _SpatialDecoderClassifier(model, inputs.shape[1], side)
        return cls(classifier, feature_mode)

    def predict_map(self, features: np.ndarray, side: int, output_size: tuple[int, int]) -> np.ndarray:
        probabilities = self.classifier.predict_proba(np.asarray(features, dtype=np.float32))[:, 1].reshape(side, side)
        return cv2.resize(probabilities, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {"classifier": self.classifier, "feature_mode": self.feature_mode},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedPatchMaskRefiner":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "classifier" in payload:
            classifier = payload["classifier"]
            feature_mode = str(payload.get("feature_mode", "default"))
        else:
            classifier = payload
            feature_mode = "default"
        if not hasattr(classifier, "predict_proba"):
            raise ValueError("invalid supervised patch refiner artifact")
        return cls(classifier, feature_mode)


class _SpatialDecoder(nn.Module):
    """Small fully-convolutional decoder for patch-grid mask refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = min(64, max(16, channels // 8))
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, max(8, hidden // 2), kernel_size=3, padding=1),
            nn.GroupNorm(1, max(8, hidden // 2)),
            nn.GELU(),
            nn.Conv2d(max(8, hidden // 2), 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _SpatialDecoderClassifier:
    """Pickle-friendly probability adapter around the convolutional decoder."""

    def __init__(self, model: _SpatialDecoder, channels: int, side: int) -> None:
        self.model = model
        self.channels = int(channels)
        self.side = int(side)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape != (self.side * self.side, self.channels):
            raise ValueError("spatial decoder input shape does not match the trained patch grid")
        tensor = torch.from_numpy(values.reshape(self.side, self.side, self.channels).transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(tensor)).squeeze(0).squeeze(0).numpy().reshape(-1)
        return np.stack((1.0 - probability, probability), axis=1).astype(np.float32)
