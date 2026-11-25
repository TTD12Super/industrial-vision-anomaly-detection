"""Small supervised RGB-plus-anomaly segmentation ablation."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.GroupNorm(4, channels_out),
            nn.GELU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.GroupNorm(4, channels_out),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _TinyUNet(nn.Module):
    """A deliberately small U-Net for a supervised ablation, not production."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(4, 16)
        self.enc2 = _ConvBlock(16, 32)
        self.enc3 = _ConvBlock(32, 64)
        self.bottleneck = _ConvBlock(64, 96)
        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = _ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = _ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = _ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.enc1(inputs)
        second = self.enc2(torch.max_pool2d(first, 2))
        third = self.enc3(torch.max_pool2d(second, 2))
        latent = self.bottleneck(torch.max_pool2d(third, 2))
        decoded = self.dec3(torch.cat((self.up3(latent), third), dim=1))
        decoded = self.dec2(torch.cat((self.up2(decoded), second), dim=1))
        decoded = self.dec1(torch.cat((self.up1(decoded), first), dim=1))
        return self.output(decoded)


class SupervisedRGBMaskRefiner:
    """RGB-plus-AnomalyCLIP supervised mask refiner used only for experiments."""

    def __init__(self, model: _TinyUNet, image_size: int = 224) -> None:
        self.model = model.to("cpu").eval()
        self.image_size = int(image_size)

    @classmethod
    def fit(
        cls,
        inputs: np.ndarray,
        targets: np.ndarray,
        image_size: int = 224,
        epochs: int = 25,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
    ) -> "SupervisedRGBMaskRefiner":
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if inputs.ndim != 4 or inputs.shape[1] != 4 or targets.shape != (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]):
            raise ValueError("RGB refiner expects [sample, 4, height, width] inputs and [sample, 1, height, width] targets")
        positives = float(targets.sum())
        negatives = float(targets.size - positives)
        if positives <= 0 or negatives <= 0:
            raise ValueError("RGB refiner targets must contain both classes")
        torch.manual_seed(42)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = _TinyUNet().to(device)
        input_tensor = torch.from_numpy(inputs)
        target_tensor = torch.from_numpy(targets)
        pos_weight = torch.tensor([min(50.0, negatives / positives)], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(42)
        model.train()
        for _ in range(max(1, int(epochs))):
            order = torch.randperm(len(input_tensor), generator=generator)
            for start in range(0, len(order), max(1, int(batch_size))):
                indices = order[start : start + max(1, int(batch_size))]
                batch_inputs = input_tensor[indices].to(device)
                batch_targets = target_tensor[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_inputs)
                probabilities = torch.sigmoid(logits)
                intersection = (probabilities * batch_targets).sum(dim=(1, 2, 3))
                dice = 1.0 - ((2.0 * intersection + 1.0) / (probabilities.sum(dim=(1, 2, 3)) + batch_targets.sum(dim=(1, 2, 3)) + 1.0)).mean()
                loss = bce(logits, batch_targets) + dice
                loss.backward()
                optimizer.step()
        return cls(model.to("cpu").eval(), image_size)

    @staticmethod
    def prepare_input(crop: np.ndarray, heatmap: np.ndarray, image_size: int = 224) -> np.ndarray:
        image = cv2.resize(np.asarray(crop), (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        score = cv2.resize(np.asarray(heatmap, dtype=np.float32), (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        return np.concatenate((image.transpose(2, 0, 1), score[None]), axis=0).astype(np.float32)

    @staticmethod
    def prepare_target(truth: np.ndarray, image_size: int = 224) -> np.ndarray:
        return cv2.resize(np.asarray(truth, dtype=np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)[None]

    def predict_map(self, crop: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(self.prepare_input(crop, heatmap, self.image_size)).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(inputs)).squeeze().numpy()
        return cv2.resize(probabilities, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"state_dict": self.model.state_dict(), "image_size": self.image_size}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedRGBMaskRefiner":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = _TinyUNet()
        model.load_state_dict(payload["state_dict"])
        return cls(model, int(payload.get("image_size", 224)))
