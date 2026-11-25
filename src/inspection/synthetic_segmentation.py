"""One-class synthetic-defect segmentation experiment."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def _conv_block(channels_in: int, channels_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, padding=1),
        nn.GroupNorm(4, channels_out),
        nn.GELU(),
        nn.Conv2d(channels_out, channels_out, 3, padding=1),
        nn.GroupNorm(4, channels_out),
        nn.GELU(),
    )


class _SyntheticSegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = _conv_block(3, 16)
        self.enc2 = _conv_block(16, 32)
        self.enc3 = _conv_block(32, 64)
        self.bottleneck = _conv_block(64, 96)
        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = _conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = _conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = _conv_block(32, 16)
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


def generate_synthetic_anomaly(base: np.ndarray, donor: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Blend a donor texture into a few irregular small regions."""

    rng = np.random.default_rng(seed)
    size = int(base.shape[0])
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        center = (int(rng.integers(size // 8, size * 7 // 8)), int(rng.integers(size // 8, size * 7 // 8)))
        axes = (int(rng.integers(max(2, size // 80), max(4, size // 12))), int(rng.integers(max(2, size // 160), max(4, size // 30))))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    if rng.random() < 0.6:
        start = (int(rng.integers(size // 8, size * 7 // 8)), int(rng.integers(size // 8, size * 7 // 8)))
        end = (int(np.clip(start[0] + rng.integers(-size // 3, size // 3), 0, size - 1)), int(np.clip(start[1] + rng.integers(-size // 3, size // 3), 0, size - 1)))
        cv2.line(mask, start, end, 255, int(rng.integers(1, max(2, size // 80))))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(rng.uniform(0.4, 1.5))) / 255.0
    donor = cv2.GaussianBlur(donor, (0, 0), sigmaX=float(rng.uniform(0.2, 1.0)))
    alpha = mask[..., None].astype(np.float32) * float(rng.uniform(0.45, 0.95))
    if rng.random() < 0.5:
        donor = np.clip(donor.astype(np.float32) * float(rng.uniform(0.25, 0.75)), 0, 255)
    else:
        donor = np.clip(donor.astype(np.float32) * float(rng.uniform(1.1, 1.8)), 0, 255)
    corrupted = np.clip(base.astype(np.float32) * (1.0 - alpha) + donor.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return corrupted, (mask >= 0.20).astype(np.float32)


class SyntheticDefectSegmenter:
    """Tiny one-class segmentation model used only for a controlled ablation."""

    def __init__(self, model: _SyntheticSegNet, image_size: int = 224) -> None:
        self.model = model.to("cpu").eval()
        self.image_size = int(image_size)

    @classmethod
    def fit(
        cls,
        good_crops: list[np.ndarray],
        image_size: int = 224,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
    ) -> "SyntheticDefectSegmenter":
        if len(good_crops) < 2:
            raise ValueError("at least two GOOD crops are required")
        normalized = [cv2.resize(np.asarray(crop), (image_size, image_size), interpolation=cv2.INTER_AREA) for crop in good_crops]
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index, base in enumerate(normalized):
            donor = normalized[(index + 1) % len(normalized)]
            corrupted, mask = generate_synthetic_anomaly(base, donor, 42 + index)
            inputs.append(corrupted.transpose(2, 0, 1).astype(np.float32) / 255.0)
            targets.append(mask[None].astype(np.float32))
            inputs.append(base.transpose(2, 0, 1).astype(np.float32) / 255.0)
            targets.append(np.zeros((1, image_size, image_size), dtype=np.float32))
        input_tensor = torch.from_numpy(np.stack(inputs))
        target_tensor = torch.from_numpy(np.stack(targets))
        positives = float(target_tensor.sum())
        negatives = float(target_tensor.numel() - positives)
        torch.manual_seed(42)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = _SyntheticSegNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(20.0, negatives / max(positives, 1.0))], device=device))
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
                loss = loss_fn(logits, batch_targets) + dice
                loss.backward()
                optimizer.step()
        return cls(model.to("cpu").eval(), image_size)

    def predict_map(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(np.asarray(crop), (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(tensor)).squeeze().numpy()
        return cv2.resize(probability, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"state_dict": self.model.state_dict(), "image_size": self.image_size}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticDefectSegmenter":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = _SyntheticSegNet()
        model.load_state_dict(payload["state_dict"])
        return cls(model, int(payload.get("image_size", 224)))
