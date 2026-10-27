"""Lazy Ultralytics YOLO adapter for complete-egg detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EggDetection:
    """One complete-egg bounding box in source-image pixel coordinates."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YoloEggDetector:
    """Thin wrapper; importing this module does not require Ultralytics."""

    def __init__(self, weights: str | Path, device: str | None = None):
        self.weights = str(weights)
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "YOLO demo requires the optional 'ultralytics' package; install requirements-egg-demo.txt"
                ) from error
            self._model = YOLO(self.weights)
        return self._model

    def train(self, data_yaml: str | Path, **kwargs: Any) -> Any:
        """Train YOLO; kwargs are passed through to Ultralytics unchanged."""

        return self._load().train(data=str(data_yaml), **kwargs)

    def evaluate(self, data_yaml: str | Path, **kwargs: Any) -> Any:
        """Run Ultralytics validation; its result contains precision/recall/mAP."""

        return self._load().val(data=str(data_yaml), **kwargs)

    def detect(self, image: str | Path | np.ndarray, confidence: float = 0.5) -> list[EggDetection]:
        if confidence < 0.5:
            raise ValueError("Egg detection confidence must be at least 0.5.")
        results = self._load().predict(source=image, conf=confidence, device=self.device, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {})
        detections = []
        for box, score, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            if float(score) <= confidence:
                continue
            index = int(class_id)
            detections.append(EggDetection(tuple(float(value) for value in box), float(score), index, str(names.get(index, index))))
        return detections


def crop_detection(
    image: Image.Image,
    detection: EggDetection,
    margin: float = 0.05,
) -> Image.Image:
    """Crop one detected egg with a small proportional context margin."""

    if margin < 0:
        raise ValueError("margin must be non-negative")
    width, height = image.size
    left, top, right, bottom = detection.xyxy
    box_width, box_height = right - left, bottom - top
    left = max(0, int(round(left - box_width * margin)))
    top = max(0, int(round(top - box_height * margin)))
    right = min(width, int(round(right + box_width * margin)))
    bottom = min(height, int(round(bottom + box_height * margin)))
    if right <= left or bottom <= top:
        raise ValueError("detection box does not overlap image")
    return image.crop((left, top, right, bottom)).convert("RGB")
