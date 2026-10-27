"""End-to-end YOLO complete-egg detection followed by PaDiM scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .anomaly import AnomalyScore, PaDiMReference
from .detection import EggDetection, YoloEggDetector, crop_detection


@dataclass(frozen=True)
class EggInference:
    detection: EggDetection
    anomaly: AnomalyScore


class EggAnomalyPipeline:
    """Keep detection and anomaly responsibilities separate in one reusable call."""

    def __init__(self, detector: YoloEggDetector, reference: PaDiMReference, confidence: float = 0.5, crop_margin: float = 0.05):
        self.detector = detector
        self.reference = reference
        self.confidence = confidence
        self.crop_margin = crop_margin

    def infer(self, image_path: str | Path) -> list[EggInference]:
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            detections = self.detector.detect(image_path, self.confidence)
            return [
                EggInference(detection, self.reference.score(crop_detection(image, detection, self.crop_margin)))
                for detection in detections
            ]

    def infer_directory(self, directory: str | Path) -> dict[str, list[EggInference]]:
        paths = sorted(path for path in Path(directory).rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return {str(path): self.infer(path) for path in paths}
