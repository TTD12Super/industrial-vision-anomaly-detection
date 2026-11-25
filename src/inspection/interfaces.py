"""Small backend-neutral contracts for detection, recognition, quality and reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    corners: np.ndarray
    score: float


@dataclass(frozen=True)
class Classification:
    p_defect: float
    model_version: str


KNOWN_PRODUCT = "KNOWN_PRODUCT"
UNKNOWN_PRODUCT = "UNKNOWN_PRODUCT"
REVIEW_PRODUCT = "REVIEW_PRODUCT"


@dataclass(frozen=True)
class RecognitionResult:
    product_key: str | None
    score: float | None
    model_version: str
    status: str
    reason: str | None = None
    margin: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {KNOWN_PRODUCT, UNKNOWN_PRODUCT, REVIEW_PRODUCT}:
            raise ValueError(f"Unsupported product recognition status: {self.status}")
        if not self.model_version:
            raise ValueError("Product recognition model_version is required")
        if self.status == KNOWN_PRODUCT and not self.product_key:
            raise ValueError("KNOWN_PRODUCT requires product_key")
        if self.status == UNKNOWN_PRODUCT and self.product_key is not None:
            raise ValueError("UNKNOWN_PRODUCT must not carry a product_key")
        if self.status == REVIEW_PRODUCT and self.product_key is not None:
            raise ValueError("REVIEW_PRODUCT must not carry a product_key")
        if self.score is not None and not np.isfinite(float(self.score)):
            raise ValueError("Product recognition score must be finite")
        if self.margin is not None and not np.isfinite(float(self.margin)):
            raise ValueError("Product recognition margin must be finite")


@dataclass(frozen=True)
class Localization:
    heatmap: np.ndarray | None
    binary_mask: np.ndarray | None
    anomaly_score: float | None
    status: str
    reason: str | None = None
    model_version: str | None = None


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class BinaryQualityClassifier(Protocol):
    def classify(self, crop: np.ndarray) -> Classification: ...


class ProductRecognizer(Protocol):
    def recognize(self, crop: np.ndarray) -> RecognitionResult: ...


class Localizer(Protocol):
    def localize(self, crop: np.ndarray) -> Localization: ...
