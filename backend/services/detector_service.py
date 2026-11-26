"""Cache and reuse of Phase 1 ``AnomalyDetector`` instances.

Strategy: lazy-load one detector per category on first request and reuse it
afterwards. Nothing is preloaded at startup (15 banks would waste RAM/VRAM).
The Phase 1 AI code is never modified here; this module only manages
lifecycles and maps setup failures to explicit errors.

``torch``/``AnomalyDetector`` are imported lazily so the backend process can
start (health/categories) even when AI dependencies are missing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from model.inference.config import default_bank_path, validate_category

logger = logging.getLogger(__name__)


class BankNotAvailableError(FileNotFoundError):
    """Feature bank file for a category does not exist (-> HTTP 503)."""


class ModelNotAvailableError(RuntimeError):
    """DINOv2/torch cannot be loaded in this environment (-> HTTP 503)."""


class DetectorService:
    """Thread-safe per-category ``AnomalyDetector`` cache."""

    def __init__(self, bank_dir: str | Path) -> None:
        self.bank_dir = Path(bank_dir)
        self._detectors: dict[str, Any] = {}
        self._lock = threading.Lock()

    def available_categories(self) -> list[str]:
        from model.inference.config import CATEGORIES

        return list(CATEGORIES)

    def bank_path(self, category: str) -> Path:
        return default_bank_path(validate_category(category), self.bank_dir)

    def is_bank_available(self, category: str) -> bool:
        return self.bank_path(category).is_file()

    def get_detector(self, category: str) -> Any:
        """Return the cached detector for *category*, loading it on first use."""
        validate_category(category)  # raises ValueError -> HTTP 400
        with self._lock:
            detector = self._detectors.get(category)
            if detector is not None:
                return detector
            bank_file = self.bank_path(category)
            if not bank_file.is_file():
                raise BankNotAvailableError(
                    f"Feature bank for category '{category}' is not available."
                )
            try:
                from model.inference.detector import AnomalyDetector
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "AI dependencies (torch) are not installed; "
                    "inference is unavailable."
                ) from exc
            try:
                detector = AnomalyDetector(category=category, bank_path=bank_file)
            except FileNotFoundError as exc:
                raise BankNotAvailableError(
                    f"Feature bank for category '{category}' is not available."
                ) from exc
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "AI dependencies are not installed; inference is unavailable."
                ) from exc
            self._detectors[category] = detector
            logger.info("Loaded detector for category '%s' (bank: %s).", category, bank_file)
            return detector

    def clear_cache(self) -> None:
        """Drop cached detectors (mainly useful for tests)."""
        with self._lock:
            self._detectors.clear()
