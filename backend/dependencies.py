"""FastAPI dependency wiring (singletons + overrides for tests)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import Settings, get_settings
from .services.detection_service import DetectionService
from .services.detector_service import DetectorService
from .storage.file_storage import FileStorage


@lru_cache(maxsize=1)
def get_detector_service() -> DetectorService:
    return DetectorService(bank_dir=get_settings().bank_dir)


@lru_cache(maxsize=1)
def get_storage() -> FileStorage:
    return FileStorage(root=get_settings().predictions_dir)


@lru_cache(maxsize=1)
def get_detection_repository() -> Any:
    """Shared history repository (raises ``DatabaseUnavailableError`` when down)."""
    from .database.mongodb import get_collection
    from .database.repository import DetectionRepository

    return DetectionRepository(get_collection())


def get_optional_repository() -> Any | None:
    """Repository or ``None`` when MongoDB is down (degraded artifact serving)."""
    from .database.mongodb import DatabaseUnavailableError

    try:
        return get_detection_repository()
    except DatabaseUnavailableError:
        return None


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    return DetectionService(
        detectors=get_detector_service(),
        storage=get_storage(),
        repository=get_detection_repository(),
    )


def get_app_settings() -> Settings:
    return get_settings()


def get_database_status() -> str:
    """``'connected'`` or ``'disconnected'`` (never raises; for health checks)."""
    from .database.mongodb import check_connection

    return "connected" if check_connection() else "disconnected"
