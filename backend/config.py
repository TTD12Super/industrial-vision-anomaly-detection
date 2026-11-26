"""Environment-based backend configuration (stdlib only, no extra dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _cors_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    max_upload_size_mb: int = 10
    bank_dir: Path = field(default_factory=lambda: _project_root() / "artifacts" / "dinov2_knn" / "banks")
    predictions_dir: Path = field(default_factory=lambda: _project_root() / "artifacts" / "predictions")
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "industrial_anomaly_detection"
    mongodb_collection: str = "detections"
    mongodb_timeout_ms: int = 2000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings from environment variables (safe defaults for development)."""
    try:
        port = int(os.environ.get("APP_PORT", "8000"))
    except ValueError:
        port = 8000
    try:
        max_mb = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "10"))
    except ValueError:
        max_mb = 10
    try:
        mongo_timeout = int(os.environ.get("MONGODB_TIMEOUT_MS", "2000"))
    except ValueError:
        mongo_timeout = 2000
    root = _project_root()
    return Settings(
        app_host=os.environ.get("APP_HOST", "0.0.0.0"),
        app_port=port,
        cors_origins=_cors_origins(os.environ.get("CORS_ORIGINS", "http://localhost:5173")),
        max_upload_size_mb=max_mb,
        bank_dir=Path(os.environ.get("BANK_DIR", str(root / "artifacts" / "dinov2_knn" / "banks"))),
        predictions_dir=Path(
            os.environ.get("PREDICTIONS_DIR", str(root / "artifacts" / "predictions"))
        ),
        mongodb_uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_database=os.environ.get("MONGODB_DATABASE", "industrial_anomaly_detection"),
        mongodb_collection=os.environ.get("MONGODB_COLLECTION", "detections"),
        mongodb_timeout_ms=mongo_timeout,
    )
