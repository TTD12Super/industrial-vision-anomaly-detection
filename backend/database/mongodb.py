"""MongoDB connection management (PyMongo, single shared client).

The client is created once per process and reused by every request.
``pymongo`` is imported lazily so the backend can start (and report a
degraded health state) when the driver or server is unavailable.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(RuntimeError):
    """MongoDB driver/server cannot be reached (-> HTTP 503)."""


@lru_cache(maxsize=1)
def get_mongo_client() -> Any:
    """Return the shared ``MongoClient``, pinging the server first."""
    settings = get_settings()
    return connect(settings)


def connect(settings: Settings | None = None) -> Any:
    """Create a client for *settings* (defaults to app settings) and ping it."""
    resolved = settings or get_settings()
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError
    except ImportError as exc:
        raise DatabaseUnavailableError(
            "pymongo is not installed; database is unavailable."
        ) from exc
    client = MongoClient(resolved.mongodb_uri, serverSelectionTimeoutMS=resolved.mongodb_timeout_ms)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as exc:
        client.close()
        raise DatabaseUnavailableError(
            f"Cannot reach MongoDB at {resolved.mongodb_uri}."
        ) from exc
    logger.info(
        "Connected to MongoDB (database=%s, collection=%s).",
        resolved.mongodb_database,
        resolved.mongodb_collection,
    )
    return client


def get_collection(client: Any = None) -> Any:
    """Return the ``detections`` collection of the configured database."""
    settings = get_settings()
    resolved = client if client is not None else get_mongo_client()
    return resolved[settings.mongodb_database][settings.mongodb_collection]


def check_connection() -> bool:
    """True when a MongoDB ping succeeds, False otherwise (never raises)."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True
    except DatabaseUnavailableError:
        return False
    except Exception:  # defensive: health must never crash
        logger.exception("Unexpected MongoDB health-check failure.")
        return False


def close_client() -> None:
    """Close the shared client (application shutdown)."""
    try:
        get_mongo_client().close()
    except DatabaseUnavailableError:
        pass
    get_mongo_client.cache_clear()
