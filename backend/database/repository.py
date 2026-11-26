"""Detection history repository over the ``detections`` collection.

Only metadata/paths/URLs are stored — never image binaries (no GridFS).
Routes and services must use this class; no MongoDB queries elsewhere.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

ASCENDING = 1
DESCENDING = -1

INDEXES = [
    ("prediction_id", ASCENDING, {"unique": True}),
    ("created_at", DESCENDING, {}),
    ("category", ASCENDING, {}),
    ("anomaly_score", ASCENDING, {}),
]

PUBLIC_FIELDS = (
    "prediction_id",
    "category",
    "anomaly_score",
    "processing_time_ms",
    "created_at",
    "result",
)


def ensure_indexes(collection: Any) -> None:
    """Create history indexes (idempotent; safe to call at startup)."""
    for field, direction, options in INDEXES:
        collection.create_index([(field, direction)], **options)
    logger.info("Ensured %d indexes on the detections collection.", len(INDEXES))


def serialize_detection(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a Mongo document to the public API shape (no ``_id``/paths)."""
    if document is None:
        return None
    created = document.get("created_at")
    if isinstance(created, datetime):
        created_at = created.isoformat().replace("+00:00", "Z")
    else:
        created_at = str(created)
    return {
        "prediction_id": document.get("prediction_id"),
        "category": document.get("category"),
        "anomaly_score": document.get("anomaly_score"),
        "processing_time_ms": document.get("processing_time_ms"),
        "created_at": created_at,
        "result": document.get("result") or {},
    }


class DetectionRepository:
    """CRUD + filtered, sorted, paginated history queries."""

    def __init__(self, collection: Any) -> None:
        self.collection = collection

    def create_detection(self, document: dict[str, Any]) -> dict[str, Any]:
        result = self.collection.insert_one(dict(document))
        logger.info("Saved detection %s.", document.get("prediction_id"))
        stored = self.collection.find_one({"_id": result.inserted_id})
        return stored if stored is not None else dict(document)

    def get_detection(self, prediction_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"prediction_id": prediction_id})

    def exists(self, prediction_id: str) -> bool:
        return self.collection.find_one({"prediction_id": prediction_id}) is not None

    def _build_filter(
        self,
        category: str | None,
        min_score: float | None,
        max_score: float | None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if category is not None:
            query["category"] = category
        if min_score is not None or max_score is not None:
            score_range: dict[str, Any] = {}
            if min_score is not None:
                score_range["$gte"] = min_score
            if max_score is not None:
                score_range["$lte"] = max_score
            query["anomaly_score"] = score_range
        return query

    def count_detections(
        self,
        category: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> int:
        return self.collection.count_documents(
            self._build_filter(category, min_score, max_score)
        )

    def list_detections(
        self,
        category: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        """Newest-first page plus total count (``skip``/``limit`` in MongoDB)."""
        if page < 1:
            raise ValueError("page must be >= 1.")
        if page_size < 1:
            raise ValueError("page_size must be >= 1.")
        direction = DESCENDING if sort_order == "desc" else ASCENDING
        query = self._build_filter(category, min_score, max_score)
        total = self.collection.count_documents(query)
        cursor = (
            self.collection.find(query)
            .sort([("created_at", direction)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        return list(cursor), total

    def delete_detection(self, prediction_id: str) -> bool:
        result = self.collection.delete_one({"prediction_id": prediction_id})
        if result.deleted_count:
            logger.info("Deleted detection %s.", prediction_id)
        return bool(result.deleted_count)
