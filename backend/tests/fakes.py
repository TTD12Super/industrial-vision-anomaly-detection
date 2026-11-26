"""Test-only doubles (no MongoDB server needed).

``FakeCollection`` is an in-memory stand-in for the small pymongo surface
used by ``DetectionRepository`` (insert/find/count/delete + sort/skip/limit
cursors + selected query operators). API tests therefore exercise the REAL
repository code without any database server.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

try:
    from pymongo.errors import DuplicateKeyError
except ImportError:  # pragma: no cover - fallback when driver is missing

    class DuplicateKeyError(Exception):
        """Fallback duplicate-key error when pymongo is not installed."""


def _matches(document: dict, query: dict) -> bool:
    for field, condition in query.items():
        value = document.get(field)
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if operator == "$gte" and not value >= expected:
                    return False
                if operator == "$lte" and not value <= expected:
                    return False
                if operator == "$gt" and not value > expected:
                    return False
                if operator == "$lt" and not value < expected:
                    return False
                if operator == "$eq" and not value == expected:
                    return False
        elif value != condition:
            return False
    return True


class FakeCursor:
    """Minimal cursor supporting sort/skip/limit/iteration."""

    def __init__(self, documents: list[dict]) -> None:
        self._documents = list(documents)

    def sort(self, key_direction: list[tuple[str, int]]) -> FakeCursor:
        ordered = list(self._documents)
        for field, direction in reversed(key_direction):
            ordered.sort(key=lambda doc: doc.get(field), reverse=(direction < 0))
        self._documents = ordered
        return self

    def skip(self, count: int) -> FakeCursor:
        self._documents = self._documents[count:]
        return self

    def limit(self, count: int) -> FakeCursor:
        self._documents = self._documents[:count]
        return self

    def __iter__(self):
        return iter(self._documents)


class FakeCollection:
    """In-memory ``detections`` collection (enforces unique prediction_id)."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.created_indexes: list = []
        self._ids = itertools.count(1)

    def create_index(self, keys, **kwargs) -> str:
        self.created_indexes.append((list(keys), dict(kwargs)))
        return "_".join(field for field, _ in keys)

    def insert_one(self, document: dict) -> SimpleNamespace:
        if any(doc.get("prediction_id") == document.get("prediction_id") for doc in self.documents):
            raise DuplicateKeyError("duplicate prediction_id")
        stored = dict(document)
        stored["_id"] = next(self._ids)
        self.documents.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    def find_one(self, query: dict) -> dict | None:
        for document in self.documents:
            if _matches(document, query):
                return document
        return None

    def find(self, query: dict) -> FakeCursor:
        return FakeCursor([doc for doc in self.documents if _matches(doc, query)])

    def count_documents(self, query: dict) -> int:
        return sum(1 for doc in self.documents if _matches(doc, query))

    def delete_one(self, query: dict) -> SimpleNamespace:
        for index, document in enumerate(self.documents):
            if _matches(document, query):
                del self.documents[index]
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    def delete_many(self, query: dict) -> SimpleNamespace:
        before = len(self.documents)
        self.documents = [doc for doc in self.documents if not _matches(doc, query)]
        return SimpleNamespace(deleted_count=before - len(self.documents))
