"""Safe inspection artifact paths and manifest generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def safe_inspection_id(value: str) -> str:
    if not value or len(value) > 80 or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
        raise ValueError("inspection_id must be 1-80 letters, numbers, '-' or '_'")
    return value


@dataclass(frozen=True)
class InspectionStore:
    root: Path
    inspection_id: str

    @property
    def directory(self) -> Path:
        return self.root / safe_inspection_id(self.inspection_id)

    def path(self, relative: str) -> Path:
        candidate = (self.directory / relative).resolve()
        root = self.directory.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Artifact path escapes the inspection directory")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def write_json(self, relative: str, value: dict[str, Any]) -> Path:
        path = self.path(relative)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def manifest(self) -> dict[str, dict[str, str]]:
        entries: dict[str, dict[str, str]] = {}
        if not self.directory.is_dir():
            return entries
        for path in sorted(item for item in self.directory.rglob("*") if item.is_file()):
            relative = path.relative_to(self.directory).as_posix()
            artifact_id = hashlib.sha256(relative.encode()).hexdigest()[:16]
            entries[artifact_id] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return entries
