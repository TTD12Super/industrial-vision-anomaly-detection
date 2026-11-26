"""Local-filesystem storage for prediction inputs and visualizations."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

_PREDICTION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ALLOWED_KINDS = {
    "input": "input.png",
    "heatmap": "heatmap.png",
    "overlay": "overlay.png",
}


def new_prediction_id() -> str:
    """Unique ID for one detection (uuid4 hex, safe to use in paths/URLs)."""
    return uuid.uuid4().hex


def validate_prediction_id(prediction_id: str) -> str:
    """Reject anything that is not a generated prediction ID (anti-traversal)."""
    if not _PREDICTION_ID_PATTERN.fullmatch(prediction_id or ""):
        raise FileNotFoundError(f"Prediction not found: {prediction_id}.")
    return prediction_id


class FileStorage:
    """Store prediction artifacts under ``<root>/<prediction_id>/``.

    Layout (git-ignored ``artifacts/`` tree, never in source control)::

        <root>/<prediction_id>/
            ├── input.png
            ├── heatmap.png
            ├── overlay.png
            └── result.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def prediction_dir(self, prediction_id: str) -> Path:
        return self.root / validate_prediction_id(prediction_id)

    def save_upload(self, prediction_id: str, source: str | Path) -> Path:
        """Persist the uploaded file bytes as ``input.png`` (normalized via PIL)."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to store uploaded images.") from exc
        destination = self.prediction_dir(prediction_id)
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.load()
            rgb = image.convert("RGB")
        output = destination / "input.png"
        rgb.save(output, "PNG")
        return output

    def save_rgb_array(self, prediction_id: str, kind: str, array) -> Path:
        """Persist a ``(H, W, 3)`` uint8 array as ``<kind>.png``."""
        if kind not in _ALLOWED_KINDS or kind == "input":
            raise ValueError(f"Unsupported artifact kind: {kind!r}.")
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("numpy and Pillow are required to store visualizations.") from exc
        pixels = np.asarray(array)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) array, got shape {pixels.shape}.")
        output = self.prediction_dir(prediction_id)
        output.mkdir(parents=True, exist_ok=True)
        destination = output / _ALLOWED_KINDS[kind]
        Image.fromarray(pixels.astype("uint8"), "RGB").save(destination, "PNG")
        return destination

    def save_metadata(self, prediction_id: str, metadata: dict) -> Path:
        import json

        destination = self.prediction_dir(prediction_id)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "result.json"
        output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return output

    def get_file(self, prediction_id: str, kind: str) -> Path:
        """Resolve a stored artifact or raise ``FileNotFoundError`` (-> 404)."""
        if kind not in _ALLOWED_KINDS:
            raise FileNotFoundError(f"Unknown artifact kind: {kind}.")
        path = self.prediction_dir(prediction_id) / _ALLOWED_KINDS[kind]
        if not path.is_file():
            raise FileNotFoundError(f"Prediction not found: {prediction_id}.")
        return path

    def copy_from(self, prediction_id: str, source: str | Path, filename: str) -> Path:
        """Copy an arbitrary file into the prediction folder (testing helper)."""
        destination = self.prediction_dir(prediction_id)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / Path(filename).name
        shutil.copyfile(source, output)
        return output
