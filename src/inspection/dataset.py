"""Validation for the portable images.jsonl and objects.jsonl dataset contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ProductConfig
from .geometry import order_corners


SPLITS = {"train", "validation", "test"}
QUALITY_LABELS = {"GOOD", "DEFECT"}


@dataclass(frozen=True)
class DatasetValidationReport:
    product_key: str
    image_count: int
    object_count: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_key": self.product_key,
            "image_count": self.image_count,
            "object_count": self.object_count,
            "split_counts": self.split_counts,
            "label_counts": self.label_counts,
            "errors": list(self.errors),
            "valid": self.valid,
        }


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"Missing manifest: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{number}: invalid JSON ({error.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path}:{number}: expected a JSON object")
            continue
        rows.append(row)
    return rows


def _safe_relative(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(config: ProductConfig, verify_hashes: bool = True) -> DatasetValidationReport:
    """Validate cross-file identity, labels, images, paths, OBBs, and grouped splits."""

    errors: list[str] = []
    images = _read_jsonl(config.dataset_path(config.image_manifest), errors)
    objects = _read_jsonl(config.dataset_path(config.object_manifest), errors)
    image_by_id: dict[str, dict[str, Any]] = {}
    group_splits: dict[str, set[str]] = {}
    split_counts = {split: 0 for split in sorted(SPLITS)}
    for number, row in enumerate(images, 1):
        prefix = f"images.jsonl:{number}"
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            errors.append(f"{prefix}: image_id is required")
            continue
        if image_id in image_by_id:
            errors.append(f"{prefix}: duplicate image_id {image_id}")
            continue
        split = row.get("split")
        if split not in SPLITS:
            errors.append(f"{prefix}: split must be one of {sorted(SPLITS)}")
            continue
        path = _safe_relative(row.get("relative_path"))
        if path is None:
            errors.append(f"{prefix}: relative_path must stay within dataset root")
            continue
        source_path = config.dataset_path(path)
        if not source_path.is_file():
            errors.append(f"{prefix}: missing image {path}")
            continue
        try:
            with Image.open(source_path) as image:
                actual_width, actual_height = image.size
        except OSError:
            errors.append(f"{prefix}: unreadable image {path}")
            continue
        if row.get("width") != actual_width or row.get("height") != actual_height:
            errors.append(f"{prefix}: width/height do not match {path}")
        expected_hash = row.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"{prefix}: sha256 is required")
        elif verify_hashes and _sha256(source_path) != expected_hash:
            errors.append(f"{prefix}: sha256 does not match {path}")
        capture_group = row.get("capture_group_id")
        if not isinstance(capture_group, str) or not capture_group:
            errors.append(f"{prefix}: capture_group_id is required")
        else:
            group_splits.setdefault(capture_group, set()).add(split)
        image_by_id[image_id] = row
        split_counts[split] += 1
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            errors.append(f"capture_group_id {group} appears in multiple splits: {sorted(splits)}")

    object_ids: set[str] = set()
    labels = {label: 0 for label in sorted(QUALITY_LABELS)}
    for number, row in enumerate(objects, 1):
        prefix = f"objects.jsonl:{number}"
        object_id = row.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            errors.append(f"{prefix}: object_id is required")
            continue
        if object_id in object_ids:
            errors.append(f"{prefix}: duplicate object_id {object_id}")
            continue
        object_ids.add(object_id)
        image = image_by_id.get(row.get("image_id"))
        if image is None:
            errors.append(f"{prefix}: image_id does not exist")
            continue
        if row.get("product_key") != config.product_key:
            errors.append(f"{prefix}: product_key must be {config.product_key}")
        label = row.get("quality_label")
        if label not in QUALITY_LABELS:
            errors.append(f"{prefix}: quality_label must be GOOD or DEFECT")
        else:
            labels[label] += 1
        try:
            corners = order_corners(row.get("obb_xyxyxyxy", []))
            if (corners[:, 0] < 0).any() or (corners[:, 1] < 0).any() or (corners[:, 0] > image["width"]).any() or (corners[:, 1] > image["height"]).any():
                errors.append(f"{prefix}: OBB is outside the image bounds")
        except (TypeError, ValueError):
            errors.append(f"{prefix}: obb_xyxyxyxy must contain four valid corners")
        mask_path = row.get("mask_relative_path")
        if mask_path is not None:
            relative_mask = _safe_relative(mask_path)
            if relative_mask is None or not config.dataset_path(relative_mask).is_file():
                errors.append(f"{prefix}: mask_relative_path does not exist within dataset root")
    if not images:
        errors.append("No images were found in images.jsonl")
    if not objects:
        errors.append("No objects were found in objects.jsonl")
    if any(labels[label] == 0 for label in QUALITY_LABELS):
        errors.append("Both GOOD and DEFECT object labels are required for supervised classification")
    return DatasetValidationReport(
        product_key=config.product_key,
        image_count=len(image_by_id),
        object_count=len(object_ids),
        split_counts=split_counts,
        label_counts=labels,
        errors=tuple(errors),
    )


def load_dataset_records(config: ProductConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load validated records for generic training/evaluation code."""

    report = validate_dataset(config)
    if not report.valid:
        raise ValueError("Dataset contract is invalid: " + "; ".join(report.errors))
    errors: list[str] = []
    images = _read_jsonl(config.dataset_path(config.image_manifest), errors)
    objects = _read_jsonl(config.dataset_path(config.object_manifest), errors)
    if errors:
        raise ValueError("Dataset manifest changed during validation: " + "; ".join(errors))
    return images, objects
