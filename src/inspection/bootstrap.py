"""One-time, non-destructive conversion of the existing screw pilot to the common contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from .config import ProductConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing dataset file: {destination}")
    destination.symlink_to(source.resolve())


def bootstrap_screw_pilot(config: ProductConfig, source_root: str | Path) -> dict[str, Any]:
    """Create symlinked images/masks plus canonical JSONL from verified pilot labels.

    This intentionally creates no copy or modification of MVTec AD source bytes.
    """

    source_root = Path(source_root)
    manifest_path = source_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing recovered OBB manifest: {manifest_path}")
    if config.dataset_root.exists() and any(config.dataset_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated product dataset: {config.dataset_root}")
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_image = Path(row["source"])
            source_label = Path(row["label"])
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(f"Broken pilot source row: {source_image} / {source_label}")
            split = {"val": "validation"}.get(row["split"], row["split"])
            if split not in counts:
                raise ValueError(f"Unexpected pilot split: {row['split']}")
            category = row["category"]
            stem = source_image.stem
            image_id = f"mvtec_ad_screw_{category}_{stem}"
            image_relative = Path("images") / split / f"{image_id}{source_image.suffix.lower()}"
            _symlink(source_image, config.dataset_path(image_relative))
            with Image.open(source_image) as image:
                width, height = image.size
            values = [float(value) for value in source_label.read_text(encoding="utf-8").split()]
            if len(values) != 9 or int(values[0]) != 0:
                raise ValueError(f"Expected one class-0 YOLO-OBB label: {source_label}")
            points = [[values[index] * width, values[index + 1] * height] for index in range(1, 9, 2)]
            mask_relative: str | None = None
            if category != "good":
                raw_screw_root = source_image.parents[2]
                mask = raw_screw_root / "ground_truth" / category / f"{stem}_mask.png"
                if mask.is_file():
                    relative_mask = Path("masks") / category / f"{stem}_mask.png"
                    _symlink(mask, config.dataset_path(relative_mask))
                    mask_relative = relative_mask.as_posix()
            image_rows.append(
                {
                    "image_id": image_id,
                    "relative_path": image_relative.as_posix(),
                    "split": split,
                    "source_dataset": "mvtec_ad_screw_supervised_development",
                    "capture_group_id": f"mvtec_ad_screw/{category}/{stem}",
                    "domain_id": "mvtec_ad_clean_background",
                    "sha256": _sha256(source_image),
                    "width": width,
                    "height": height,
                }
            )
            object_rows.append(
                {
                    "object_id": f"{image_id}_obj01",
                    "image_id": image_id,
                    "product_key": config.product_key,
                    "obb_xyxyxyxy": points,
                    "quality_label": "GOOD" if category == "good" else "DEFECT",
                    "defect_types": [] if category == "good" else [category],
                    "mask_relative_path": mask_relative,
                }
            )
            counts[split] += 1
    image_manifest = config.dataset_path(config.image_manifest)
    object_manifest = config.dataset_path(config.object_manifest)
    image_manifest.parent.mkdir(parents=True, exist_ok=True)
    image_manifest.write_text("".join(json.dumps(row) + "\n" for row in image_rows), encoding="utf-8")
    object_manifest.write_text("".join(json.dumps(row) + "\n" for row in object_rows), encoding="utf-8")
    (config.dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "product_key": config.product_key, "split_counts": counts}, sort_keys=False),
        encoding="utf-8",
    )
    provenance = {
        "source_root": str(source_root.resolve()),
        "input_manifest": str(manifest_path.resolve()),
        "images_are_symlinks_to_raw_mvtec_ad": True,
        "official_test_data_used_for_supervised_development": True,
        "counts": counts,
    }
    (config.dataset_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance
