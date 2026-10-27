"""Dataset inspection and deterministic preparation for the egg demo."""

from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
from pathlib import Path
import random
import shutil
from typing import Any
from zipfile import ZipFile

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_yolo_dataset(root: str | Path) -> dict[str, Any]:
    """Validate YOLO labels without changing the raw dataset."""

    root = Path(root)
    yaml_path = root / "data/data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YOLO data.yaml not found: {yaml_path}")
    names = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
             for line in yaml_path.read_text(encoding="utf-8").splitlines()
             if ":" in line and line.strip().split(":", 1)[0].strip().isdigit()}
    class_names = [names[key] for key in sorted(names, key=int)]
    result: dict[str, Any] = {"root": str(root.resolve()), "class_names": class_names, "splits": {}}
    for split in ("train", "val"):
        image_root = root / "data/images" / split
        label_root = root / "data/labels" / split
        images = _image_files(image_root)
        missing_labels = []
        invalid_labels = []
        boxes = Counter()
        dimensions = Counter()
        for image in images:
            with Image.open(image) as source:
                dimensions[tuple(source.size)] += 1
            label = label_root / f"{image.stem}.txt"
            if not label.is_file():
                missing_labels.append(str(label))
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                    valid = len(fields) == 5 and 0 <= class_id < len(class_names) and all(0 <= value <= 1 for value in coordinates)
                except (IndexError, ValueError):
                    valid = False
                    class_id = -1
                if not valid:
                    invalid_labels.append(f"{label}:{line_number}")
                else:
                    boxes[class_names[class_id]] += 1
        result["splits"][split] = {
            "images": len(images),
            "boxes": int(sum(boxes.values())),
            "boxes_by_class": dict(boxes),
            "dimensions": {f"{width}x{height}": count for (width, height), count in dimensions.items()},
            "missing_labels": missing_labels,
            "invalid_labels": invalid_labels,
        }
    return result


def write_local_yolo_yaml(root: str | Path, output: str | Path) -> Path:
    """Create a local YOLO config and sanitized label staging area.

    Some public YOLO archives include a UTF-8 BOM at the beginning of every
    label file. The raw download remains untouched; only a small staging tree
    beside the generated YAML is normalized for Ultralytics.
    """

    root = Path(root).resolve()
    report = validate_yolo_dataset(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_root = output.parent / "dataset"
    stage_data = stage_root / "data"
    stage_data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        image_source = root / "data/images" / split
        image_target = stage_data / "images" / split
        image_target.parent.mkdir(parents=True, exist_ok=True)
        if image_target.is_symlink():
            image_target.unlink()
        shutil.copytree(image_source, image_target, dirs_exist_ok=True)
        label_source = root / "data/labels" / split
        label_target = stage_data / "labels" / split
        label_target.mkdir(parents=True, exist_ok=True)
        for label_path in sorted(label_source.glob("*.txt")):
            normalized = label_path.read_text(encoding="utf-8-sig")
            (label_target / label_path.name).write_text(normalized, encoding="utf-8")
    lines = [f"path: {stage_root.resolve()}", "train: data/images/train", "val: data/images/val", "names:"]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(report["class_names"]))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def inspect_quality_archive(archive: str | Path) -> dict[str, Any]:
    """Inspect archive members; masks are only reported when files exist."""

    archive = Path(archive)
    with ZipFile(archive) as source:
        image_names = [name for name in source.namelist() if Path(name).suffix.lower() in IMAGE_SUFFIXES]
        classes = Counter()
        mask_like = []
        dimensions = Counter()
        for name in image_names:
            lower = name.lower()
            if "/good eggs/" in lower:
                classes["good"] += 1
            elif "/bad eggs/" in lower:
                classes["bad"] += 1
            if any(token in lower for token in ("mask", "ground_truth", "ground-truth", "segmentation")):
                mask_like.append(name)
            if len(dimensions) < 20:
                with source.open(name) as handle:
                    with Image.open(handle) as image:
                        dimensions[tuple(image.size)] += 1
    return {
        "archive": str(archive.resolve()),
        "members": len(image_names),
        "image_counts": dict(classes),
        "sample_dimensions": {f"{width}x{height}": count for (width, height), count in dimensions.items()},
        "mask_like_members": mask_like,
        "has_pixel_masks": bool(mask_like),
        "license": "CC BY 4.0",
    }


def _write_resized(source: Image.Image, output: Path, image_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    source.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS).save(output, quality=95)


def prepare_quality_dataset(
    archive: str | Path,
    output_root: str | Path,
    image_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract original good/bad images and split good images reproducibly."""

    archive = Path(archive)
    output_root = Path(output_root)
    records: list[dict[str, str]] = []
    with ZipFile(archive) as source:
        names = [name for name in source.namelist() if Path(name).suffix.lower() in IMAGE_SUFFIXES]
        for name in sorted(names):
            lower = name.lower()
            label = "good" if "/good eggs/" in lower else "bad" if "/bad eggs/" in lower else None
            if label is None:
                continue
            records.append({"source": name, "label": label, "sha256": sha256(source.read(name)).hexdigest()})
        good = [record for record in records if record["label"] == "good"]
        random.Random(seed).shuffle(good)
        good_train = int(len(good) * 0.70)
        good_validation = int(len(good) * 0.15)
        split_by_hash = {
            record["sha256"]: "train" if index < good_train else "validation" if index < good_train + good_validation else "test"
            for index, record in enumerate(good)
        }
        manifest_rows = []
        for record in records:
            split = split_by_hash.get(record["sha256"], "test")
            relative = Path(record["label"]) / split / f"{record['sha256'][:12]}.jpg"
            with source.open(record["source"]) as handle:
                with Image.open(handle) as image:
                    _write_resized(image, output_root / relative, image_size)
            manifest_rows.append({**record, "split": split, "image_path": str(relative)})
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "label", "split", "sha256", "image_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "output_root": str(output_root.resolve()),
        "image_count": len(manifest_rows),
        "counts": {f"{label}/{split}": count for (label, split), count in Counter((row["label"], row["split"]) for row in manifest_rows).items()},
        "manifest": str((output_root / "manifest.csv").resolve()),
        "protocol": "original archive only; good images split 70/15/15 with seed; bad images reserved for test; augmented archive excluded",
    }


def prepare_freshness_dataset(
    archive: str | Path,
    output_root: str | Path,
    image_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare Fresh images for the normal bank and Rotten images for testing."""

    archive = Path(archive)
    output_root = Path(output_root)
    records: list[dict[str, str]] = []
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            normalized = name.lower().replace("-", "_").replace(" ", "_")
            if "rotten" in normalized or "day45" in normalized or "day_45" in normalized:
                label = "bad"
                source_label = "rotten"
            elif "fresh" in normalized and "half" not in normalized or "day0" in normalized or "day_0" in normalized:
                label = "good"
                source_label = "fresh"
            else:
                continue
            records.append({
                "source": name,
                "label": label,
                "source_label": source_label,
                "sha256": sha256(source.read(name)).hexdigest(),
            })

        good = [record for record in records if record["label"] == "good"]
        random.Random(seed).shuffle(good)
        train_end = int(len(good) * 0.70)
        validation_end = train_end + int(len(good) * 0.15)
        good_split = {
            record["sha256"]: "train" if index < train_end else "validation" if index < validation_end else "test"
            for index, record in enumerate(good)
        }
        manifest_rows: list[dict[str, str]] = []
        for record in records:
            split = good_split.get(record["sha256"], "test")
            relative = Path(record["label"]) / split / f"{record['sha256'][:16]}.jpg"
            with source.open(record["source"]) as handle:
                with Image.open(handle) as image:
                    _write_resized(image, output_root / relative, image_size)
            manifest_rows.append({**record, "split": split, "image_path": str(relative)})

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "source_label", "label", "split", "sha256", "image_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "output_root": str(output_root.resolve()),
        "image_count": len(manifest_rows),
        "counts": {f"{label}/{split}": count for (label, split), count in Counter((row["label"], row["split"]) for row in manifest_rows).items()},
        "manifest": str(manifest.resolve()),
        "protocol": "Fresh only for the normal bank; Fresh split 70/15/15; Rotten reserved for test; Half-Fresh excluded",
    }
