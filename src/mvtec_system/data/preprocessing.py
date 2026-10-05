"""Read-only MVTec validation and deterministic processed-dataset creation."""

from __future__ import annotations

from collections import Counter
import hashlib
import csv
import random
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = 256
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)
RGB_RESAMPLE = Image.Resampling.LANCZOS
MASK_RESAMPLE = Image.Resampling.NEAREST


def is_supported_image(path: Path) -> bool:
    """Return whether a path is a supported image file."""

    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def image_files(directory: str | Path) -> list[Path]:
    """Return supported image files below a directory in stable order."""

    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if is_supported_image(path))


def category_directories(raw_root: str | Path) -> list[Path]:
    """Discover categories with train, test, and ground_truth directories."""

    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory is missing: {raw_root}")
    required = {"train", "test", "ground_truth"}
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir()
        and required.issubset(
            child.name for child in path.iterdir() if child.is_dir()
        )
    )


def discover_dataset_root(
    datasets_root: str | Path = "datasets",
    explicit_root: str | Path | None = None,
) -> Path:
    """Find one MVTec-style dataset without depending on its directory name.

    Processed siblings such as ``*_processed`` are ignored.  If more than one
    valid raw dataset is present, callers must choose one explicitly so a run
    never silently uses the wrong data.
    """

    if explicit_root is not None:
        root = Path(explicit_root).resolve()
        if not category_directories(root):
            raise ValueError(f"Dataset has no MVTec-style categories: {root}")
        return root
    parent = Path(datasets_root).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"Datasets directory is missing: {parent}")
    candidates = [
        path
        for path in sorted(parent.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and not path.name.endswith("_processed")
        and category_directories(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No MVTec-style dataset found under {parent}; expected category/train, test, ground_truth folders."
        )
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise ValueError(f"More than one dataset found; use --dataset-root: {names}")
    return candidates[0]


def create_classification_manifest(
    processed_root: str | Path,
    manifest_path: str | Path,
    validation_fraction: float = 0.30,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a deterministic train/validation manifest from train/good only."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    root = Path(processed_root).resolve()
    categories = sorted(path.name for path in category_directories(root))
    rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for category in categories:
        images = image_files(root / category / "train" / "good")
        if not images:
            raise ValueError(f"Category has no train/good images: {category}")
        shuffled = list(images)
        random.Random(seed + sum(ord(char) for char in category)).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - 1) if len(shuffled) > 1 else 0
        validation_paths = set(shuffled[:validation_count])
        train_count = 0
        validation_count_actual = 0
        for image in sorted(images):
            split = "validation" if image in validation_paths else "train"
            rows.append(
                {
                    "image_path": image.relative_to(root).as_posix(),
                    "category": category,
                    "split": split,
                }
            )
            if split == "validation":
                validation_count_actual += 1
            else:
                train_count += 1
        counts[category] = {"train": train_count, "validation": validation_count_actual}

    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "category", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "manifest_path": str(output.resolve()),
        "processed_root": str(root),
        "categories": categories,
        "row_count": len(rows),
        "counts": counts,
        "validation_fraction": validation_fraction,
        "seed": seed,
    }


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(root: str | Path) -> dict[str, tuple[int, str]]:
    """Return size and SHA-256 for every file below a dataset root."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {root}")
    return {
        str(path.relative_to(root)): (path.stat().st_size, file_sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def probe_image(path: str | Path) -> dict[str, Any]:
    """Read image metadata without modifying the source file."""

    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()
            return {
                "readable": True,
                "size": image.size,
                "mode": image.mode,
                "channels": len(image.getbands()),
                "format": image.format,
                "error": None,
            }
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return {
            "readable": False,
            "size": None,
            "mode": None,
            "channels": None,
            "format": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _records(
    categories: list[Path],
    split: str,
) -> list[dict[str, Any]]:
    records = []
    for category in categories:
        split_root = category / split
        for path in image_files(split_root):
            relative = path.relative_to(split_root)
            records.append(
                {
                    "category": category.name,
                    "split": split,
                    "label": relative.parts[0],
                    "path": path,
                }
            )
    return records


def _validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [{**record, **probe_image(record["path"])} for record in records]
    failures = [record for record in checked if not record["readable"]]
    return {
        "total": len(checked),
        "readable": len(checked) - len(failures),
        "failures": [str(record["path"]) for record in failures],
        "unexpected": [],
        "records": checked,
    }


def validate_raw_dataset(raw_root: str | Path) -> dict[str, Any]:
    """Validate category structure, decodability, and mask pairing."""

    raw_root = Path(raw_root).resolve()
    categories = category_directories(raw_root)
    records = {split: _records(categories, split) for split in ("train", "test", "ground_truth")}
    validations = {split: _validate_records(items) for split, items in records.items()}

    for category in categories:
        for split in ("train", "test", "ground_truth"):
            split_root = category / split
            validations[split]["unexpected"] = [
                str(path)
                for path in split_root.rglob("*")
                if path.is_file() and not is_supported_image(path)
            ]

    expected_masks = {
        (
            record["category"],
            record["label"],
            f"{record['path'].stem}_mask",
        )
        for record in records["test"]
        if record["label"] != "good"
    }
    found_masks = {
        (record["category"], record["label"], record["path"].stem)
        for record in records["ground_truth"]
    }
    mismatches = []
    for record in records["ground_truth"]:
        if not record["path"].stem.endswith("_mask"):
            continue
        source = (
            raw_root
            / record["category"]
            / "test"
            / record["label"]
            / f"{record['path'].stem.removesuffix('_mask')}.png"
        )
        source_probe = probe_image(source)
        if (
            record["path"].is_file()
            and record["path"].exists()
            and record["path"].is_file()
            and source_probe["readable"]
            and probe_image(record["path"])["readable"]
            and probe_image(record["path"])["size"] != source_probe["size"]
        ):
            mismatches.append((str(record["path"]), str(source)))

    counts = {}
    for category in categories:
        test_records = [record for record in records["test"] if record["category"] == category.name]
        counts[category.name] = {
            "train": sum(record["category"] == category.name for record in records["train"]),
            "test": len(test_records),
            "test_good": sum(record["label"] == "good" for record in test_records),
            "test_defect": sum(record["label"] != "good" for record in test_records),
            "ground_truth": sum(record["category"] == category.name for record in records["ground_truth"]),
            "test_labels": sorted({record["label"] for record in test_records}),
        }

    failures = {
        split: validations[split]["failures"] + validations[split]["unexpected"]
        for split in validations
    }
    failures["ground_truth"] += [str(path) for path in sorted(expected_masks - found_masks)]
    failures["ground_truth"] += [str(path) for path in sorted(found_masks - expected_masks)]
    failures["ground_truth"] += [f"dimension mismatch: {item}" for item in mismatches]
    return {
        "raw_root": str(raw_root),
        "categories": [category.name for category in categories],
        "category_count": len(categories),
        "counts": counts,
        "splits": {
            split: {
                key: value
                for key, value in validation.items()
                if key != "records"
            }
            for split, validation in validations.items()
        },
        "failures": failures,
        "valid": not any(failures.values()),
        "manifest": dataset_manifest(raw_root),
    }


def processed_output_path(
    path: str | Path,
    raw_root: str | Path,
    processed_root: str | Path,
) -> Path:
    """Map one raw path to its processed PNG path."""

    path = Path(path).resolve()
    raw_root = Path(raw_root).resolve()
    if not path.is_relative_to(raw_root):
        raise ValueError(f"Path leaves raw dataset root: {path}")
    return (Path(processed_root).resolve() / path.relative_to(raw_root)).with_suffix(".png")


def preprocess_image(
    path: str | Path,
    raw_root: str | Path,
    processed_root: str | Path,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
    mask: bool = False,
    overwrite: bool = False,
) -> Path:
    """Convert one image to the fixed processed PNG contract."""

    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    if len(size) != 2 or min(size) < 1:
        raise ValueError("image_size must contain two positive dimensions")
    destination = processed_output_path(path, raw_root, processed_root)
    if destination.is_file() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as source:
        source.load()
        image = source.copy() if mask else source.convert("RGB")
        image.resize((size[1], size[0]), MASK_RESAMPLE if mask else RGB_RESAMPLE).save(
            destination, "PNG"
        )
    return destination


def validate_processed_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
) -> dict[str, Any]:
    """Validate processed image dimensions and RGB mode against raw records."""

    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    categories = category_directories(raw_root)
    results = {}
    for split in ("train", "test", "ground_truth"):
        failures = []
        for record in _records(categories, split):
            output = processed_output_path(record["path"], raw_root, processed_root)
            probe = probe_image(output)
            valid = probe["readable"] and probe["size"] == (size[1], size[0])
            if split != "ground_truth":
                valid = valid and probe["mode"] == "RGB"
            if not valid:
                failures.append(str(output))
        results[split] = {"failures": failures, "count": len(_records(categories, split))}
    return {
        "processed_root": str(Path(processed_root).resolve()),
        "image_size": list(size),
        "splits": results,
        "valid": not any(item["failures"] for item in results.values()),
    }


def preprocess_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Preprocess all raw train/test images and ground-truth masks."""

    raw_root = Path(raw_root).resolve()
    processed_root = Path(processed_root).resolve()
    categories = category_directories(raw_root)
    records = {split: _records(categories, split) for split in ("train", "test", "ground_truth")}
    created = 0
    reused = 0
    for split, items in records.items():
        for record in items:
            destination = processed_output_path(record["path"], raw_root, processed_root)
            existed = destination.is_file()
            destination = preprocess_image(
                record["path"],
                raw_root,
                processed_root,
                image_size,
                mask=split == "ground_truth",
                overwrite=overwrite,
            )
            if existed:
                reused += 1
            else:
                created += 1
    return {
        "created": created,
        "reused": reused,
        "validation": validate_processed_dataset(raw_root, processed_root, image_size),
    }


def prepare_mvtec_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    image_size: int | tuple[int, int] = IMAGE_SIZE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate raw data, preprocess safely, and verify raw data was unchanged."""

    before = validate_raw_dataset(raw_root)
    if not before["valid"]:
        raise ValueError(f"Raw dataset validation failed: {before['failures']}")
    raw_manifest = before["manifest"]
    processing = preprocess_dataset(raw_root, processed_root, image_size, overwrite)
    after_manifest = dataset_manifest(raw_root)
    processing["raw_dataset_unchanged"] = raw_manifest == after_manifest
    if not processing["raw_dataset_unchanged"]:
        raise RuntimeError("Raw dataset changed during preprocessing.")
    return {
        "raw": {
            key: value for key, value in before.items() if key != "manifest"
        },
        "processing": processing,
    }
