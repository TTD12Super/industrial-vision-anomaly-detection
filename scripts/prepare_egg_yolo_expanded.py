#!/usr/bin/env python3
"""Build an expanded egg detector dataset with hard background negatives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_BOX_AREA = 0.002


def _read_labels(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if class_id not in (0, 1) or not all(0 <= value <= 1 for value in values):
            continue
        if values[2] * values[3] < MIN_BOX_AREA:
            continue
        rows.append(" ".join(["0", *[f"{value:.8f}" for value in values]]))
    return rows


def _copy_pair(image: Path, label: Path, root: Path, split: str, prefix: str) -> bool:
    target_image = root / "data/images" / split / f"{prefix}_{image.name}"
    target_label = root / "data/labels" / split / f"{target_image.stem}.txt"
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(image.resolve(), target_image)
    except OSError:
        shutil.copy2(image, target_image)
    rows = _read_labels(label)
    target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return True


def _box_xyxy(row: str, width: int, height: int) -> tuple[float, float, float, float]:
    _, x, y, box_width, box_height = (float(value) for value in row.split())
    return (
        (x - box_width / 2) * width,
        (y - box_height / 2) * height,
        (x + box_width / 2) * width,
        (y + box_height / 2) * height,
    )


def _iou(left: tuple[int, int, int, int], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _add_backgrounds(
    images: list[Path],
    root: Path,
    split: str,
    count: int,
    seed: int,
    start_index: int,
) -> int:
    rng = random.Random(seed)
    created = 0
    for source in images:
        if created >= count:
            break
        label = source.parent.parent / "labels" / f"{source.stem}.txt"
        with Image.open(source) as image:
            width, height = image.size
            rows = _read_labels(label)
            boxes = [_box_xyxy(row, width, height) for row in rows]
            crop_width = max(96, int(width * 0.30))
            crop_height = max(96, int(height * 0.30))
            for _ in range(80):
                if created >= count:
                    break
                left = rng.randint(0, max(0, width - crop_width))
                top = rng.randint(0, max(0, height - crop_height))
                candidate = (left, top, left + crop_width, top + crop_height)
                if any(_iou(candidate, box) > 0.01 for box in boxes):
                    continue
                output = root / "data/images" / split / f"background_{start_index + created:03d}.jpg"
                label_output = root / "data/labels" / split / f"{output.stem}.txt"
                output.parent.mkdir(parents=True, exist_ok=True)
                label_output.parent.mkdir(parents=True, exist_ok=True)
                image.crop(candidate).convert("RGB").save(output, quality=95)
                label_output.write_text("", encoding="utf-8")
                created += 1
                break
    return created


def build(output: Path, raw_root: Path, kaggle_root: Path, seed: int) -> dict:
    if output.exists():
        shutil.rmtree(output)
    raw_train = sorted((raw_root / "data/images/train").glob("*"))
    raw_val = sorted((raw_root / "data/images/val").glob("*"))
    external_train = sorted((kaggle_root / "train/images").glob("*"))
    external_val = sorted((kaggle_root / "valid/images").glob("*"))
    external_train = [path for path in external_train if path.suffix.lower() in IMAGE_SUFFIXES]
    external_val = [path for path in external_val if path.suffix.lower() in IMAGE_SUFFIXES]

    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}
    for split, images in (("train", raw_train), ("val", raw_val)):
        for image in images:
            label = raw_root / "data/labels" / split / f"{image.stem}.txt"
            _copy_pair(image, label, output, split, "raw")
            counts[f"{split}_images"] += 1
            counts[f"{split}_boxes"] += len(_read_labels(label))
    for split, source_split, images in (
        ("train", "train", external_train),
        ("val", "valid", external_val),
    ):
        for image in images:
            label = kaggle_root / source_split / "labels" / f"{image.stem}.txt"
            _copy_pair(image, label, output, split, "kaggle")
            counts[f"{split}_images"] += 1
            counts[f"{split}_boxes"] += len(_read_labels(label))

    train_backgrounds = _add_backgrounds(external_train, output, "train", 20, seed, 0)
    val_backgrounds = _add_backgrounds(external_val, output, "val", 5, seed + 1, train_backgrounds)
    counts.update({"train_backgrounds": train_backgrounds, "val_backgrounds": val_backgrounds})
    (output / "data/data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (output / "data/data.yaml").write_text(
        f"path: {output.resolve()}\ntrain: data/images/train\nval: data/images/val\nnames:\n  0: egg\n",
        encoding="utf-8",
    )
    report = {
        "output": str(output.resolve()),
        "seed": seed,
        "source_counts": {
            "raw_train": len(raw_train),
            "raw_val": len(raw_val),
            "kaggle_train": len(external_train),
            "kaggle_valid": len(external_val),
        },
        "split_counts": counts,
        "class_policy": "collapse white-egg and brown-egg into one egg class",
        "min_box_area": MIN_BOX_AREA,
        "background_policy": "empty-label crops sampled outside annotated boxes",
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/egg_yolo_expanded"))
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw/egg_detection"))
    parser.add_argument("--kaggle-root", type=Path, default=Path("datasets/external/egg_damaged_kaggle/fold_0"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.raw_root, args.kaggle_root, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
