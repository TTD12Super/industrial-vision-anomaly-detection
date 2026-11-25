#!/usr/bin/env python3
"""Build a YOLO-OBB MVTec AD screw dataset from green-box review images."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from screw_yolo_obb import order_corners, write_yolo_label


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def corners_from_green_box(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read overlay image: {path}")
    # OpenCV stores pixels as BGR; the annotation is exact bright green.
    mask = (
        (image[:, :, 1] > 200)
        & (image[:, :, 0] < 80)
        & (image[:, :, 2] < 80)
    ).astype(np.uint8)
    points = np.column_stack(np.where(mask > 0)[::-1]).astype(np.float32)
    if len(points) < 20:
        raise ValueError(f"No usable green box found in {path}")
    rectangle = cv2.minAreaRect(points)
    return order_corners(cv2.boxPoints(rectangle))


def split_images(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    # 15/3/2 per category: keep every defect type in every split.
    return {"train": shuffled[:15], "val": shuffled[15:18], "test": shuffled[18:]}


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="mvtec_ad_screw_bbox_") as temporary:
        extracted = Path(temporary)
        safe_extract(archive, extracted)
        roots = list(extracted.glob("*/"))
        if len(roots) != 1:
            raise ValueError("Expected one top-level directory in the archive")
        overlay_root = roots[0]
        categories = sorted(path for path in overlay_root.iterdir() if path.is_dir())
        if not categories:
            raise ValueError("No category directories found in archive")
        for category in categories:
            overlays = sorted(category.glob("*.png"))
            if not overlays:
                raise ValueError(f"No PNG images in {category}")
            splits = split_images(overlays, args.seed + sum(category.name.encode("utf-8")))
            raw_category = source_root / category.name
            if not raw_category.is_dir():
                raise FileNotFoundError(raw_category)
            for split, split_overlays in splits.items():
                for overlay in split_overlays:
                    raw_image = raw_category / overlay.name
                    if not raw_image.is_file():
                        raise FileNotFoundError(raw_image)
                    corners = corners_from_green_box(overlay)
                    with Image.open(raw_image) as image:
                        width, height = image.size
                        if (width, height) != (1024, 1024):
                            raise ValueError(f"Unexpected image size for {raw_image}: {image.size}")
                    stem = f"mvtec_ad_{category.name}_{overlay.stem}"
                    destination_image = output_root / "images" / split / f"{stem}.png"
                    destination_label = output_root / "labels" / split / f"{stem}.txt"
                    destination_image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(raw_image, destination_image)
                    write_yolo_label(destination_label, [corners], width, height)
                    records.append(
                        {
                            "category": category.name,
                            "split": split,
                            "overlay": str(overlay),
                            "source": str(raw_image),
                            "image": str(destination_image),
                            "label": str(destination_label),
                            "corners": corners.tolist(),
                        }
                    )

    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        f"path: {output_root.relative_to(Path.cwd()).as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: screw\n",
        encoding="utf-8",
    )
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "split", "overlay", "source", "image", "label"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "corners"} for row in records)
    counts = {
        split: sum(row["split"] == split for row in records)
        for split in ("train", "val", "test")
    }
    summary = {
        "class_names": ["screw"],
        "images": len(records),
        "objects": len(records),
        "counts_by_split": counts,
        "counts_by_category": {
            category: {split: sum(row["category"] == category and row["split"] == split for row in records) for split in ("train", "val", "test")}
            for category in sorted({str(row["category"]) for row in records})
        },
        "source_root": str(source_root),
        "archive": str(archive),
        "data_yaml": str(data_yaml),
        "manifest": str(manifest),
        "images_use_raw_mvtec": True,
        "labels_recovered_from_green_overlay": True,
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("datasets/mvtec_ad/screw/test"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/mvtec_ad_screw_bbox_obb"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
