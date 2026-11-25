#!/usr/bin/env python3
"""Create synthetic YOLO-OBB training images from labelled screw scenes.

The augmentation is applied to full scenes so YOLO still learns localization:
the scene is rotated/scaled, while each labelled OBB also receives local
brightness/blur/noise changes.  Crop-only images are not used as detector
training samples because they remove the background and object placement.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from screw_yolo_obb import polygon_area, write_yolo_label


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_objects(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    objects = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 9 or int(values[0]) != 0:
            continue
        points = np.asarray(values[1:], dtype=np.float32).reshape(4, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        objects.append(points)
    return objects


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homogeneous @ matrix.T)[:, :2].astype(np.float32)


def scene_transform(
    image: np.ndarray,
    objects: list[np.ndarray],
    rng: random.Random,
) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-35.0, 35.0)
    scale = rng.uniform(0.88, 1.12)
    matrix_2x3 = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix_2x3[0, 2] += rng.uniform(-0.04, 0.04) * width
    matrix_2x3[1, 2] += rng.uniform(-0.04, 0.04) * height
    transformed = cv2.warpAffine(
        image,
        matrix_2x3,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    matrix = np.vstack([matrix_2x3, [0.0, 0.0, 1.0]])
    return transformed, [transform_points(points, matrix) for points in objects]


def local_appearance(
    image: np.ndarray,
    objects: list[np.ndarray],
    rng: random.Random,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for points in objects:
        if rng.random() > 0.75:
            continue
        polygon = np.rint(points).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        if rng.random() < 0.55:
            blurred = cv2.GaussianBlur(output, (5, 5), 0)
            output = np.where(mask[..., None] > 0, blurred, output).astype(np.uint8)
        if rng.random() < 0.75:
            factor = rng.uniform(0.75, 1.25)
            adjusted = np.clip(output.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            output = np.where(mask[..., None] > 0, adjusted, output).astype(np.uint8)
        if rng.random() < 0.35:
            noise = rng.uniform(1.0, 8.0)
            noisy = np.clip(output.astype(np.float32) + np.random.default_rng(rng.randrange(1_000_000)).normal(0, noise, output.shape), 0, 255).astype(np.uint8)
            output = np.where(mask[..., None] > 0, noisy, output).astype(np.uint8)
    if rng.random() < 0.25:
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return output


def valid_objects(objects: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    valid = []
    for points in objects:
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        if polygon_area(clipped) > 4.0:
            valid.append(clipped)
    return valid


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def augment_dataset(input_root: Path, output_root: Path, fraction: float, seed: int) -> dict[str, int]:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    rng = random.Random(seed)
    counts = {"original_train": 0, "augmented_train": 0, "original_val": 0, "original_test": 0}
    for split in ("train", "val", "test"):
        image_dir = input_root / "images" / split
        label_dir = input_root / "labels" / split
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            link_or_copy(image_path, output_root / "images" / split / image_path.name)
            label_destination = output_root / "labels" / split / label_path.name
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, label_destination)
            counts[f"original_{split}"] += 1
            if split != "train" or rng.random() > fraction:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            objects = read_objects(label_path, width, height)
            if not objects:
                continue
            augmented, transformed_objects = scene_transform(image, objects, rng)
            augmented = local_appearance(augmented, transformed_objects, rng)
            transformed_objects = valid_objects(transformed_objects, width, height)
            if not transformed_objects:
                continue
            name = f"aug_{counts['augmented_train']:05d}_{image_path.name}"
            destination = output_root / "images" / "train" / name
            label_destination = output_root / "labels" / "train" / f"{Path(name).stem}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), augmented)
            write_yolo_label(label_destination, transformed_objects, width, height)
            counts["augmented_train"] += 1
    relative_root = Path(os.path.relpath(output_root, Path.cwd())).as_posix()
    (output_root / "data.yaml").write_text(
        f"path: {relative_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: screw\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.fraction <= 1.0:
        raise SystemExit("--fraction must be between 0 and 1")
    print(augment_dataset(args.input_root.resolve(), args.output_root.resolve(), args.fraction, args.seed))


if __name__ == "__main__":
    main()
