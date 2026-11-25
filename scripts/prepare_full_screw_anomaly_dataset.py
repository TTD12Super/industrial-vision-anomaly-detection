"""Prepare the full MVTec AD screw pool for classifier/localizer experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inspection.geometry import order_corners


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/mvtec_ad/screw"
PILOT_MANIFEST = ROOT / "datasets/mvtec_ad_screw_bbox_obb/manifest.csv"
OUTPUT = ROOT / "datasets/products/screw_anomaly_full"
CATEGORIES = ("good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.symlink_to(source.resolve())


def pilot_boxes() -> dict[Path, list[list[float]]]:
    boxes: dict[Path, list[list[float]]] = {}
    with PILOT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values = [float(value) for value in Path(row["label"]).read_text(encoding="utf-8").split()]
            image = Path(row["source"])
            width, height = Image.open(image).size
            boxes[image.resolve()] = [[values[i] * width, values[i + 1] * height] for i in range(1, 9, 2)]
    return boxes


def auto_box(image_path: Path) -> list[list[float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    background = cv2.GaussianBlur(image, (0, 0), 21)
    difference = cv2.absdiff(image, background)
    _, binary = cv2.threshold(difference, 12, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"Cannot segment screw foreground: {image_path}")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.01:
        raise ValueError(f"Foreground too small for {image_path}")
    rectangle = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    center = rectangle.mean(axis=0)
    rectangle = center + (rectangle - center) * 1.06
    rectangle[:, 0] = np.clip(rectangle[:, 0], 0, image.shape[1] - 1)
    rectangle[:, 1] = np.clip(rectangle[:, 1], 0, image.shape[0] - 1)
    return order_corners(rectangle).astype(float).tolist()


def split_for_category(paths: list[Path]) -> dict[Path, str]:
    total = len(paths)
    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.15)
    return {path: ("train" if index < train_end else "validation" if index < validation_end else "test") for index, path in enumerate(paths)}


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated dataset: {OUTPUT}")
    boxes = pilot_boxes()
    OUTPUT.mkdir(parents=True)
    image_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for category in CATEGORIES:
        if category == "good":
            paths = sorted((SOURCE / "train/good").glob("*.png")) + sorted((SOURCE / "test/good").glob("*.png"))
        else:
            paths = sorted((SOURCE / "test" / category).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No images found for {category}")
        assignments = split_for_category(paths)
        for source in paths:
            split = assignments[source]
            source_partition = source.parent.parent.name
            image_id = f"mvtec_ad_full_{category}_{source_partition}_{source.stem}"
            relative_image = Path("images") / split / f"{image_id}.png"
            symlink(source, OUTPUT / relative_image)
            with Image.open(source) as image:
                width, height = image.size
            mask_relative: str | None = None
            if category != "good":
                mask_source = SOURCE / "ground_truth" / category / f"{source.stem}_mask.png"
                if not mask_source.is_file():
                    raise FileNotFoundError(mask_source)
                relative_mask = Path("masks") / category / f"{image_id}_mask.png"
                symlink(mask_source, OUTPUT / relative_mask)
                mask_relative = relative_mask.as_posix()
            obb = boxes.get(source.resolve(), auto_box(source))
            image_rows.append({
                "image_id": image_id,
                "relative_path": relative_image.as_posix(),
                "split": split,
                "source_dataset": "mvtec_ad_screw_full_anomaly_pool",
                "capture_group_id": f"mvtec_ad_screw_full/{source_partition}/{category}/{source.stem}",
                "domain_id": "mvtec_ad_clean_background",
                "sha256": sha256(source),
                "width": width,
                "height": height,
            })
            object_rows.append({
                "object_id": f"{image_id}_obj01",
                "image_id": image_id,
                "product_key": "screw",
                "obb_xyxyxyxy": obb,
                "quality_label": "GOOD" if category == "good" else "DEFECT",
                "defect_types": [] if category == "good" else [category],
                "mask_relative_path": mask_relative,
            })
            counts[split] += 1
    manifests = OUTPUT / "manifests"
    manifests.mkdir()
    (manifests / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in image_rows), encoding="utf-8")
    (manifests / "objects.jsonl").write_text("".join(json.dumps(row) + "\n" for row in object_rows), encoding="utf-8")
    (OUTPUT / "dataset.yaml").write_text(json.dumps({"schema_version": 1, "product_key": "screw", "split_counts": counts}, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "source_root": str(SOURCE.resolve()),
        "images": len(image_rows),
        "counts": counts,
        "purpose": "GOOD/DEFECT classifier and three heatmap methods; not YOLO training",
        "yolo_dataset": "datasets/products/screw",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "images": len(image_rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
