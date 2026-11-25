#!/usr/bin/env python3
"""Train an independent RT-DETR axis-aligned-box candidate from canonical manifests.

This experiment converts each canonical OBB to its enclosing axis-aligned box
(AABB) for standard object detection. It is a candidate comparison only: the
result must not be scored with OBB metrics or treated as a direct YOLO-OBB
replacement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable
import warnings

import yaml

from inspection.config import ProductConfig, load_product_config


AABB_WARNING = (
    "RT-DETR candidate uses axis-aligned boxes derived from OBB corners; "
    "do not compare it with OBB metrics or treat it as a direct YOLO-OBB replacement."
)
EXPECTED_IMAGE_COUNTS = {"train": 90, "val": 18, "test": 12}


def corners_to_normalized_xywh(
    corners: Iterable[Iterable[float]], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert four pixel-space corners to clipped, normalized AABB ``xywh``."""

    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    try:
        points = [(float(point[0]), float(point[1])) for point in corners]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("corners must contain four finite (x, y) points") from error
    if len(points) != 4 or any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("corners must contain four finite (x, y) points")
    x_min = max(0.0, min(point[0] for point in points))
    y_min = max(0.0, min(point[1] for point in points))
    x_max = min(float(width), max(point[0] for point in points))
    y_max = min(float(height), max(point[1] for point in points))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("corners do not form a positive in-image bounding box")
    return (
        ((x_min + x_max) / 2.0) / width,
        ((y_min + y_max) / 2.0) / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    )


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to overwrite RT-DETR input: {destination}")
    destination.symlink_to(source.resolve())


def _prepare_output_root(output_root: Path) -> None:
    if output_root.is_symlink() or (output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir()))):
        raise FileExistsError(f"Refusing to overwrite populated RT-DETR output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_rtdetr_dataset(config: ProductConfig, output_root: str | Path) -> dict[str, Any]:
    """Materialize a symlinked AABB detection view without touching source data."""

    warnings.warn(AABB_WARNING, RuntimeWarning, stacklevel=2)
    from inspection.dataset import load_dataset_records

    images, objects = load_dataset_records(config)
    output_root = Path(output_root)
    _prepare_output_root(output_root)
    objects_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for object_row in objects:
        objects_by_image[object_row["image_id"]].append(object_row)

    counts = {split: 0 for split in EXPECTED_IMAGE_COUNTS}
    object_counts = {split: 0 for split in EXPECTED_IMAGE_COUNTS}
    for image in images:
        split = {"validation": "val"}.get(image["split"], image["split"])
        if split not in counts:
            raise ValueError(f"Unsupported canonical split: {image['split']}")
        source = config.dataset_path(Path(image["relative_path"]))
        if not source.is_file():
            raise FileNotFoundError(f"Missing canonical image: {source}")
        suffix = source.suffix.lower() or ".png"
        _symlink(source, output_root / "images" / split / f"{image['image_id']}{suffix}")

        lines: list[str] = []
        for object_row in objects_by_image[image["image_id"]]:
            xywh = corners_to_normalized_xywh(
                object_row["obb_xyxyxyxy"], int(image["width"]), int(image["height"])
            )
            lines.append(" ".join(["0", *(f"{value:.7f}" for value in xywh)]))
        label = output_root / "labels" / split / f"{image['image_id']}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1
        object_counts[split] += len(lines)

    if counts != EXPECTED_IMAGE_COUNTS:
        raise ValueError(f"Expected canonical image split {EXPECTED_IMAGE_COUNTS}, got {counts}")

    data_yaml = output_root / "data.yaml"
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: config.product_key},
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    summary = {
        "schema_version": "1.0",
        "experiment": "screw_rtdetr_aabb_candidate",
        "product_key": config.product_key,
        "config": str(config.source_path),
        "root": str(output_root.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "geometry": "axis_aligned_bbox",
        "source_geometry": "obb_xyxyxyxy",
        "image_counts": counts,
        "object_counts": object_counts,
        "warning": AABB_WARNING,
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def train_rtdetr(
    config: ProductConfig,
    output_root: str | Path,
    model: str = "rtdetr-l.pt",
    epochs: int = 100,
    image_size: int = 1024,
    batch: int = 4,
    device: str | None = None,
) -> dict[str, Any]:
    """Train RT-DETR on the isolated AABB materialization."""

    if epochs < 1 or image_size < 32 or batch < 1:
        raise ValueError("epochs, image_size and batch must be positive; image_size must be at least 32")
    output_root = Path(output_root).resolve()
    summary = materialize_rtdetr_dataset(config, output_root)
    training_directory = (output_root / "training").resolve()
    expected_checkpoint = training_directory / "run" / "weights" / "best.pt"
    training_args: dict[str, Any] = {
        "model": model,
        "task": "detect",
        "data": summary["data_yaml"],
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "project": str(training_directory),
        "name": "run",
        "expected_checkpoint": str(expected_checkpoint.resolve()),
        "warning": AABB_WARNING,
    }
    if device is not None:
        training_args["device"] = device
    (output_root / "training_args.json").write_text(json.dumps(training_args, indent=2) + "\n", encoding="utf-8")

    try:
        from ultralytics import RTDETR
    except ImportError as error:
        raise RuntimeError("ultralytics is required to train RT-DETR") from error
    model_instance = RTDETR(model)
    results = model_instance.train(**{key: value for key, value in training_args.items() if key not in {"expected_checkpoint", "warning"}})
    save_directory = Path(getattr(results, "save_dir", training_directory / "run"))
    checkpoint = save_directory / "weights" / "best.pt"
    training_summary = {
        **summary,
        "training_args": training_args,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_exists": checkpoint.is_file(),
    }
    (output_root / "training_summary.json").write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")
    return training_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/products/screw.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/screw_rtdetr_aabb"))
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = train_rtdetr(
        load_product_config(args.config),
        args.output_root,
        model=args.model,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
