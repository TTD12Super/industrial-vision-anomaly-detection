#!/usr/bin/env python3
"""Convert WEEE/MVTec COCO-like screw annotations to YOLO-OBB.

The converter keeps one class (``screw``).  WEEE segmentation polygons are
converted to true minimum-area rotated rectangles.  Axis-aligned fallback is
disabled by default to prevent accidentally training fake OBB labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from screw_yolo_obb import (
    annotation_to_corners,
    is_screw_category,
    mvtec_type_number,
    read_json,
    sanitize_filename,
    write_yolo_label,
)


def find_jsons(root: Path) -> list[Path]:
    candidates = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
            candidates.append(path)
    # MVTec ships one complete JSON plus train/val/test JSONs.  Prefer the
    # split files when all three are present so annotations are not counted
    # twice when the root is passed without an explicit --*-annotations list.
    split_candidates = [
        path
        for path in candidates
        if any(token in path.stem.lower() for token in ("train", "val", "test"))
    ]
    if len(split_candidates) >= 3:
        return split_candidates
    return candidates


def resolve_image(root: Path, annotation_file: Path, file_name: str) -> Path | None:
    raw = Path(file_name)
    split = next(
        (token for token in ("train", "val", "test") if token in annotation_file.stem.lower()),
        None,
    )
    candidates = [raw]
    if split is not None:
        candidates.append(root / "images" / split / raw)
    candidates.extend([root / "images" / raw, root / raw, annotation_file.parent / raw])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    image_root = root / "images"
    image_matches = sorted(image_root.rglob(raw.name)) if image_root.is_dir() else []
    if image_matches:
        return image_matches[0].resolve()
    matches = sorted(
        path
        for path in root.rglob(raw.name)
        if "Annotations" not in path.parts and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    return matches[0].resolve() if matches else None


def load_records(
    root: Path,
    annotation_files: list[Path],
    source: str,
    allow_axis_fallback: bool,
    mvtec_types: set[int] | None,
) -> list[dict[str, Any]]:
    records: dict[Path, dict[str, Any]] = {}
    for annotation_file in annotation_files:
        data = read_json(annotation_file)
        categories = {item["id"]: str(item.get("name", "")) for item in data.get("categories", [])}
        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            category_name = categories.get(annotation.get("category_id"), "screw")
            selected = is_screw_category(category_name)
            if source == "mvtec":
                type_number = mvtec_type_number(category_name)
                selected = mvtec_types is None or type_number in mvtec_types
            if selected:
                annotations_by_image[annotation.get("image_id")].append(annotation)
        for image in data.get("images", []):
            image_path = resolve_image(root, annotation_file, image.get("file_name", ""))
            if image_path is None:
                raise FileNotFoundError(f"Image not found for {annotation_file}: {image.get('file_name')}")
            objects = []
            for annotation in annotations_by_image.get(image.get("id"), []):
                objects.append(annotation_to_corners(annotation, allow_axis_fallback=allow_axis_fallback))
            if not objects:
                continue
            record = records.setdefault(
                image_path,
                {"path": image_path, "source": source, "objects": [], "split_hint": annotation_file.stem.lower()},
            )
            existing = {
                tuple(round(float(value), 3) for value in corners.reshape(-1))
                for corners in record["objects"]
            }
            for corners in objects:
                key = tuple(round(float(value), 3) for value in corners.reshape(-1))
                if key not in existing:
                    record["objects"].append(corners)
                    existing.add(key)
    return list(records.values())


def assign_splits(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    output = {"train": [], "val": [], "test": []}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    for source_records in by_source.values():
        hinted = {record["split_hint"] for record in source_records}
        if any(split in hint for hint in hinted for split in ("train", "val", "test")):
            for record in source_records:
                hint = record["split_hint"]
                split = next((name for name in ("train", "val", "test") if name in hint), None)
                if split is None:
                    output["train"].append(record)
                else:
                    output[split].append(record)
            continue
        shuffled = list(source_records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_val = max(1, round(n * 0.1)) if n >= 5 else 0
        output["test"].extend(shuffled[:n_test])
        output["val"].extend(shuffled[n_test : n_test + n_val])
        output["train"].extend(shuffled[n_test + n_val :])
    return output


def convert(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    all_records = []
    source_summaries = {}
    for source, root, explicit in (("weee", args.weee_root, args.weee_annotations), ("mvtec", args.mvtec_root, args.mvtec_annotations)):
        if root is None:
            continue
        root = root.resolve()
        annotation_files = [path.resolve() for path in explicit] if explicit else find_jsons(root)
        if not annotation_files:
            raise FileNotFoundError(f"No COCO-like JSON annotations found under {root}")
        records = load_records(root, annotation_files, source, args.allow_axis_fallback, args.mvtec_types)
        source_summaries[source] = {"root": str(root), "annotation_files": [str(p) for p in annotation_files], "images": len(records)}
        all_records.extend(records)
    if not all_records:
        raise RuntimeError("No screw records found")

    splits = assign_splits(all_records, args.seed)
    counts = {"images": {}, "objects": {}}
    for split, records in splits.items():
        object_count = 0
        for record in records:
            source = record["source"]
            digest = hashlib.sha1(str(record["path"]).encode("utf-8")).hexdigest()[:10]
            name = f"{source}_{digest}_{sanitize_filename(record['path'].name)}"
            image_destination = output_root / "images" / split / name
            label_destination = output_root / "labels" / split / f"{Path(name).stem}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            if args.link_images:
                os.link(record["path"], image_destination)
            else:
                shutil.copy2(record["path"], image_destination)
            with Image.open(record["path"]) as image:
                width, height = image.size
            object_count += write_yolo_label(label_destination, record["objects"], width, height)
        counts["images"][split] = len(records)
        counts["objects"][split] = object_count

    data_yaml = output_root / "data.yaml"
    relative_dataset_root = Path(os.path.relpath(output_root, Path.cwd())).as_posix()
    data_yaml.write_text(
        f"path: {relative_dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: screw\n",
        encoding="utf-8",
    )
    summary = {
        "class_names": ["screw"],
        "allow_axis_fallback": args.allow_axis_fallback,
        "link_images": args.link_images,
        "seed": args.seed,
        "mvtec_screw_types": sorted(args.mvtec_types) if args.mvtec_types is not None else "all",
        "sources": source_summaries,
        "splits": counts,
        "data_yaml": str(data_yaml),
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weee-root", type=Path)
    parser.add_argument("--weee-annotations", type=Path, action="append")
    parser.add_argument("--mvtec-root", type=Path)
    parser.add_argument("--mvtec-annotations", type=Path, action="append")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/screw_obb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-axis-fallback", action="store_true")
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Hard-link source images instead of copying them; useful for local smoke tests",
    )
    parser.add_argument(
        "--mvtec-types",
        default="1,2,3,4,5,6,8,13",
        help="MVTec type numbers to keep as screw (or 'all'); default excludes nut types",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.weee_root is None and arguments.mvtec_root is None:
        raise SystemExit("Provide --weee-root and/or --mvtec-root")
    if arguments.mvtec_types.strip().lower() == "all":
        arguments.mvtec_types = None
    else:
        try:
            arguments.mvtec_types = {int(value.strip()) for value in arguments.mvtec_types.split(",") if value.strip()}
        except ValueError as exc:
            raise SystemExit("--mvtec-types must be comma-separated integers or 'all'") from exc
    print(json.dumps(convert(arguments), indent=2))
