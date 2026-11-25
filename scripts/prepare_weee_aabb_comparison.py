#!/usr/bin/env python3
"""Materialize a content-disjoint WEEE AABB view for YOLO/RT-DETR comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_screw_yolo_obb import load_records, sanitize_filename  # noqa: E402
from prepare_weee_obb_adaptation import content_hash_split_manifest, sha256_file  # noqa: E402
from train_screw_rtdetr import corners_to_normalized_xywh  # noqa: E402


def _materialize_split(records: Sequence[dict[str, Any]], output_root: Path, split: str) -> dict[str, int]:
    objects = 0
    for record in records:
        source = Path(record["path"]).resolve()
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
        name = f"weee_{digest}_{sanitize_filename(source.name)}"
        destination = output_root / "images" / split / name
        label = output_root / "labels" / split / f"{Path(name).stem}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source, destination)
        with Image.open(source) as image:
            width, height = image.size
        lines = []
        for corners in record["objects"]:
            xywh = corners_to_normalized_xywh(corners, width, height)
            lines.append(" ".join(["0", *(f"{value:.7f}" for value in xywh)]))
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        objects += len(lines)
    return {"images": len(records), "objects": objects}


def prepare_weee_aabb_comparison(
    weee_root: str | Path,
    train_annotations: str | Path,
    validation_annotations: str | Path,
    test_annotations: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(weee_root).resolve()
    annotation_paths = {
        "train": Path(train_annotations).resolve(),
        "validation": Path(validation_annotations).resolve(),
        "test_locked": Path(test_annotations).resolve(),
    }
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_root}")
    records = {
        split: load_records(root, [path], "weee", False, None)
        for split, path in annotation_paths.items()
    }
    manifest = content_hash_split_manifest(
        [row["path"] for row in records["train"]],
        [row["path"] for row in records["validation"]],
        [row["path"] for row in records["test_locked"]],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": _materialize_split(records["train"], output_root, "train"),
        "val": _materialize_split(records["validation"], output_root, "val"),
        "test": _materialize_split(records["test_locked"], output_root, "test"),
    }
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: screw\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "experiment": "weee_yolo_rtdetr_aabb_comparison",
        "geometry": "axis_aligned_bbox_from_segmentation_min_area_obb_envelope",
        "class_names": ["screw"],
        "train_val_test_materialized": True,
        "test_locked": True,
        "source_root": str(root),
        "annotation_files": {key: str(path) for key, path in annotation_paths.items()},
        "annotation_sha256": {key: sha256_file(path) for key, path in annotation_paths.items()},
        "splits": counts,
        "source_split_manifest": manifest,
        "data_yaml": str(data_yaml),
        "warning": "AABB comparison only; it is not a direct replacement for YOLO-OBB oriented cropping.",
    }
    (output_root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weee-root", type=Path, default=Path("datasets/external/weee_screw"))
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_weee_aabb_comparison(
        args.weee_root,
        args.train_annotations,
        args.validation_annotations,
        args.test_annotations,
        args.output_root,
    ), indent=2))


if __name__ == "__main__":
    main()
