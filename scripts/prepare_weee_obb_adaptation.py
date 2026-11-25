#!/usr/bin/env python3
"""Materialize a leakage-free WEEE YOLO-OBB train/val adaptation dataset.

Only Training_set.json and Validation_set.json are materialized. Test_set.json
is read for content-hash auditing and remains locked outside the training YAML.
"""

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

from prepare_screw_yolo_obb import load_records, sanitize_filename, write_yolo_label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash_split_manifest(
    train_paths: Sequence[str | Path],
    validation_paths: Sequence[str | Path],
    test_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Return auditable source-image hashes and reject cross-split duplicates."""

    split_paths = {
        "train": sorted(Path(path).resolve() for path in train_paths),
        "validation": sorted(Path(path).resolve() for path in validation_paths),
        "test_locked": sorted(Path(path).resolve() for path in test_paths),
    }
    split_hashes = {
        split: {str(path): sha256_file(path) for path in paths}
        for split, paths in split_paths.items()
    }
    overlaps = {}
    for first, second in (("train", "validation"), ("train", "test_locked"), ("validation", "test_locked")):
        shared = sorted(set(split_hashes[first].values()) & set(split_hashes[second].values()))
        if shared:
            overlaps[f"{first}__{second}"] = shared
    if overlaps:
        raise ValueError(f"WEEE source-image content leakage detected: {overlaps}")
    entries = {
        split: [
            {"path": path, "sha256": digest, "bytes": Path(path).stat().st_size}
            for path, digest in sorted(hashes.items())
        ]
        for split, hashes in split_hashes.items()
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "policy": "sha256 exact file bytes; train, validation, and locked test must be content-disjoint",
        "counts": {split: len(paths) for split, paths in split_paths.items()},
        "entries": entries,
        "manifest_sha256": manifest_sha256,
        "content_disjoint": True,
    }


def _materialize_split(
    records: Sequence[dict[str, Any]], output_root: Path, split: str, link_images: bool
) -> dict[str, int]:
    object_count = 0
    for record in records:
        source = Path(record["path"]).resolve()
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
        name = f"weee_{digest}_{sanitize_filename(source.name)}"
        image_destination = output_root / "images" / split / name
        label_destination = output_root / "labels" / split / f"{Path(name).stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        if link_images:
            os.link(source, image_destination)
        else:
            shutil.copy2(source, image_destination)
        with Image.open(source) as image:
            width, height = image.size
        object_count += write_yolo_label(label_destination, record["objects"], width, height)
    return {"images": len(records), "objects": object_count}


def prepare_weee_adaptation_dataset(
    weee_root: str | Path,
    train_annotations: str | Path,
    validation_annotations: str | Path,
    test_annotations: str | Path,
    output_root: str | Path,
    link_images: bool = True,
) -> dict[str, Any]:
    root = Path(weee_root).resolve()
    train_annotations = Path(train_annotations).resolve()
    validation_annotations = Path(validation_annotations).resolve()
    test_annotations = Path(test_annotations).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty adaptation dataset: {output_root}")
    records = {
        "train": load_records(root, [train_annotations], "weee", False, None),
        "validation": load_records(root, [validation_annotations], "weee", False, None),
        "test_locked": load_records(root, [test_annotations], "weee", False, None),
    }
    manifest = content_hash_split_manifest(
        [record["path"] for record in records["train"]],
        [record["path"] for record in records["validation"]],
        [record["path"] for record in records["test_locked"]],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    split_counts = {
        "train": _materialize_split(records["train"], output_root, "train", link_images),
        "val": _materialize_split(records["validation"], output_root, "val", link_images),
        "test": {"images": 0, "objects": 0},
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
    annotation_hashes = {
        "train": sha256_file(train_annotations),
        "validation": sha256_file(validation_annotations),
        "test_locked": sha256_file(test_annotations),
    }
    summary = {
        "schema_version": "1.0",
        "class_names": ["screw"],
        "allow_axis_fallback": False,
        "link_images": link_images,
        "train_val_only": True,
        "test_locked": True,
        "source_root": str(root),
        "annotation_files": {
            "train": str(train_annotations),
            "validation": str(validation_annotations),
            "test_locked": str(test_annotations),
        },
        "annotation_sha256": annotation_hashes,
        "splits": split_counts,
        "source_split_manifest": manifest,
        "data_yaml": str(data_yaml),
    }
    (output_root / "adaptation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weee-root", type=Path, default=Path("datasets/external/weee_screw"))
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_weee_adaptation_dataset(
                args.weee_root,
                args.train_annotations,
                args.validation_annotations,
                args.test_annotations,
                args.output_root,
                link_images=not args.copy_images,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
