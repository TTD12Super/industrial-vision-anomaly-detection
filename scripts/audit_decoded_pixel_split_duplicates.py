"""Audit exact decoded-pixel duplicates across MVTec train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import platform

from PIL import Image, __version__ as PILLOW_VERSION


def decoded_pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        digest = hashlib.sha256()
        digest.update(rgba.width.to_bytes(4, "big"))
        digest.update(rgba.height.to_bytes(4, "big"))
        digest.update(rgba.tobytes())
        return digest.hexdigest()


def audit(processed_root: Path, manifest_path: Path) -> dict[str, object]:
    root = processed_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    records: list[tuple[Path, str, str]] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "category", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"manifest must contain columns: {sorted(required)}")
        for row in reader:
            split = row["split"]
            if split not in {"train", "validation"}:
                raise ValueError(f"unsupported manifest split: {split}")
            path = (root / row["image_path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"manifest path escapes processed root: {row['image_path']}")
            records.append((path, row["category"], split))

    for path in sorted(root.glob("*/test/*/*.png")):
        records.append((path, path.parents[2].name, "test"))

    if not records or {split for _, _, split in records} != {"train", "validation", "test"}:
        raise ValueError("expected non-empty train, validation, and test splits")

    counts: dict[str, int] = defaultdict(int)
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    test_inventory: list[str] = []
    dataset_inventory: list[str] = []
    for path, category, split in records:
        counts[split] += 1
        pixel_hash = decoded_pixel_sha256(path)
        relative_path = path.relative_to(root).as_posix()
        hashes[pixel_hash].append(
            (split, category, relative_path)
        )
        dataset_inventory.append(f"{split}\t{category}\t{relative_path}\t{pixel_hash}")
        if split == "test":
            test_inventory.append(f"{relative_path}\t{category}\t{pixel_hash}")

    overlaps = [
        {"first": left, "second": right}
        for group in hashes.values()
        for index, left in enumerate(group)
        for right in group[index + 1 :]
        if left[0] != right[0]
    ]
    return {
        "protocol": "SHA-256 over decoded RGBA pixels and dimensions",
        "audit_date": date.today().isoformat(),
        "decoder": f"Pillow {PILLOW_VERSION}",
        "python_version": platform.python_version(),
        "processed_root": processed_root.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset_inventory_sha256": hashlib.sha256(
            ("\n".join(sorted(dataset_inventory)) + "\n").encode("utf-8")
        ).hexdigest(),
        "test_image_inventory_sha256": hashlib.sha256(
            ("\n".join(sorted(test_inventory)) + "\n").encode("utf-8")
        ).hexdigest(),
        "image_count": len(records),
        "files_by_split": dict(sorted(counts.items())),
        "unique_decoded_pixel_hashes": len(hashes),
        "decoded_pixel_hash_failures": 0,
        "cross_split_identical_pixel_pair_count": len(overlaps),
        "cross_split_identical_pixel_pairs": overlaps,
        "implementation": "scripts/audit_decoded_pixel_split_duplicates.py",
        "limitations": [
            "Does not detect perceptual near-duplicates or acquisition-session leakage.",
            "The official MVTec test split was previously inspected; this audit does not make its benchmark results an untouched-holdout estimate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.processed_root, args.manifest)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
