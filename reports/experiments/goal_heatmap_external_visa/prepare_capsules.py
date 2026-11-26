"""Prepare only the official VisA 1-class capsules split in MVTec layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

from PIL import Image


SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
EXPECTED_SPLIT_SHA256 = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
EXPECTED_COUNTS = {"train_good": 542, "test_good": 60, "test_bad": 100, "masks": 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Extracted VisA archive root containing capsules/")
    parser.add_argument("--split-csv", type=Path, required=True, help="Official split_csv/1cls.csv")
    parser.add_argument("--output-root", type=Path, required=True, help="New, empty MVTec-style output directory")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    with args.split_csv.open("rb") as split_file:
        split_hash = hashlib.file_digest(split_file, "sha256").hexdigest()
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise SystemExit(f"Unexpected official split CSV SHA-256: {split_hash}")
    if not (raw_root / "capsules").is_dir():
        raise SystemExit(f"Missing capsules directory under {raw_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_root}")

    destinations = {
        "train_good": output_root / "capsules/train/good",
        "test_good": output_root / "capsules/test/good",
        "test_bad": output_root / "capsules/test/bad",
        "masks": output_root / "capsules/ground_truth/bad",
    }
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)

    counts = {key: 0 for key in EXPECTED_COUNTS}
    with args.split_csv.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["object", "split", "label", "image", "mask"]:
            raise SystemExit(f"Unexpected split CSV columns: {rows.fieldnames}")
        for row in rows:
            if row["object"] != "capsules":
                continue
            image_rel = PurePosixPath(row["image"])
            if image_rel.is_absolute() or ".." in image_rel.parts:
                raise SystemExit(f"Unsafe image path in split CSV: {row['image']}")
            source = (raw_root / image_rel).resolve()
            if not source.is_relative_to(raw_root) or not source.is_file():
                raise SystemExit(f"Missing or unsafe source image: {source}")

            split, label = row["split"], row["label"]
            if split == "train" and label == "normal":
                kind, destination = "train_good", destinations["train_good"] / source.name
            elif split == "test" and label == "normal":
                kind, destination = "test_good", destinations["test_good"] / source.name
            elif split == "test" and label == "anomaly":
                kind, destination = "test_bad", destinations["test_bad"] / source.name
            else:
                raise SystemExit(f"Unexpected official split row: {row}")
            if destination.exists():
                raise SystemExit(f"Duplicate destination in official split: {destination}")
            with Image.open(source) as image:
                image.verify()
            shutil.copyfile(source, destination)
            counts[kind] += 1

            if kind == "test_bad":
                mask_rel = PurePosixPath(row["mask"])
                if not row["mask"] or mask_rel.is_absolute() or ".." in mask_rel.parts:
                    raise SystemExit(f"Invalid anomaly mask path: {row['mask']}")
                mask_source = (raw_root / mask_rel).resolve()
                if not mask_source.is_relative_to(raw_root) or not mask_source.is_file():
                    raise SystemExit(f"Missing or unsafe mask: {mask_source}")
                with Image.open(mask_source) as image:
                    binary = image.convert("L").point(lambda value: 255 if value else 0)
                    binary.save(destinations["masks"] / f"{source.stem}_mask.png")
                counts["masks"] += 1

    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Official capsules split count mismatch: {counts} != {EXPECTED_COUNTS}")
    provenance = {
        "dataset": "VisA capsules, official 1-class split",
        "split_url": SPLIT_URL,
        "split_csv_sha256": split_hash,
        "counts": counts,
        "mask_conversion": "all nonzero source mask values mapped to 255, per official prepare_data.py",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), **provenance}, indent=2))


if __name__ == "__main__":
    main()
