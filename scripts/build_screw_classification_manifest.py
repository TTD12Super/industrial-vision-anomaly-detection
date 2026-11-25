#!/usr/bin/env python3
"""Build the canonical GOOD/DEFECT manifest and audit its split boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _overlap_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    values_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values_by_split[row["split"]].add(str(row[key]))
    splits = sorted(values_by_split)
    overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            shared = sorted(values_by_split[left] & values_by_split[right])
            if shared:
                overlap[f"{left}__{right}"] = shared
    return overlap


def build_manifest(dataset_root: str | Path, manifest_path: str | Path, audit_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    images = _read_jsonl(root / "manifests" / "images.jsonl")
    objects = _read_jsonl(root / "manifests" / "objects.jsonl")
    objects_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        objects_by_image[str(obj["image_id"])].append(obj)
    if len({row["image_id"] for row in images}) != len(images):
        raise ValueError("images.jsonl contains duplicate image_id values")

    rows: list[dict[str, Any]] = []
    for image in images:
        image_id = str(image["image_id"])
        image_objects = objects_by_image.get(image_id, [])
        if len(image_objects) != 1:
            raise ValueError(f"Expected exactly one object for {image_id}, found {len(image_objects)}")
        obj = image_objects[0]
        label = str(obj["quality_label"])
        if label not in {"GOOD", "DEFECT"}:
            raise ValueError(f"Unexpected quality label for {image_id}: {label}")
        image_path = (root / str(image["relative_path"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(
            {
                "image_path": str(image_path),
                "label": label,
                "source_class": ",".join(obj.get("defect_types", [])) or "GOOD",
                "split": str(image["split"]),
                "image_id": image_id,
                "capture_group_id": str(image["capture_group_id"]),
                "sha256": str(image["sha256"]),
            }
        )

    hash_overlap = _overlap_by(rows, "sha256")
    group_overlap = _overlap_by(rows, "capture_group_id")
    if hash_overlap or group_overlap:
        raise ValueError({"sha256_overlap": hash_overlap, "capture_group_overlap": group_overlap})

    manifest_path = Path(manifest_path).resolve()
    audit_path = Path(audit_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "label", "source_class", "split", "image_id", "capture_group_id", "sha256"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    split_label_counts = Counter((row["split"], row["label"]) for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    defect_counts = Counter(row["source_class"] for row in rows if row["label"] == "DEFECT")
    audit = {
        "schema_version": "1.0",
        "dataset_root": str(root),
        "manifest": str(manifest_path),
        "images": len(rows),
        "unique_sha256": len({row["sha256"] for row in rows}),
        "unique_capture_groups": len({row["capture_group_id"] for row in rows}),
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_counts": {f"{split}__{label}": count for (split, label), count in sorted(split_label_counts.items())},
        "defect_type_counts": dict(sorted(defect_counts.items())),
        "sha256_overlap": hash_overlap,
        "capture_group_overlap": group_overlap,
        "leakage_free_exact_content": not bool(hash_overlap),
        "leakage_free_capture_groups": not bool(group_overlap),
        "source_manifest_sha256": hashlib.sha256((root / "manifests" / "images.jsonl").read_bytes()).hexdigest(),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/products/screw_anomaly_full"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.dataset_root, args.manifest, args.audit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
