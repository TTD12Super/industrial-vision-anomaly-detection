#!/usr/bin/env python3
"""Create a development manifest that trains on prior validation hard negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FIELDS = ["image_path", "label", "source_class", "split", "image_id", "capture_group_id", "sha256"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stratified_calibration(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["source_class"])].append(row)
    rng = random.Random(seed)
    calibration: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    for key in sorted(grouped):
        group = list(grouped[key])
        rng.shuffle(group)
        count = max(1, round(len(group) * fraction)) if len(group) > 1 else 0
        calibration.extend(group[:count])
        train.extend(group[count:])
    if not calibration or not train or {row["label"] for row in calibration} != {"GOOD", "DEFECT"}:
        raise ValueError("Calibration split must contain both labels and leave a non-empty train split")
    return train, calibration


def build_manifest(source_manifest: str | Path, output_manifest: str | Path, audit_path: str | Path, calibration_fraction: float = 0.2, seed: int = 42) -> dict[str, Any]:
    if not 0.05 <= calibration_fraction <= 0.5:
        raise ValueError("calibration_fraction must be between 0.05 and 0.5")
    source_manifest = Path(source_manifest).resolve()
    rows = _read(source_manifest)
    locked_test = [dict(row) for row in rows if row["split"] == "test"]
    original_train = [dict(row) for row in rows if row["split"] == "train"]
    prior_validation = [dict(row) for row in rows if row["split"] == "validation"]
    if not locked_test or not original_train or not prior_validation:
        raise ValueError("Source manifest must contain train, validation and test rows")
    train_rows, calibration_rows = _stratified_calibration(original_train, calibration_fraction, seed)
    # Prior validation is deliberately promoted to training as hard-negative data.
    train_rows.extend(prior_validation)
    for row in train_rows:
        row["split"] = "train"
    for row in calibration_rows:
        row["split"] = "validation"
    for row in locked_test:
        row["split"] = "test"
    output_rows = train_rows + calibration_rows + locked_test
    split_hashes: dict[str, set[str]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        split_hashes[row["split"]].add(row["sha256"])
        split_groups[row["split"]].add(row["capture_group_id"])
    if split_hashes["train"] & split_hashes["validation"] or split_hashes["train"] & split_hashes["test"] or split_hashes["validation"] & split_hashes["test"]:
        raise ValueError("Exact-content leakage detected")
    if split_groups["train"] & split_groups["validation"] or split_groups["train"] & split_groups["test"] or split_groups["validation"] & split_groups["test"]:
        raise ValueError("Capture-group leakage detected")

    output_manifest = Path(output_manifest).resolve()
    audit_path = Path(audit_path).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    audit = {
        "schema_version": "1.0",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "output_manifest": str(output_manifest),
        "policy": "prior validation rows promoted to train; calibration sampled only from prior train; original test remains locked",
        "seed": seed,
        "calibration_fraction_of_prior_train": calibration_fraction,
        "split_counts": dict(Counter(row["split"] for row in output_rows)),
        "split_label_counts": dict(Counter(f"{row['split']}__{row['label']}" for row in output_rows)),
        "locked_test_image_ids": [row["image_id"] for row in locked_test],
        "exact_content_disjoint": True,
        "capture_group_disjoint": True,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.source_manifest, args.output_manifest, args.audit, args.calibration_fraction, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
