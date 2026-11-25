#!/usr/bin/env python3
"""Validate YOLO-OBB labels and optionally render a small visual QA set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def validate(root: Path, render_dir: Path | None = None, limit: int = 12) -> dict:
    errors = []
    summary = {}
    rendered = 0
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(image_dir.glob("*"))
        labels = sorted(label_dir.glob("*.txt"))
        split_errors = []
        objects = 0
        for label in labels:
            image = next((candidate for candidate in images if candidate.stem == label.stem), None)
            if image is None:
                split_errors.append(f"missing image for {label}")
                continue
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            objects += len(rows)
            for row_number, row in enumerate(rows, 1):
                if len(row) != 9:
                    split_errors.append(f"{label}:{row_number}: expected 9 values, got {len(row)}")
                    continue
                try:
                    values = np.asarray([float(value) for value in row], dtype=float)
                except ValueError:
                    split_errors.append(f"{label}:{row_number}: non-numeric value")
                    continue
                if int(values[0]) != 0 or np.any(values[1:] < 0) or np.any(values[1:] > 1):
                    split_errors.append(f"{label}:{row_number}: invalid class or normalized coordinate")
                points = values[1:].reshape(4, 2)
                if abs(cv2.contourArea(points.astype(np.float32))) <= 1e-8:
                    split_errors.append(f"{label}:{row_number}: degenerate polygon")
        summary[split] = {"images": len(images), "labels": len(labels), "objects": objects, "errors": len(split_errors)}
        errors.extend(split_errors)

        if render_dir is not None and rendered < limit:
            render_dir.mkdir(parents=True, exist_ok=True)
            for label in labels:
                if rendered >= limit:
                    break
                image_path = next((candidate for candidate in images if candidate.stem == label.stem), None)
                if image_path is None:
                    continue
                canvas = cv2.imread(str(image_path))
                if canvas is None:
                    continue
                height, width = canvas.shape[:2]
                for row in label.read_text(encoding="utf-8").splitlines():
                    if not row.strip():
                        continue
                    values = np.asarray([float(value) for value in row.split()[1:]], dtype=float).reshape(4, 2)
                    polygon = np.rint(values * [width, height]).astype(np.int32)
                    cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
                cv2.imwrite(str(render_dir / f"{split}_{image_path.name}"), canvas)
                rendered += 1
    result = {"root": str(root.resolve()), "splits": summary, "errors": errors}
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/screw_obb"))
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.render_dir, args.limit), indent=2))
