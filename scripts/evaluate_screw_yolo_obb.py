#!/usr/bin/env python3
"""Evaluate a trained YOLO-OBB model on val and test splits."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/screw_obb/data.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("runs/screw_yolo_obb_eval"))
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install Ultralytics in the GPU environment: pip install ultralytics") from exc
    model = YOLO(str(args.weights))
    for split in ("val", "test"):
        options = {"data": str(args.data), "split": split, "plots": True, "project": str(args.project), "name": split}
        if args.device is not None:
            options["device"] = args.device
        print(split, model.val(**options))


if __name__ == "__main__":
    main()
