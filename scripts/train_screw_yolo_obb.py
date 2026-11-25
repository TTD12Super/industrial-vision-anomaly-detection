#!/usr/bin/env python3
"""Train YOLO-OBB; Ultralytics is intentionally imported only at runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/screw_obb/data.yaml"))
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=Path("runs/screw_yolo_obb"))
    parser.add_argument("--name", default="combined")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Missing data file: {args.data}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install Ultralytics in the GPU environment: pip install ultralytics") from exc

    model = YOLO(args.model)
    options = {
        "data": str(args.data),
        "task": "obb",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project),
        "name": args.name,
        "degrees": 20.0,
        "fliplr": 0.5,
        "close_mosaic": 10,
    }
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)


if __name__ == "__main__":
    main()
