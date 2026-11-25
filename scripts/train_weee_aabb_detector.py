#!/usr/bin/env python3
"""Train one AABB detector under the shared WEEE comparison protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(
    model_kind: str,
    data: str | Path,
    model: str,
    output_root: str | Path,
    epochs: int = 5,
    image_size: int = 1024,
    batch: int = 2,
    device: str | None = None,
) -> dict[str, object]:
    if model_kind not in {"yolo", "rtdetr"}:
        raise ValueError("model_kind must be yolo or rtdetr")
    if epochs < 1 or image_size < 32 or batch < 1:
        raise ValueError("epochs, image_size and batch must be positive")
    from ultralytics import RTDETR, YOLO

    data = Path(data).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_instance = YOLO(model) if model_kind == "yolo" else RTDETR(model)
    options: dict[str, object] = {
        "task": "detect",
        "data": str(data),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "workers": 0,
        "project": str(output_root),
        "name": model_kind,
        "patience": epochs,
        "seed": 0,
        "deterministic": True,
    }
    if device is not None:
        options["device"] = device
    results = model_instance.train(**options)
    save_dir = Path(str(getattr(results, "save_dir", output_root / model_kind))).resolve()
    checkpoint = save_dir / "weights" / "best.pt"
    summary = {
        "schema_version": "1.0",
        "experiment": "weee_yolo_rtdetr_aabb_comparison",
        "model_kind": model_kind,
        "model": model,
        "data": str(data),
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "save_dir": str(save_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "warning": "AABB candidate only; compare crop metrics, not OBB metrics.",
    }
    (save_dir / "comparison_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("yolo", "rtdetr"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(json.dumps(train(
        args.kind,
        args.data,
        args.model,
        args.output_root,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch=args.batch,
        device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
