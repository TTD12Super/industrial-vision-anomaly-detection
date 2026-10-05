"""Validate and preprocess an MVTec-style dataset without mutating raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvtec_system import prepare_mvtec_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/mvtec_ad"))
    parser.add_argument(
        "--processed-root", type=Path, default=Path("datasets/mvtec_ad_processed")
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="rewrite processed PNGs")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_mvtec_dataset(
        args.raw_root,
        args.processed_root,
        args.image_size,
        overwrite=args.force,
    )
    payload = json.dumps(report, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
