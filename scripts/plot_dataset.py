"""Create the dataset split-size chart from the Python validation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from mvtec_system import plot_dataset_distribution, validate_raw_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/mvtec_ad"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_validation/dataset_split_sizes.png"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_raw_dataset(args.raw_root)
    path = plot_dataset_distribution(report, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
