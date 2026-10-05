"""CLI wrapper for reproducing the saved MVTec baseline metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from mvtec_system import reproduce_saved_baseline, save_reproduction_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/dinov2_knn/final_run"))
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("datasets/mvtec_ad_processed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline_reproduction/baseline_reproduction.json"),
    )
    args = parser.parse_args()
    report = reproduce_saved_baseline(args.run_root, args.processed_root)
    save_reproduction_report(args.output, report)
    print(f"Saved reproduction report to {args.output}")
    print(f"Categories: {report['category_count']}")
    print(f"Max absolute macro delta: {report['max_abs_macro_delta']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
