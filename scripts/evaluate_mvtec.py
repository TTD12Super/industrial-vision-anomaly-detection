"""Offline MVTec AD evaluation CLI (no API, no MongoDB, no thresholds).

Reuses the existing ``model.inference`` pipeline and the existing feature
banks; it never rebuilds a bank and never contacts FastAPI/MongoDB.

Examples::

    python scripts/evaluate_mvtec.py --categories bottle --no-save-maps
    python scripts/evaluate_mvtec.py --categories bottle cable --batch-size 4
    python scripts/evaluate_mvtec.py --all            # all 15 categories

Each invocation creates a brand-new directory
``artifacts/dinov2_knn/eval_<timestamp>/`` and never overwrites a previous
run. ``--limit N`` evaluates only the first N sorted test images per category
and marks the run as a subset/smoke run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("datasets") / "mvtec_ad_processed"
DEFAULT_BANK_DIR = Path("artifacts") / "dinov2_knn" / "banks"
DEFAULT_RUN_ROOT = Path("artifacts") / "dinov2_knn"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--categories",
        nargs="+",
        metavar="CATEGORY",
        help="One or more MVTec categories, e.g. bottle cable.",
    )
    group.add_argument("--all", action="store_true", help="Evaluate all 15 categories.")
    parser.add_argument("--batch-size", type=int, default=8, help="Images per forward pass (default 8).")
    parser.add_argument(
        "--save-maps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist per-category anomaly_maps.npz (default: do not save).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N sorted test images per category (subset/smoke run).",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def resolve_categories(args: argparse.Namespace) -> list[str]:
    from model.inference.config import CATEGORIES, validate_category

    if args.all:
        return list(CATEGORIES)
    requested = list(dict.fromkeys(args.categories))  # dedupe, keep order
    for category in requested:
        try:
            validate_category(category)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
    return requested


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    categories = resolve_categories(args)

    from model.evaluation.evaluate import run_evaluation

    run_dir, results = run_evaluation(
        categories=categories,
        processed_root=args.processed_dir,
        bank_dir=args.bank_dir,
        run_root=args.run_root,
        batch_size=args.batch_size,
        limit=args.limit,
        save_maps=args.save_maps,
        device=args.device,
    )

    print(f"\nEvaluation run: {run_dir}")
    print(
        f"{'category':11}{'status':>8}{'test':>6}{'normal':>7}{'defect':>7}"
        f"{'image_auroc':>13}{'image_ap':>10}{'pixel_auroc':>13}{'pixel_ap':>10}{'seconds':>9}"
    )
    for result in results:
        print(
            f"{result.category:11}{result.status:>8}{result.test_images:>6}"
            f"{result.normal_images:>7}{result.defective_images:>7}"
            f"{_cell(result.image_auroc):>13}{_cell(result.image_ap):>10}"
            f"{_cell(result.pixel_auroc):>13}{_cell(result.pixel_ap):>10}"
            f"{result.pipeline_seconds:>9.1f}"
        )
    failed = [result for result in results if result.status != "ok"]
    if failed:
        print(
            f"\n{len(failed)} categor{'y' if len(failed) == 1 else 'ies'} failed; "
            f"see {run_dir / 'failures.json'}."
        )
        return 1
    if args.limit is not None:
        print(
            f"\nSUBSET/SMOKE RUN: only the first {args.limit} sorted test images per "
            "category were evaluated; these are not full-category metrics."
        )
    print("\nNo thresholds were selected and no test labels were used for calibration.")
    return 0


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    sys.exit(main())
