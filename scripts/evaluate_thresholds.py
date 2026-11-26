"""Offline evaluation of the frozen Phase 4.7C thresholds (read-only).

Reuses the unchanged production scoring path (``evaluate_category``), applies
each category's already-calibrated threshold with the strict rule
``image_score > threshold``, and never selects, recalibrates, or modifies a
threshold. Feature banks are verified against the Phase 4.7D baseline manifest
before and after scoring (fail closed). No FastAPI / MongoDB interaction and
no bank rebuilding.

Examples::

    py scripts/evaluate_thresholds.py --preflight-only
    py scripts/evaluate_thresholds.py
    py scripts/evaluate_thresholds.py --batch-size 4

Each invocation creates a brand-new directory
``artifacts/dinov2_knn/threshold_eval_<timestamp>/`` and never overwrites a
previous run (including Phase 4.5/4.6 evaluations and Phase 4.7C calibration).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_RUN = (
    Path("artifacts") / "dinov2_knn" / "calibration_20261021T100030_338275Z"
)
DEFAULT_BASELINE_MANIFEST = Path("artifacts") / "dinov2_knn" / "bank_sha256_baseline.json"
DEFAULT_REFERENCE_EVAL = Path("artifacts") / "dinov2_knn" / "eval_20261018T144913_671963Z"
DEFAULT_PROCESSED_DIR = Path("datasets") / "mvtec_ad_processed"
DEFAULT_BANK_DIR = Path("artifacts") / "dinov2_knn" / "banks"
DEFAULT_RUN_ROOT = Path("artifacts") / "dinov2_knn"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--calibration-run", type=Path, default=DEFAULT_CALIBRATION_RUN)
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument(
        "--reference-eval-dir",
        type=Path,
        default=DEFAULT_REFERENCE_EVAL,
        help="Phase 4.6 eval directory for read-only scoring-consistency comparison.",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--batch-size", type=int, default=8, help="Images per forward pass (default 8).")
    parser.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto).")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify thresholds, bank baseline, and dataset availability, then exit.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def preflight(args: argparse.Namespace) -> int:
    """Read-only verification; never creates a run directory."""
    from model.evaluation.threshold_evaluation import (
        BaselineIntegrityError,
        ThresholdDataError,
        load_frozen_thresholds,
        verify_bank_baseline,
    )
    from model.evaluation.metrics import image_files, image_labels
    from model.inference.config import CATEGORIES

    expected = sorted(CATEGORIES)
    try:
        thresholds = load_frozen_thresholds(args.calibration_run, expected)
        banks = verify_bank_baseline(args.bank_dir, args.baseline_manifest, expected)
    except (ThresholdDataError, BaselineIntegrityError) as exc:
        print(f"PREFLIGHT STOP: {exc}")
        return 2

    totals = {"test": 0, "normal": 0, "defective": 0}
    for category in expected:
        paths = image_files(Path(args.processed_dir) / category / "test")
        labels = image_labels(paths)
        if not paths or labels.all() or (~labels).all():
            print(f"PREFLIGHT STOP: {category} test split unusable.")
            return 2
        totals["test"] += len(paths)
        totals["normal"] += int((~labels).sum())
        totals["defective"] += int(labels.sum())

    print("PREFLIGHT OK (read-only; no run directory created)")
    print(f"  calibration run : {args.calibration_run}")
    print(f"  baseline        : {args.baseline_manifest}")
    print(f"  thresholds      : {len(thresholds)} categories, alpha={thresholds[expected[0]].alpha}")
    print(f"  banks verified  : {len(banks)}/15 match the manifest (sha256 + size)")
    print(f"  test set        : {totals}")
    print(f"  {'category':11} {'threshold':>10} {'k':>5} {'m':>5}  bank_sha256 (prefix)")
    for category in expected:
        entry = thresholds[category]
        print(
            f"  {category:11} {entry.threshold:>10.6f} {entry.index_k:>5} "
            f"{entry.train_images:>5}  {banks[category]['sha256'][:16]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.preflight_only:
        return preflight(args)

    from model.evaluation.threshold_evaluation import (
        AGGREGATE_METRIC_KEYS,
        BaselineIntegrityError,
        DECISION_RULE_APPLIED,
        ThresholdDataError,
        run_threshold_evaluation,
    )
    from model.inference.config import CATEGORIES

    try:
        run_dir, results = run_threshold_evaluation(
            categories=sorted(CATEGORIES),
            calibration_run=args.calibration_run,
            bank_baseline_manifest=args.baseline_manifest,
            processed_root=args.processed_dir,
            bank_dir=args.bank_dir,
            run_root=args.run_root,
            batch_size=args.batch_size,
            device=args.device,
            reference_eval_dir=args.reference_eval_dir,
        )
    except (ThresholdDataError, BaselineIntegrityError) as exc:
        print(f"\nSTOP: {exc}")
        return 2

    print(f"\nThreshold evaluation run: {run_dir}")
    print(f"Decision rule: {DECISION_RULE_APPLIED}")
    print(
        f"{'category':11}{'normal':>7}{'defect':>7}{'threshold':>11}{'FP':>5}{'FN':>5}"
        f"{'fpr':>10}{'recall':>10}{'precision':>10}{'f1':>10}"
    )
    for result in results:
        m = result.metrics
        print(
            f"{result.category:11}{result.n_normal:>7}{result.n_defective:>7}"
            f"{result.threshold_entry.threshold:>11.6f}{result.fp:>5}{result.fn:>5}"
            f"{_fmt(m['fpr']):>10}{_fmt(m['recall']):>10}"
            f"{_fmt(m['precision']):>10}{_fmt(m['f1']):>10}"
        )

    from model.evaluation.threshold_evaluation import build_summary

    import json

    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    summary = build_summary(results, failures)
    print("\nMacro averages (unweighted; undefined excluded, counted, never zero-filled):")
    for key in AGGREGATE_METRIC_KEYS:
        block = summary["macro"][key]
        print(
            f"  {key:12} {_fmt(block['value']):>10}  "
            f"(categories={block['categories_included']}"
            + (
                f", undefined={','.join(block['categories_undefined'])}"
                if block["categories_undefined"]
                else ""
            )
            + ")"
        )
    print("Micro averages (pooled counts — not a macro average):")
    for key in AGGREGATE_METRIC_KEYS:
        print(f"  {key:12} {_fmt(summary['micro'][key]):>10}")

    print("\nThresholds were frozen and unchanged; no test label influenced any")
    print("threshold or prediction. Pixel metrics omitted (image-level rule).")
    print(f"Measured test rates do not prove the Phase 4.7B guarantee (see report.md).")

    if failures:
        print(f"\n{len(failures)} categor(y/ies) failed; see {run_dir / 'failures.json'}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
