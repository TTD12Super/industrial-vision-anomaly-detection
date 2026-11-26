"""LOO normal-score threshold calibration CLI (train/good only, no thresholds applied).

Derives category-specific GOOD/DEFECT thresholds from leave-one-image-out
scores of the normal ``train/good`` images against the existing, immutable
feature banks (Phase 4.7B/4.7C). It never reads test images, test labels,
ground-truth masks, or evaluation scores, never applies a threshold, never
classifies an image, never rebuilds a bank, and never contacts FastAPI,
MongoDB, or the API.

Examples::

    py scripts/calibrate_thresholds.py --categories bottle
    py scripts/calibrate_thresholds.py --all
    py scripts/calibrate_thresholds.py --all --alpha 0.05

Each invocation creates a brand-new directory
``artifacts/dinov2_knn/calibration_<UTC>/`` and never overwrites a previous
run (including the Phase 4.5/4.6 evaluation directories).
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
    group.add_argument("--all", action="store_true", help="Calibrate all 15 categories.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Pre-registered false-positive level (default 0.05).",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
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


def _peak_working_set_bytes() -> int | None:
    """Peak resident set size of this process (Windows diagnostics; best effort)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:  # pragma: no cover - diagnostics only
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    categories = resolve_categories(args)

    from model.evaluation.calibration import DEFAULT_ALPHA, run_calibration

    alpha = DEFAULT_ALPHA if args.alpha is None else args.alpha
    if not (0.0 < alpha < 1.0):
        raise SystemExit(f"error: --alpha must lie in (0, 1), got {alpha!r}.")

    run_dir, results = run_calibration(
        categories=categories,
        processed_root=args.processed_dir,
        bank_dir=args.bank_dir,
        run_root=args.run_root,
        alpha=alpha,
    )

    print(f"\nCalibration run: {run_dir}")
    print(
        f"{'category':11}{'normal':>8}{'alpha':>8}{'k':>6}{'threshold':>14}"
        f"{'feasible':>10}{'seconds':>9}"
    )
    for result in results:
        threshold = result.threshold
        cell = "inf" if not threshold.feasible else f"{threshold.threshold:.6f}"
        print(
            f"{result.category:11}{threshold.sample_count:>8}{threshold.alpha:>8g}"
            f"{threshold.index_k:>6}{cell:>14}"
            f"{str(threshold.feasible):>10}{result.seconds:>9.1f}"
        )
    peak = _peak_working_set_bytes()
    if peak is not None:
        print(f"\nPeak working set: {peak / (1024 ** 3):.2f} GiB")
    print("\nCalibration reads train/good normal images only.")
    print("No test data, test labels, or evaluation scores were accessed.")
    print("Thresholds are frozen artifacts; no threshold was applied to any image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
