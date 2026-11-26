"""Build normal feature banks from ``train/good`` images (CLI).

Thin wrapper around Phase 1 logic (no algorithm change). Expects the
preprocessed dataset layout from ``notebooks/01_data_exploration.ipynb``::

    <processed-dir>/<category>/train/good/*.png

Usage::

    python -m model.inference.build_feature_bank --category bottle
    python -m model.inference.build_feature_bank --all

Banks are written to ``<bank-dir>/<category>/feature_bank.pt`` (git-ignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CATEGORIES, validate_category
from .feature_bank import build_feature_bank, list_train_images
from .feature_extractor import DINOv2FeatureExtractor

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("datasets") / "mvtec_ad_processed"
DEFAULT_BANK_DIR = Path("artifacts") / "dinov2_knn" / "banks"


def build_category(
    category: str,
    processed_dir: Path,
    bank_dir: Path,
    batch_size: int,
    device: str | None,
    torch_hub_dir: str | None,
) -> Path:
    """Build and save the bank for one category; return the saved path."""
    validate_category(category)
    train_dir = processed_dir / category / "train" / "good"
    train_paths = list_train_images(train_dir)
    logger.info("%s: %d train/good images.", category, len(train_paths))
    extractor = DINOv2FeatureExtractor(device=device, torch_hub_dir=torch_hub_dir)
    destination = bank_dir / category / "feature_bank.pt"
    bank, _ = build_feature_bank(
        category,
        train_paths,
        extractor,
        batch_size=batch_size,
        save_path=destination,
        dataset_root=processed_dir,
    )
    logger.info("%s: saved bank %s to %s.", category, tuple(bank.shape), destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--category", help="Single MVTec category, e.g. bottle.")
    group.add_argument("--all", action="store_true", help="Build all 15 categories.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto).")
    parser.add_argument("--torch-hub-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    categories = list(CATEGORIES) if args.all else [args.category]
    for category in categories:
        build_category(
            category,
            args.processed_dir,
            args.bank_dir,
            args.batch_size,
            args.device,
            args.torch_hub_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
