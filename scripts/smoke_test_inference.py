"""Smoke-test REAL AI inference (DINOv2 + feature bank + kNN).

Requires genuine data — refuses to run otherwise (no synthetic fallback,
no mock). Needs:

* ``datasets/mvtec_ad_processed/<category>/test/...`` (one real test image), or
  an explicit ``--image`` pointing at a real image, AND
* ``<bank-dir>/<category>/feature_bank.pt`` built by
  ``python -m model.inference.build_feature_bank``.

Checks: category, finite anomaly_score, 361 patch scores, 256x256 anomaly
map, heatmap/overlay shapes. Prints device, timing, and score. Never adds a
threshold or Normal/Anomaly label.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="MVTec category, e.g. bottle.")
    parser.add_argument("--image", type=Path, help="Real test image (default: first test image).")
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/mvtec_ad_processed"))
    parser.add_argument("--bank-dir", type=Path, default=Path("artifacts/dinov2_knn/banks"))
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def resolve_test_image(category: str, processed_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Test image not found: {explicit}.")
        return explicit
    test_dir = processed_dir / category / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Processed dataset not found: {test_dir}. "
            "Place the genuine MVTec AD processed dataset there first."
        )
    from model.inference.config import SUPPORTED_EXTENSIONS

    candidates = sorted(
        path
        for path in test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No test images under {test_dir}.")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    from model.inference.config import validate_category
    from model.inference.detector import AnomalyDetector

    validate_category(args.category)
    image = resolve_test_image(args.category, args.processed_dir, args.image)
    bank_file = args.bank_dir / args.category / "feature_bank.pt"
    if not bank_file.is_file():
        raise FileNotFoundError(
            f"Feature bank not found: {bank_file}. "
            "Build it with: python -m model.inference.build_feature_bank "
            f"--category {args.category}"
        )

    detector = AnomalyDetector(category=args.category, bank_path=bank_file, device=args.device)
    started = time.perf_counter()
    result = detector.predict(str(image))
    elapsed = time.perf_counter() - started

    score = result["anomaly_score"]
    assert isinstance(score, float) and math.isfinite(score), "anomaly_score must be finite"
    assert result["patch_scores"].shape == (361,), result["patch_scores"].shape
    assert result["anomaly_map"].shape == (256, 256), result["anomaly_map"].shape
    assert result["heatmap"].shape == (256, 256, 3), result["heatmap"].shape
    assert result["overlay"].shape == (256, 256, 3), result["overlay"].shape
    assert "is_anomaly" not in result

    print(f"image:          {image}")
    print(f"category:       {result['category']}")
    print(f"device:         {detector.extractor.device}")
    print(f"anomaly_score:  {score:.6f}")
    print(f"patch_scores:   {tuple(result['patch_scores'].shape)}")
    print(f"anomaly_map:    {tuple(result['anomaly_map'].shape)}")
    print(f"bank_shape:     {result['bank_shape']}")
    print(f"wall_time_s:    {elapsed:.2f}")
    print("SMOKE TEST PASSED (real inference, no threshold applied).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
