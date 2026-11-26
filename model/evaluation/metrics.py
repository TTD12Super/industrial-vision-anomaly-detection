"""Reusable MVTec AD evaluation metrics.

Reference implementation: ``notebooks/02_first_model.ipynb`` (cells 12, 62, 64,
66, 68). The semantics below are extracted verbatim from that notebook so the
offline evaluator stays comparable with the recorded baseline results:

* image label = ``test_path.parent.name != "good"`` (a test image is normal
  only when it lives in the ``good`` folder)
* ``good`` test images use an all-zero mask; a defective image uses exactly one
  ``<stem>_mask.png`` from the matching ``ground_truth/<defect_type>/`` folder
* masks must match the processed image size, contain only binary values
  ``{0, 1, 255}``, and must not be empty
* ranking metrics are exact sklearn AUROC/AP over the full population; a metric
  is reported as ``None`` when the labels contain a single class
* pixel metrics are computed over **all** test pixels, including normal-image
  all-zero masks (reference convention; this is what makes pixel AP low)
* PRO/AUPRO is **not implemented** (explicitly out of scope for this project)

No threshold is ever derived from test labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..inference.config import PROCESSED_SIZE, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

#: Reference notebook guard: exact pixel sorting must stay under this budget.
MAX_PIXEL_METRIC_GIB = 3.0

#: Values allowed in a ground-truth mask by the reference notebook.
ALLOWED_MASK_VALUES = (0, 1, 255)

#: Metric definitions recorded in every run configuration / metrics payload.
METRIC_DEFINITIONS = {
    "image_auroc": (
        "Exact sklearn ROC-AUROC over per-image anomaly scores (max patch "
        "distance); label 1 = defective (parent directory is not 'good')."
    ),
    "image_ap": (
        "Exact sklearn average precision over per-image anomaly scores "
        "(max patch distance)."
    ),
    "pixel_auroc": (
        "Exact sklearn ROC-AUROC over every pixel of every test anomaly map, "
        "against binarized ground-truth masks (normal images contribute "
        "all-zero masks)."
    ),
    "pixel_ap": (
        "Exact sklearn average precision over every pixel of every test "
        "anomaly map, against binarized ground-truth masks (normal images "
        "contribute all-zero masks; this is why pixel AP is low)."
    ),
    "aupro": "PRO/AUPRO is not implemented in this project.",
}

RANKING_METRIC_NAMES = ("auroc", "average_precision")


def image_files(folder: str | Path) -> list[Path]:
    """Sorted supported image files under *folder* (notebook cell 12)."""
    directory = Path(folder)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def image_labels(test_paths: Sequence[Path]) -> np.ndarray:
    """Boolean labels (``True`` = defective) from directory names only.

    Reference rule: an image is normal only when its parent directory is
    ``good``; every other test-type directory is defective. Ground truth is
    never consulted to derive the image label.
    """
    return np.array([Path(path).parent.name != "good" for path in test_paths])


def find_mask_path(category: str, test_path: str | Path, processed_root: str | Path) -> Path | None:
    """Resolve the ground-truth mask for one test image (notebook cell 62).

    ``good`` images have no mask (``None``). A defective image must map to
    exactly one ``<stem>_mask`` file under ``ground_truth/<defect_type>/``;
    anything else (zero or several matches) is an error rather than a guess.
    """
    test_path = Path(test_path)
    if test_path.parent.name == "good":
        return None
    mask_folder = Path(processed_root) / category / "ground_truth" / test_path.parent.name
    mask_stem = test_path.stem + "_mask"
    matches = [path for path in image_files(mask_folder) if path.stem == mask_stem]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one ground-truth mask for {test_path}.")
    return matches[0]


def load_mask(category: str, test_path: str | Path, processed_root: str | Path) -> np.ndarray:
    """Load and validate the ground-truth mask for one test image (cell 64).

    Returns a boolean ``(H, W)`` array. ``good`` images yield an all-zero
    mask; defective images are validated (size, binary values, non-empty) and
    binarized with ``> 0``.
    """
    from PIL import Image

    mask_path = find_mask_path(category, test_path, processed_root)
    if mask_path is None:
        return np.zeros(PROCESSED_SIZE, dtype=bool)
    with Image.open(mask_path) as image:
        image.load()
        mask = np.array(image, copy=True)
    if mask.shape != PROCESSED_SIZE:
        raise ValueError(
            f"Ground-truth shape does not match the processed image: {mask_path}"
        )
    if not np.isin(mask, list(ALLOWED_MASK_VALUES)).all():
        raise ValueError(f"Expected a binary ground-truth mask: {mask_path}")
    binary = mask > 0
    if not binary.any():
        raise ValueError(f"The anomalous image has an empty ground-truth mask: {mask_path}")
    return binary


def load_masks(category: str, test_paths: Sequence[Path], processed_root: str | Path) -> np.ndarray:
    """Stack per-image masks into a boolean ``(N, H, W)`` array."""
    return np.stack(
        [load_mask(category, path, processed_root) for path in test_paths]
    )


def ranking_metrics(labels: Iterable, scores: Iterable) -> dict[str, float | None]:
    """Exact AUROC and average precision (notebook cell 66).

    Raises ``ValueError`` on shape mismatch or non-finite scores. Returns
    ``{"auroc": None, "average_precision": None}`` when the labels contain a
    single class, because a ranking metric is undefined in that case.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    label_array = np.asarray(labels).reshape(-1)
    score_array = np.asarray(scores).reshape(-1)
    if label_array.shape != score_array.shape:
        raise ValueError("Metric labels and scores must have the same shape.")
    if not np.isfinite(score_array).all():
        raise ValueError("Metric scores contain invalid values.")
    if np.unique(label_array).size < 2:
        logger.info("AUROC/AP unavailable: both positive and negative labels are needed.")
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(label_array, score_array)),
        "average_precision": float(average_precision_score(label_array, score_array)),
    }


def pixel_metric_memory_gib(maps: np.ndarray) -> float:
    """Estimated exact-pixel-metric budget in GiB (reference formula)."""
    return float(maps.size) * 64.0 / (1024.0**3)


def pixel_metrics(masks: np.ndarray, maps: np.ndarray) -> dict[str, float | None]:
    """Exact pixel-level AUROC/AP with the reference memory guard (cell 68)."""
    if masks.shape != maps.shape:
        raise ValueError(
            f"Pixel metrics need matching shapes, got masks={masks.shape} maps={maps.shape}."
        )
    estimated_gib = pixel_metric_memory_gib(maps)
    if estimated_gib > MAX_PIXEL_METRIC_GIB:
        raise MemoryError(
            f"Exact pixel metrics would use {estimated_gib:.2f} GiB, above the "
            f"reference limit of {MAX_PIXEL_METRIC_GIB} GiB."
        )
    return ranking_metrics(np.asarray(masks).reshape(-1), np.asarray(maps).reshape(-1))


def evaluate_ranking(
    labels: Iterable,
    scores: Iterable,
    masks: np.ndarray | None,
    maps: np.ndarray | None,
) -> dict[str, float | None]:
    """Image-level and pixel-level metrics for one category (cell 68).

    Pixel metrics are skipped (``None``) when *masks*/*maps* are omitted, so
    callers can run a cheap image-only pass without changing any semantics.
    """
    image_result = ranking_metrics(labels, scores)
    if masks is None or maps is None:
        pixel_result = {"auroc": None, "average_precision": None}
    else:
        pixel_result = pixel_metrics(masks, maps)
    return {
        "image_auroc": image_result["auroc"],
        "image_ap": image_result["average_precision"],
        "pixel_auroc": pixel_result["auroc"],
        "pixel_ap": pixel_result["average_precision"],
    }


def macro_average(values: Sequence[float | None]) -> tuple[float | None, int]:
    """Mean of the valid (non-``None``) values and how many were included.

    ``None`` entries are excluded and counted separately; they are never
    treated as zero.
    """
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return None, 0
    return float(np.mean(valid)), len(valid)
