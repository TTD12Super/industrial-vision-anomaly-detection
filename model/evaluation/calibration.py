"""Leave-one-image-out (LOO) normal-score calibration for GOOD/DEFECT thresholds.

Phase 4.7B/4.7C specification. This module only *derives thresholds from
normal calibration data*; it never applies a threshold, never classifies an
image, and never reads test images, test labels, ground-truth masks, or
evaluation scores.

Data used: ``<processed>/<category>/train/good`` images and the rows of the
existing, immutable feature bank. The bank tensor is opened read-only and is
never written, rebuilt, moved, or modified.

Why LOO: the image score of a training image against a bank that contains its
own 361 patch rows collapses to the ``torch.cdist`` numerical floor (~1e-3),
so non-excluded training scores are unusable for calibration. Every query
image therefore has **all** of its own ``patch_count`` reference rows masked
out of the k=1 search before nearest neighbours are selected. Masking lives
entirely in this module; ``model.inference.knn.nearest_neighbor_scores`` and
the detector/API keep their original behaviour.

Threshold (pre-registered alpha, default 0.05)::

    k = ceil((m + 1) * (1 - alpha))
    threshold = k-th smallest LOO score  (ascending, 1-based index k)

If ``k > m`` the threshold is ``+infinity`` and the run records that this
alpha allows no rejections at the available sample size; alpha is never
silently relaxed.

Statistical claim (see ``assumptions_and_proof.md`` written into every run):
the resulting rule ``score > threshold`` carries a *marginal*, finite-sample
upper bound on the in-distribution false-positive rate ``<= alpha`` under the
documented assumptions, obtained from the Phase 4.7B symmetric-construction
and monotonicity-domination argument. This is **not** a split-conformal
guarantee and must never be described as one; the LOO scores are dependent and
are computed against an ``(m-1)``-image reference bank, so the ordinary
split-conformal theorem does not apply to them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..inference.config import (
    BANK_CHUNK_SIZE,
    CATEGORIES,
    FEATURE_KEY,
    IMAGE_AGGREGATION,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    K_NEIGHBORS,
    L2_NORMALIZE_FEATURES,
    MODEL_NAME,
    MODEL_REVISION,
    PROCESSED_SIZE,
    QUERY_CHUNK_SIZE,
    default_bank_path,
    validate_category,
)
from ..inference.feature_bank import load_feature_bank
from .evaluate import configuration_hash
from .metrics import image_files

logger = logging.getLogger(__name__)

#: Pre-registered default; must not be tuned after looking at test results.
DEFAULT_ALPHA = 0.05

#: Baseline patch geometry (DINOv2 ViT-B/14 on 256x256 -> 19x19 tokens).
EXPECTED_PATCH_GRID = (19, 19)
EXPECTED_PATCH_COUNT = EXPECTED_PATCH_GRID[0] * EXPECTED_PATCH_GRID[1]

THRESHOLD_FORMULA = "k = ceil((m + 1) * (1 - alpha)); threshold = k-th smallest LOO score (1-based index k)"
SORTING_CONVENTION = "scores sorted ascending; ties broken by stable original order; selection by rank only"
DECISION_RULE = "score > threshold (strict greater-than); documented only, NOT implemented in this phase"
BOUND_TYPE = (
    "marginal finite-sample upper bound on the in-distribution false-positive "
    "rate, P(score > threshold | normal, in-distribution) <= alpha"
)
NOT_SPLIT_CONFORMAL = (
    "Not split-conformal: LOO scores are mutually dependent and are computed "
    "against an (m-1)-image reference bank, so the ordinary split-conformal "
    "theorem does not apply to them."
)
DATA_STATEMENT = (
    "Calibration reads only train/good normal images and feature-bank rows; "
    "no test images, test labels, ground-truth masks, or evaluation scores "
    "are accessed."
)

#: All assumptions required by the Phase 4.7B proof (single source of truth).
BOUND_ASSUMPTIONS = [
    "(a) Exchangeability: the category's train/good images and future normal "
    "inputs are exchangeable (i.i.d. from the same distribution); no "
    "distribution shift is assumed away by this bound.",
    "(b) Fixed scoring map: the scoring function is deterministic and "
    "independent of the calibration sample (frozen DINOv2 weights, fixed "
    "preprocessing; the model is not fitted to this data).",
    "(c) Monotonicity in the reference bank: adding reference rows can never "
    "increase a k=1 patch distance, and the image score is the maximum over "
    "patches of those distances.",
    "(d) Nestedness at deployment: production scores with the full bank of "
    "the same m calibration images; calibration image i is scored against "
    "that bank minus its own patch_count rows.",
    "(e) Sample-size feasibility: k = ceil((m+1)*(1-alpha)) <= m, i.e. "
    "alpha >= 1/(m+1); otherwise the threshold is +infinity (no rejections).",
    "(f) Ties handled conservatively via the strict comparison score > threshold.",
    "(g) The guarantee is marginal over draws of the (m+1) exchangeable "
    "images; it is NOT conditional on this realized bank.",
]

PROOF_SKETCH = r"""
## Proof (Phase 4.7B symmetric construction + monotonicity domination)

Let the good images of one category be an exchangeable sample
`x_1 .. x_m`, let `R_i` be image i's `patch_count` bank rows, let
`B = R_1 u ... u R_m` be the full production bank, and let

* `S_i = f(x_i ; B \ R_i)` - the **computed** LOO score (all of image i's own
  rows excluded), i = 1..m,
* `t` = k-th smallest of `S_1 .. S_m`, `k = ceil((m+1)*(1-alpha))`,
* `S_prod = f(x_* ; B)` - the production score of a future good image.

The LOO scores `S_i` are mutually dependent (they share reference features)
and are scored against `(m-1)`-image references while `S_prod` uses an
`m`-image reference, so **(S_1..S_m, S_prod) is not exchangeable and the
ordinary split-conformal theorem does not apply to them.**

Extend the exchangeable sample with `x_{m+1} = x_*` and consider the
*symmetric* leave-one-out scores (a proof device, never computed at runtime)

    H_i = f(x_i ; {x_j : j <= m+1, j != i}),   i = 1 .. m+1.

1. `(H_1 .. H_{m+1})` is exchangeable, because each `H_i` is the same
   permutation-symmetric construction applied to an exchangeable sample.
2. The rank lemma for exchangeable scores gives
   `P(H_{m+1} > H_t) <= (m+1-k)/(m+1) <= alpha`, where `H_t` is the k-th
   smallest of `H_1 .. H_m` (ties handled with `>` conservatively).
3. Monotonicity (assumption c): the reference of `H_i` is
   `(B \ R_i) u R_{m+1} ⊇ (B \ R_i)`, the reference of `S_i`, hence
   `H_i <= S_i` pointwise for every i <= m, therefore `H_t <= t`.
4. Identity at deployment: `H_{m+1} = f(x_* ; B) = S_prod`.

Consequently `{S_prod > t} ⊆ {H_{m+1} > H_t}` and

    P(S_prod > t) <= P(H_{m+1} > H_t) <= alpha,

a finite-sample **marginal** bound for the *deployed* scoring procedure
(production uses the full bank B), with one-sided slack of exactly one
image's rows. It is not a conditional guarantee and says nothing about
defective images or about data drawn under distribution shift.
""".strip()


@dataclass(frozen=True)
class ThresholdResult:
    """Order-statistic threshold for one category."""

    threshold: float
    index_k: int
    alpha: float
    sample_count: int
    feasible: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "index_k": int(self.index_k),
            "alpha": float(self.alpha),
            "sample_count": int(self.sample_count),
            "feasible": bool(self.feasible),
            "note": self.note,
            "formula": THRESHOLD_FORMULA,
            "sorting_convention": SORTING_CONVENTION,
            "decision_rule": DECISION_RULE,
        }


@dataclass
class CategoryCalibration:
    """Calibration output for a single category."""

    category: str
    train_paths: list[str]
    scores: np.ndarray
    threshold: ThresholdResult
    patch_grid: list[int]
    padding: list[int]
    embedding_dimension: int
    bank_path: str
    seconds: float = 0.0


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bank_hashes(
    bank_dir: str | Path, categories: Sequence[str] = CATEGORIES
) -> dict[str, str | None]:
    """SHA-256 for every category's bank; ``None`` where no bank file exists."""
    root = Path(bank_dir)
    hashes: dict[str, str | None] = {}
    for category in categories:
        path = default_bank_path(category, root)
        hashes[category] = sha256_file(path) if path.is_file() else None
    return hashes


def train_relative_paths(processed_root: str | Path, category: str) -> list[str]:
    """Ordered ``train/good`` paths relative to *processed_root* (evaluator rule)."""
    root = Path(processed_root)
    train_folder = root / category / "train" / "good"
    return [str(path.relative_to(root)) for path in image_files(train_folder)]


def assert_paths_under_train_good(
    processed_root: str | Path, category: str, relative_paths: Sequence[str]
) -> None:
    """Reject any path that is not inside ``<processed>/<category>/train/good``."""
    root = Path(processed_root)
    train_good = (root / category / "train" / "good").resolve()
    forbidden = {"test", "ground_truth"}
    for rel in relative_paths:
        parts = set(Path(rel).parts)
        if parts & forbidden:
            raise ValueError(
                f"{category}: calibration refuses non-train path {rel!r} "
                "(test/ground_truth components are never calibration input)."
            )
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(train_good):
            raise ValueError(
                f"{category}: path {rel!r} is outside {train_good}."
            )


def validate_bank_for_calibration(
    category: str,
    bank: Any,
    metadata: dict[str, Any],
    processed_root: str | Path,
) -> int:
    """Validate provenance, geometry, and ordered listing; return m."""
    expected_meta = {
        "category": category,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "feature_key": FEATURE_KEY,
        "image_size": list(PROCESSED_SIZE),
        "l2_normalization": L2_NORMALIZE_FEATURES,
        "sampling": "none; all train/good patches",
    }
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Feature bank for '{category}' has {key}={metadata.get(key)!r}, "
                f"expected {value!r}; refusing to calibrate with incompatible settings."
            )
    if list(metadata.get("mean", [])) != list(IMAGENET_MEAN) or list(
        metadata.get("std", [])
    ) != list(IMAGENET_STD):
        raise ValueError(f"{category}: bank normalization statistics are not ImageNet mean/std.")
    grid = tuple(int(v) for v in metadata.get("patch_grid", []))
    if grid != EXPECTED_PATCH_GRID:
        raise ValueError(
            f"{category}: bank patch_grid={grid}, expected {EXPECTED_PATCH_GRID}."
        )
    patch_count = grid[0] * grid[1]
    if patch_count != EXPECTED_PATCH_COUNT:
        raise ValueError(f"{category}: patch_count={patch_count}, expected 361.")
    if int(metadata.get("embedding_dimension", -1)) != int(bank.shape[1]):
        raise ValueError(f"{category}: declared embedding dimension mismatch.")
    if getattr(bank, "ndim", 0) != 2:
        raise ValueError(f"{category}: bank must be a 2D tensor.")

    stored = [str(path) for path in metadata.get("train_paths", [])]
    expected = train_relative_paths(processed_root, category)
    if not expected:
        raise ValueError(f"{category}: no train/good images found for calibration.")
    if stored != expected:
        raise ValueError(
            f"Feature bank for '{category}' train_paths does not match the ordered "
            "train/good listing element by element; refusing to calibrate."
        )
    assert_paths_under_train_good(processed_root, category, stored)

    m = len(stored)
    if m < 2:
        raise ValueError(f"{category}: LOO calibration needs at least 2 training images.")
    expected_rows = m * patch_count
    if int(bank.shape[0]) != expected_rows:
        raise ValueError(
            f"{category}: bank has {bank.shape[0]} rows, expected "
            f"{expected_rows} (= {m} images x {patch_count} patches)."
        )
    return m


def loo_patch_scores(
    bank: Any,
    image_indices: Sequence[int],
    patch_count: int = EXPECTED_PATCH_COUNT,
    *,
    k: int = K_NEIGHBORS,
    bank_chunk_size: int = BANK_CHUNK_SIZE,
) -> Any:
    """Patch-level LOO distances for the given bank images (torch, CPU).

    For every query row the **entire** ``patch_count``-row block of its own
    image is masked out of the k=1 search, so excluded rows can never be
    selected as nearest neighbours. The bank tensor is only read.
    """
    import torch as _torch

    num_images = len(image_indices)
    if num_images < 1:
        raise ValueError("At least one query image is required.")
    if patch_count < 1:
        raise ValueError(f"patch_count must be positive, got {patch_count}.")
    if k < 1 or k > len(bank):
        raise ValueError("k must be between 1 and the bank size.")
    if bank_chunk_size < 1:
        raise ValueError(f"bank_chunk_size must be positive, got {bank_chunk_size}.")
    total_rows = len(bank)
    if any(i < 0 or (i + 1) * patch_count > total_rows for i in image_indices):
        raise ValueError("Image index outside the bank's row range.")

    with _torch.inference_mode():
        scores: list[Any] = []
        for image_index in image_indices:
            start = image_index * patch_count
            stop = start + patch_count
            queries = bank[start:stop]  # view of the bank; read-only
            best = _torch.full((patch_count, k), float("inf"), device=bank.device)
            for bank_start in range(0, total_rows, bank_chunk_size):
                bank_stop = min(bank_start + bank_chunk_size, total_rows)
                keep = _torch.ones(bank_stop - bank_start, dtype=_torch.bool)
                low = max(bank_start, start) - bank_start
                high = min(bank_stop, stop) - bank_start
                if low < high:
                    keep[low:high] = False
                if not bool(keep.any()):
                    continue
                chunk = bank[bank_start:bank_stop][keep]  # copy of allowed rows only
                distances = _torch.cdist(queries, chunk, p=2)
                if k == 1:
                    patch_best = distances.min(dim=1).values.unsqueeze(1)
                    best = _torch.minimum(best, patch_best)
                else:
                    take = min(k, distances.shape[1])
                    candidates = _torch.cat(
                        (best, distances.topk(take, dim=1, largest=False).values), dim=1
                    )
                    best = candidates.topk(k, dim=1, largest=False).values
            if not _torch.isfinite(best).all().item():
                raise ValueError(
                    "kNN with exclusion returned non-finite distances "
                    "(bank smaller than the excluded block?)."
                )
            scores.append(best.mean(dim=1).detach().cpu())
        return _torch.stack(scores)  # (num_images, patch_count)


def loo_image_scores(
    bank: Any,
    image_indices: Sequence[int],
    patch_count: int = EXPECTED_PATCH_COUNT,
    *,
    k: int = K_NEIGHBORS,
    bank_chunk_size: int = BANK_CHUNK_SIZE,
) -> np.ndarray:
    """Image-level LOO scores: max over patches, matching the baseline rule."""
    patch_scores = loo_patch_scores(
        bank,
        image_indices,
        patch_count,
        k=k,
        bank_chunk_size=bank_chunk_size,
    )
    return patch_scores.max(dim=1).values.numpy().astype(np.float64)


def order_statistic_threshold(scores: Sequence[float], alpha: float) -> ThresholdResult:
    """k-th smallest LOO score with ``k = ceil((m + 1) * (1 - alpha))``."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}.")
    values = [float(v) for v in scores]
    m = len(values)
    if m < 1:
        raise ValueError("At least one calibration score is required.")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Calibration scores must be finite.")
    k = math.ceil((m + 1) * (1.0 - alpha))
    if k > m:
        note = (
            f"alpha={alpha} requires k={k} > m={m}; this alpha allows no "
            "rejections at the available sample size; alpha was not relaxed."
        )
        return ThresholdResult(
            threshold=math.inf,
            index_k=k,
            alpha=float(alpha),
            sample_count=m,
            feasible=False,
            note=note,
        )
    if k < 1:  # pragma: no cover - unreachable for 0 < alpha < 1
        raise ValueError(f"Invalid order-statistic index k={k}.")
    threshold = sorted(values)[k - 1]
    return ThresholdResult(
        threshold=threshold,
        index_k=k,
        alpha=float(alpha),
        sample_count=m,
        feasible=True,
        note=f"k={k} <= m={m}; threshold is the {k}-th smallest of {m} LOO scores.",
    )


def calibrate_category(
    category: str,
    *,
    processed_root: str | Path,
    bank_dir: str | Path,
    alpha: float = DEFAULT_ALPHA,
    bank_chunk_size: int = BANK_CHUNK_SIZE,
) -> CategoryCalibration:
    """LOO calibration for one category (read-only bank access)."""
    validate_category(category)
    started = time.perf_counter()
    bank_path = default_bank_path(category, bank_dir)
    if not bank_path.is_file():
        raise FileNotFoundError(
            f"Feature bank for category '{category}' is not available: {bank_path}."
        )
    bank, metadata = load_feature_bank(bank_path)
    m = validate_bank_for_calibration(category, bank, metadata, processed_root)
    patch_count = EXPECTED_PATCH_COUNT
    scores = loo_image_scores(
        bank,
        list(range(m)),
        patch_count,
        bank_chunk_size=bank_chunk_size,
    )
    threshold = order_statistic_threshold(scores, alpha)
    result = CategoryCalibration(
        category=category,
        train_paths=[str(path) for path in metadata["train_paths"]],
        scores=scores,
        threshold=threshold,
        patch_grid=[int(v) for v in metadata["patch_grid"]],
        padding=[int(v) for v in metadata["padding"]],
        embedding_dimension=int(metadata["embedding_dimension"]),
        bank_path=str(bank_path),
        seconds=time.perf_counter() - started,
    )
    logger.info(
        "%s: %d normal calibration images; alpha=%s k=%d threshold=%s (%.1fs).",
        category,
        m,
        f"{alpha:g}",
        threshold.index_k,
        "inf" if not threshold.feasible else f"{threshold.threshold:.6f}",
        result.seconds,
    )
    return result


def write_calibration_scores(run_dir: Path, result: CategoryCalibration) -> None:
    """Write ``calibration_scores/<category>.csv`` (train paths only)."""
    folder = run_dir / "calibration_scores"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{result.category}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "image_path",
                "row_start",
                "row_end",
                "patch_count",
                "loo_image_score",
            ],
        )
        writer.writeheader()
        patch_count = EXPECTED_PATCH_COUNT
        for index, (path, score) in enumerate(zip(result.train_paths, result.scores)):
            writer.writerow(
                {
                    "category": result.category,
                    "image_path": path,
                    "row_start": index * patch_count,
                    "row_end": (index + 1) * patch_count,
                    "patch_count": patch_count,
                    "loo_image_score": repr(float(score)),
                }
            )


def write_thresholds_csv(run_dir: Path, results: Sequence[CategoryCalibration]) -> None:
    """Write ``thresholds.csv`` for all calibrated categories."""
    fields = [
        "category",
        "train_images",
        "alpha",
        "index_k",
        "threshold",
        "alpha_feasible",
        "formula",
        "sorting_convention",
        "decision_rule",
        "seconds",
    ]
    with (run_dir / "thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            threshold = result.threshold
            writer.writerow(
                {
                    "category": result.category,
                    "train_images": threshold.sample_count,
                    "alpha": repr(threshold.alpha),
                    "index_k": threshold.index_k,
                    "threshold": repr(float(threshold.threshold)),
                    "alpha_feasible": str(threshold.feasible).lower(),
                    "formula": THRESHOLD_FORMULA,
                    "sorting_convention": SORTING_CONVENTION,
                    "decision_rule": DECISION_RULE,
                    "seconds": round(result.seconds, 3),
                }
            )


def write_assumptions_doc(run_dir: Path, alpha: float) -> None:
    """Write ``assumptions_and_proof.md`` (bound nature + assumptions + proof)."""
    lines = [
        "# LOO calibration: assumptions, bound, and proof",
        "",
        f"* Pre-registered alpha: **{alpha}** (never tuned against test data).",
        f"* Threshold formula: `{THRESHOLD_FORMULA}`.",
        f"* Sorting: {SORTING_CONVENTION}.",
        f"* Decision rule (documented only, not implemented in this phase): `{DECISION_RULE}`.",
        f"* Bound claimed: {BOUND_TYPE}.",
        f"* {NOT_SPLIT_CONFORMAL}",
        f"* {DATA_STATEMENT}",
        "* Calibration data contain **normal images only**; they contain no "
        "defective images, so nothing about defect recall, accuracy, or "
        "precision can be estimated from them.",
        "",
        "## Assumptions required by the bound",
        "",
    ]
    lines.extend(f"{assumption}" for assumption in BOUND_ASSUMPTIONS)
    lines.extend(["", PROOF_SKETCH, ""])
    lines.extend(
        [
            "## Not claimed",
            "",
            "* No split-conformal guarantee and no conditional (bank-specific) guarantee.",
            "* No guarantee under distribution shift; no deployment-performance claim.",
            "* No claim about defective images (recall, accuracy, precision, F1).",
            "* No pixel-level or per-defect-type threshold.",
            "",
        ]
    )
    (run_dir / "assumptions_and_proof.md").write_text("\n".join(lines), encoding="utf-8")


def _source_revision() -> str | None:
    """Best-effort git revision of the working tree (never fails the run)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def build_calibration_config(
    *,
    run_dir: Path,
    processed_root: str | Path,
    bank_dir: str | Path,
    results: Sequence[CategoryCalibration],
    alpha: float,
    bank_hashes_before: dict[str, str | None],
    bank_hashes_after: dict[str, str | None],
) -> dict[str, Any]:
    """Reproducibility payload written to ``calibration_config.json``."""
    try:
        import platform

        import torch as _torch

        packages = {
            "python": platform.python_version(),
            "torch": getattr(_torch, "__version__", "unknown"),
            "numpy": np.__version__,
        }
    except Exception:  # pragma: no cover - diagnostics only
        packages = {"numpy": np.__version__}
    first = results[0] if results else None
    settings = {
        "alpha": float(alpha),
        "threshold_formula": THRESHOLD_FORMULA,
        "model_revision": MODEL_REVISION,
        "feature_key": FEATURE_KEY,
        "k_neighbors": K_NEIGHBORS,
        "image_aggregation": IMAGE_AGGREGATION,
        "categories": sorted(result.category for result in results),
    }
    hashes_unchanged = bank_hashes_before == bank_hashes_after
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_root": str(processed_root),
        "feature_bank_root": str(bank_dir),
        "calibration_categories": [
            {
                "category": result.category,
                "train_images": result.threshold.sample_count,
                "alpha": result.threshold.alpha,
                "index_k": result.threshold.index_k,
                "threshold": float(result.threshold.threshold),
                "alpha_feasible": result.threshold.feasible,
                "note": result.threshold.note,
                "patch_count": EXPECTED_PATCH_COUNT,
                "seconds": round(result.seconds, 3),
            }
            for result in results
        ],
        "alpha": float(alpha),
        "threshold_formula": THRESHOLD_FORMULA,
        "sorting_convention": SORTING_CONVENTION,
        "decision_rule": DECISION_RULE,
        "bound": {
            "claimed": BOUND_TYPE,
            "is_split_conformal": False,
            "is_conditional": False,
            "not_split_conformal_statement": NOT_SPLIT_CONFORMAL,
            "assumptions": list(BOUND_ASSUMPTIONS),
            "proof_document": "assumptions_and_proof.md",
        },
        "calibration_data_statement": DATA_STATEMENT,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "feature_key": FEATURE_KEY,
        "image_size": list(PROCESSED_SIZE),
        "input_image_size": int(IMAGE_SIZE),
        "patch_grid": list(first.patch_grid) if first else list(EXPECTED_PATCH_GRID),
        "padding": list(first.padding) if first else [10, 10],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "l2_normalization": L2_NORMALIZE_FEATURES,
        "k_neighbors": K_NEIGHBORS,
        "image_aggregation": IMAGE_AGGREGATION,
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "bank_chunk_size": BANK_CHUNK_SIZE,
        "bank_hashes_before": "bank_hashes_before.json",
        "bank_hashes_after": "bank_hashes_after.json",
        "bank_hashes_unchanged": bool(hashes_unchanged),
        "source_revision": _source_revision(),
        "packages": packages,
        "config_hash": configuration_hash(settings),
    }


def run_calibration(
    *,
    categories: Sequence[str],
    processed_root: str | Path,
    bank_dir: str | Path,
    run_root: str | Path,
    alpha: float = DEFAULT_ALPHA,
    bank_chunk_size: int = BANK_CHUNK_SIZE,
) -> tuple[Path, list[CategoryCalibration]]:
    """Calibrate *categories* into a brand-new ``calibration_<UTC>`` folder.

    Creates the run directory with ``exist_ok=False`` so an existing run
    (Phase 4.5/4.6 evaluation or a previous calibration) is never overwritten.
    Raises ``RuntimeError`` if any feature bank changed during the run.
    """
    processed_root = Path(processed_root)
    bank_dir = Path(bank_dir)
    run_root = Path(run_root)
    if not categories:
        raise ValueError("At least one category is required.")
    for category in categories:
        validate_category(category)
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = run_root / f"calibration_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Creating calibration run %s (alpha=%s).", run_dir, alpha)

    hashes_before = bank_hashes(bank_dir)
    (run_dir / "bank_hashes_before.json").write_text(
        json.dumps(hashes_before, indent=2), encoding="utf-8"
    )

    results: list[CategoryCalibration] = []
    for category in categories:
        result = calibrate_category(
            category,
            processed_root=processed_root,
            bank_dir=bank_dir,
            alpha=alpha,
            bank_chunk_size=bank_chunk_size,
        )
        write_calibration_scores(run_dir, result)
        results.append(result)

    write_thresholds_csv(run_dir, results)
    write_assumptions_doc(run_dir, alpha)

    hashes_after = bank_hashes(bank_dir)
    (run_dir / "bank_hashes_after.json").write_text(
        json.dumps(hashes_after, indent=2), encoding="utf-8"
    )
    if hashes_before != hashes_after:
        changed = [
            name
            for name in set(hashes_before) | set(hashes_after)
            if hashes_before.get(name) != hashes_after.get(name)
        ]
        raise RuntimeError(
            f"Feature banks changed during calibration: {sorted(changed)}. "
            "Calibration never writes banks; stop and investigate."
        )

    config = build_calibration_config(
        run_dir=run_dir,
        processed_root=processed_root,
        bank_dir=bank_dir,
        results=results,
        alpha=alpha,
        bank_hashes_before=hashes_before,
        bank_hashes_after=hashes_after,
    )
    (run_dir / "calibration_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    logger.info("Calibration run complete: %s", run_dir)
    return run_dir, results
