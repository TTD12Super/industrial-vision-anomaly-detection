# Multi-detector automation implementation

This document records the two new detector architectures implemented on top of
the existing frozen DINOv2 + product-routing pipeline.

The historical `exact_knn` detector remains the default. The new detectors are
selected at runtime; no category names or dataset names are hard-coded in the
detector code.

## Implemented architectures

### PatchCore-style

1. Read selected frozen DINOv2 blocks, default `[5, 8, 11]` (zero-based).
2. Concatenate and L2-normalize aligned patch features.
3. Apply configurable local neighborhood aggregation, default `3×3` average pooling.
4. Select a deterministic representative normal coreset from `train/good`.
5. Score test descriptors with exact kNN against the coreset.
6. Convert patch scores to an image score and continuous anomaly map.

Artifacts are stored under:

```text
artifacts/production/<dataset>/detectors/patchcore/<category>/coreset.pt
```

### PaDiM-style

1. Read the same selected frozen DINOv2 blocks (`[5, 8, 11]` by default).
2. Fuse aligned spatial features.
3. Reduce the fused dimension with a deterministic seeded projection.
4. Fit one mean and regularized inverse covariance per spatial position using
   normal `train/good` descriptors only.
5. Score each query position with Mahalanobis distance.
6. Convert spatial distances to an image score and continuous anomaly map.

Artifacts are stored under:

```text
artifacts/production/<dataset>/detectors/padim/<category>/statistics.pt
```

## Shared automation flow

```text
dataset discovery
  → raw validation
  → RGB/mask preprocessing
  → deterministic train/validation split
  → frozen DINOv2 global embeddings
  → product classifier when category_count > 1
  → confidence routing (0.50 default)
  → selected detector artifact per category
  → normal-only image/pixel threshold calibration
  → official test inference
  → anomaly maps and heatmaps
  → CSV/JSON metrics and final_report.md
```

The classifier is skipped for a one-category dataset. Rejected low-confidence
inputs remain `unknown_product_category` and do not load a detector artifact.
Official test labels are used only for final evaluation, never for fitting,
coreset selection, covariance estimation, or threshold calibration.

## One-command usage

Baseline:

```bash
PYTHONPATH=src DAenv/bin/python scripts/run_production.py \
  --detector exact_knn
```

PatchCore-style:

```bash
PYTHONPATH=src DAenv/bin/python scripts/run_production.py \
  --detector patchcore \
  --layers 6,9,12 \
  --neighborhood-size 3 \
  --coreset-ratio 0.10
```

PaDiM-style:

```bash
PYTHONPATH=src DAenv/bin/python scripts/run_production.py \
  --detector padim \
  --layers 5,8,11 \
  --padim-reduced-dim 100 \
  --covariance-regularization 0.01
```

Comparison run:

```bash
PYTHONPATH=src DAenv/bin/python scripts/compare_detectors.py --device cpu
```

This keeps Exact kNN, PatchCore, and PaDiM independent, calibrates each
detector on the same known-normal validation split, evaluates all on the same
official test split, and records per-image mean/median/p95 latency. The
combined timing is the sequential cost of running all three; no official
fusion is enabled until it improves validation quality or reliability.

All reports are written automatically to:

```text
reports/production/<dataset-name>/<run-timestamp>/final_report.md
```

The selected detector and all detector settings are recorded in
`final_summary.json`.

## Source map

| Responsibility | File |
|---|---|
| Multi-layer DINOv2 output | `src/mvtec_system/core/dinov2.py` |
| Fusion, local aggregation, coreset, projection, statistics | `src/mvtec_system/anomaly/multi_layer.py` |
| Detector interface and registry | `src/mvtec_system/anomaly/detectors.py` |
| Dataset-driven orchestration | `src/mvtec_system/production/pipeline.py` |
| One-command CLI | `scripts/run_production.py` |

## Reproducibility and practical limits

- DINOv2 remains frozen; no detector fine-tunes the backbone.
- Exact kNN is preserved unchanged as the baseline path.
- PatchCore uses a seeded representative-memory selection. For very large
  banks it uses a bounded uniform fallback to avoid unbounded RAM/latency.
- PaDiM uses regularized pseudo-inverse covariance matrices for stable small or
  correlated feature sets.
- The first run downloads the pinned DINOv2 weights; later runs reuse the
  ignored local cache when available.
- Detector comparison is performed by running the same dataset with each
  detector and comparing the generated reports; the code does not assume a
  winner in advance.
