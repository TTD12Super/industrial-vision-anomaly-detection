# MVTec AD continued-system report

## Scope

This report documents the completed refactor defined in
`AI config/mvtec_continue.md`. The work reuses the existing local MVTec AD
dataset, processed images, cached DINOv2 weights, historical feature banks,
and notebook score artifacts. Raw data is read-only.

## Data and preprocessing

- Raw root: `datasets/mvtec_ad/`
- Processed root: `datasets/mvtec_ad_processed/`
- Categories: 15
- Raw audit: 3,629 train images, 1,725 test images, 1,258 masks
- Processed contract: RGB images at `256×256`; masks stay evaluation-only
- Classification manifest: 70/30 per-category split, seed 42, 2,540 train and
  1,089 validation images

## Inference architecture

```text
image
  └─ shared frozen DINOv2 (one forward)
       ├─ global embedding → logistic classifier → open-set router
       └─ patch embeddings → predicted category bank → exact Euclidean kNN
                                      └─ max patch score → anomaly map
```

The router rejects when validation-calibrated confidence or nearest-global
distance fails. A rejected input returns `unknown_product_category` and does
not load a category bank. Compatible banks preserve the existing notebook
schema and are reused without copying large files.

## Threshold protocol

Thresholds are selected from held-out known-normal validation images only. The
selected protocol is the 99th percentile with linear interpolation, stored
with split, sample count, target/observed acceptance, and decision rule
metadata. Image score is `max(patch_scores)`. A score greater than or equal to
the category threshold is `DEFECT`; otherwise it is `GOOD`.

No official MVTec test label is used to select a threshold. Therefore the
calibration protocol controls normal false rejects but does not claim defect
recall, AUROC, AP, or F1 during calibration.

## Evaluation

The evaluator keeps continuous ranking metrics separate from thresholded
metrics. It reports image AUROC/AP, pixel AUROC/AP, image precision/recall/F1,
specificity and confusion matrix, and pixel precision/recall/F1/Dice/IoU when
fixed binary predictions are supplied. Oracle-versus-automatic routing returns
an `automatic_minus_oracle` delta without tuning.

The saved-score reproduction was run on all 15 categories. Refactored output
matched the historical macro reference with maximum absolute delta `0.0`:

```text
Image AUROC = 0.9803247058806648
Image AP    = 0.987077244579432
Pixel AUROC = 0.9612823597274186
Pixel AP    = 0.5162392690459406
```

Detailed category values are in
`reports/baseline_reproduction/baseline_reproduction.json`.

## CLI and performance

Run automatic inference with:

```bash
PYTHONPATH=src DAenv/bin/python predict.py \
  --image datasets/mvtec_ad_processed/bottle/test/good/000.png \
  --device cpu
```

On the local machine, one `bottle/good` image measured approximately:

| Device | DINOv2 | kNN | Total | Bank | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 0.072 s | 0.065 s | 0.243 s | 221.05 MiB | ok |
| MPS | 0.158 s | 0.063 s | 0.161 s | 221.05 MiB | ok |
| CUDA | — | — | — | — | unavailable |

These are single-image wall-clock measurements, not optimization claims. The
full JSON is `reports/performance/performance.json`.

## Tests and versioning

The full suite has 67 passing tests covering dataset validation, manifest
reproducibility, shared DINOv2 contracts, classifier, router, banks, exact
kNN, maps, thresholds, metrics, pipeline, CLI helpers, reproduction, artifact
layout, and performance helpers.

New outputs use the ignored versioned layout
`artifacts/mvtec_system/v1/`. Historical `artifacts/dinov2_knn/` and
`artifacts/product_classifier/` artifacts are preserved and referenced rather
than copied.

## Limitations and next steps

1. Actual per-category anomaly threshold artifacts still need to be generated
   from held-out normal score runs before production GOOD/DEFECT decisions.
2. A known defect may be rejected as `unknown_product_category` when the
   global feature shifts beyond the normal routing thresholds; this is a
   policy trade-off, not a hidden fallback.
3. Full automatic-routing versus oracle-routing test numbers require generating
   global embeddings for the official test split; the comparison code exists,
   but no numbers are fabricated.
4. The external-image experiment was intentionally not included in the final
   evaluation; MVTec category distribution and camera setup remain different
   from arbitrary internet images.
