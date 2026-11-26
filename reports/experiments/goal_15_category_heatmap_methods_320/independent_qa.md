# Independent QA notes: 15-category heatmap methods

## Scope

QA reviewed saved reports for raw PatchCore, DINO affinity, `heatmap_open5`, and Exact kNN. It did not rerun models or edit prediction artifacts.

## Split identity and exact-hash audit

All four reports use the same train/validation manifest SHA-256:

```text
fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432
```

The manifest contains 2,540 train and 1,089 validation rows. The 1,725 official-test files are discovered from the test folders; their sorted inventory fingerprint, including decoded-pixel hashes, is `a88785e62f357832c19188baddb3df95ca252e1257dc6050c26f728c23ac32be`. Both the exact file SHA-256 audit and a decoded-pixel SHA-256 audit report zero cross-split identical images; the latter covers all 5,354 files and is recorded in [`decoded_pixel_split_audit.json`](decoded_pixel_split_audit.json). It catches identical pixel arrays even if PNG encodings differ. No reliable perceptual near-duplicate or acquisition-session audit has been completed. A coarse low-resolution difference hash produced many candidates among visually similar, distinct MVTec objects, so it was discarded as duplicate evidence. The official test has been inspected in earlier project experiments, so all results remain exploratory rather than unbiased final estimates.

## Threshold calibration — follow-up source audit

This section clarifies the initial static QA note using the current production code and saved run artifacts; the QA agent did not rerun model inference during this follow-up.

- Raw PatchCore, open5 and Exact kNN have method-specific threshold artifacts with known-normal validation calibration at q=.99 and no test-label selection.
- DINO affinity calibrates the configured refined maps on known-normal validation data through `_calibrate_detector_thresholds` → `_score_detector_paths` → `_make_refined_anomaly_map`; inference uses the resulting in-memory thresholds. The q=.99 sidecar's per-category pixel counts match the validation split at 320×320. The summary's detector-scoped artifact path was later overwritten by the raw PatchCore run, and the sidecar lacks a manifest digest, so artifact lineage is weaker. The reported threshold-dependent metrics can be compared as per-method q=.99 operating points; AP/AUROC remain threshold-free.
- The production writer now stores non-`none` refiner thresholds under `thresholds/<detector>/refiners/<refiner>`; the legacy raw threshold path is preserved. This prevents future refiner runs from overwriting the raw detector's thresholds.

## Metric check

| Method | Micro Pixel AP | Macro Pixel AP | Micro Pixel AUROC | Macro Pixel AUROC | Test maps |
|---|---:|---:|---:|---:|---:|
| Raw PatchCore | 55.376% | 61.279% | 96.265% | 97.585% | 1,725/1,725 |
| DINO affinity | 54.960% | 59.618% | 95.283% | 96.361% | 1,725/1,725 |
| open5 | 55.347% | 61.243% | 96.268% | 97.590% | 1,725/1,725 |
| Exact kNN | 43.751% | 53.464% | 94.236% | 96.239% | 1,722/1,725 |

DINO affinity falls behind raw PatchCore on threshold-free AP and AUROC. Open5 is nearly neutral but slightly lower on AP. Exact kNN is clearly lower and skips three images before localization.

The product classifier is unchanged. Saved reports show 100.00% Top-1 on 1,089 normal validation images and 99.826% on 1,725 official test images (three UNKNOWN/rejected count as incorrect). The validation split was also used to choose among three regularization values, and the official test split was inspected in prior work. Both numbers clear the numeric >90% target, but neither supports an untouched-holdout generalization claim.

## Independent software QA

A separate focused run passed 24 production/anomaly unit tests, including the fail-closed calibration preflight and artifact-cache validation checks. Static review found no error in the saved split identity or Exact kNN/raw/open5 arithmetic. The saved Exact kNN run lacks a source-tree hash captured at execution time; paired bootstrap intervals are absent because per-image maps were not retained.

## Conclusion

Keep raw PatchCore as the current heatmap baseline. Do not promote open5, DINO affinity or Exact kNN for heatmap localization on this evidence. The goal is not complete because no method improved Pixel AP and the test set has prior exposure.
