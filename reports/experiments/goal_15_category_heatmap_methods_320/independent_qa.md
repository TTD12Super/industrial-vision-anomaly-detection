# Independent QA notes: 15-category heatmap methods

## Scope

QA reviewed saved reports for raw PatchCore, DINO affinity, `heatmap_open5`, and Exact kNN. It did not rerun models or edit prediction artifacts.

## Split identity and exact-hash audit

All four reports use classification manifest SHA-256:

```text
fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432
```

The split has 2,540 train, 1,089 validation and 1,725 official-test images across 15 categories. The exact SHA-256 audit reports zero cross-split exact duplicates. No perceptual near-duplicate or acquisition-session audit was done. The official test has been inspected in earlier project experiments, so all results remain exploratory rather than unbiased final estimates.

## Threshold calibration

- Raw PatchCore, open5 and Exact kNN have method-specific threshold artifacts with known-normal validation calibration at q=.99 and no test-label selection.
- DINO affinity's `final_summary.json` references the shared raw PatchCore threshold. A separate q=.99 `affinity_thresholds/pixel_thresholds.json` exists, but the run summary does not reference it and the file has no manifest digest. Its use for the reported predictions is unverified. Treat thresholded precision, recall, F1/Dice and IoU as provisional; use threshold-free AP/AUROC for method ranking.

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
