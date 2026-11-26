# 15-category heatmap methods (320px)

**Decision:** retain raw PatchCore. None of the tested screw-branch transfers improved the heatmap ranking metrics.

## Protocol

All four methods use the same 15-category train/validation/test manifest, SHA-256 `fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432`, with 2,540 train, 1,089 validation and 1,725 test images at 320×320. The effective split has zero exact-hash overlap. Raw PatchCore, open5 and Exact kNN thresholds use known-normal validation pixels at q=.99. The DINO-affinity report references the shared raw PatchCore threshold file rather than preserving an affinity-specific calibration artifact; its thresholded precision/recall/F1/IoU are therefore not a fair calibrated operating-point comparison. Pixel AP and AUROC are threshold-free.

The official MVTec test split had already informed prior method and resolution choices. These comparisons are **exploratory**, not an unbiased final estimate; the exact-hash audit does not address near-duplicates or acquisition sessions.

## Results

| Method | Pixel AP micro | Pixel AP macro | Pixel AUROC micro | Precision | Recall | Dice/F1 | Map coverage | Δ micro AP vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw PatchCore | 55.38% | 61.28% | 96.26% | 20.11% | 95.33% | 33.21% | 1725/1725 (100.00%) | +0.000 pp |
| DINO affinity | 54.96% | 59.62% | 95.28% | 12.93% | 97.47% | 22.82% | 1725/1725 (100.00%) | -0.416 pp |
| Open5 | 55.35% | 61.24% | 96.27% | 20.01% | 95.42% | 33.08% | 1725/1725 (100.00%) | -0.029 pp |
| Exact kNN | 43.75% | 53.46% | 94.24% | 20.05% | 89.83% | 32.79% | 1722/1725 (99.83%) | -11.625 pp |

![Overall Pixel AP comparison](pixel_ap_overall.png)

[Vector version](pixel_ap_overall.svg)

![Per-category Pixel AP](pixel_ap_by_category.png)

[Vector version](pixel_ap_by_category.svg)

- **DINO affinity / relation propagation:** recall rises, while Pixel AP, AUROC, precision and Dice decrease. The map becomes more diffuse.
- **open5 morphology:** near-neutral AP with small declines in precision, Dice and IoU; the recall gain is 0.089 percentage points.
- **Exact patch kNN:** Pixel AP is lower by over 11 percentage points; 3 UNKNOWN/rejected test images have no map. Do not replace PatchCore for localization.
- **Product classification:** unchanged. Top-1 is 100.00% on the 1,089-image normal validation split and 99.826% on the 1,725-image official test split (the 3 UNKNOWN/rejects count as incorrect). The validation score was also used to choose among three classifier regularization values, so it is a model-selection score; the test score is exploratory because the official test was previously inspected. Both exceed 90% numerically, but neither establishes performance on an untouched camera/product holdout.

## Screw-method transfer boundary

The tested non-supervised transfers were DINO affinity/relation propagation and morphological opening; both failed to beat raw PatchCore. Exact patch kNN was also weaker for localization. The screw supervised patch-mask refiner requires labeled defect masks. MVTec training data are GOOD-only; using its official test masks to train the refiner would leak evaluation labels. A separate labeled development set is needed before that refiner can be fairly trained and assessed.

## Evidence and QA

- Per-category Pixel AP/AUROC: [`per_category_comparison.csv`](per_category_comparison.csv).
- Machine-readable metrics, split identity and limitations: [`comparison_summary.json`](comparison_summary.json).
- [Raw PatchCore report](../goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/final_report.md).
- [DINO-affinity report](../goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/final_report.md).
- [open5 report](../goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/final_report.md).
- [Exact kNN report](../goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/final_report.md).
- Independent QA verified all four runs use the same split manifest and checked metric arithmetic. It also found that the affinity report reuses the raw PatchCore threshold artifact; affinity thresholded metrics should not be compared as calibrated operating points. 24 focused production/unit tests passed before the report was generated.

Limitations: Exact kNN run did not record a source-tree hash; per-image maps were not retained for paired bootstrap intervals; no perceptual/session duplicate audit was run.

## Next action

Keep raw PatchCore locked as the baseline. Do not train the screw refiner from the exposed MVTec masks. Obtain labeled defect development data independent of the evaluation holdout, or collect an untouched production-camera holdout, before training/selecting another learned refiner.
