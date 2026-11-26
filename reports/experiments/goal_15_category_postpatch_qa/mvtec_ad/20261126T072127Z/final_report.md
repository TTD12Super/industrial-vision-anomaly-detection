# Production run report: `mvtec_ad`

Metrics were refreshed from the completed run's saved outputs after a report-schema-only update; inference predictions, scores, masks and thresholds are unchanged.

## Pipeline

`raw validation → preprocessing → train/validation split → product classification/routing → independent PatchCore heatmap and mask evaluation → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/industrial-vision-mvtec-reuse/mvtec_ad`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_processed_320`
- Categories: **15**
- Classification: **enabled**
- Orchestration: **independent**
- Configured detector: **patchcore**
- Image decision authority: **patchcore**
- Product confidence threshold: **0.50**
- kNN neighbors: **1**

## Benchmark validity

- Status: **exploratory development benchmark**, not an untouched final holdout. The official MVTec test set has been inspected in earlier method and resolution comparisons.
- Validation-derived thresholds were reused. Exact SHA-256 audit found zero cross-split exact duplicates and zero within-split duplicate groups. The audit does not rule out near-duplicates, acquisition/session leakage, or domain shift.
- See `qa_leakage_audit.json` and `benchmark_provenance.json` for audit counts and file hashes.

## Product classification

- Official test images: **1725**
- Top-1 accuracy over all images: **99.83%**
- Top-1 accuracy over accepted images: **100.00%**
- UNKNOWN/rejected: **3 (0.17%)**
- Macro precision / recall / F1: **1.0000 / 0.9982 / 0.9991**
- Good-image accuracy: **100.00%**; defect-image accuracy: **99.76%**
- Accuracy drop on defect images: **0.24%**

## Defect detection

- Images with anomaly inference: **1725**
- Images without completed anomaly inference: **0**
- Image AUROC / AP: **0.9299030104547172 / 0.9760952436273674**
- Pixel AUROC / AP: **0.9626459924713564 / 0.5537590635433803**

## Localization coverage

- Classifier-rejected rows: **3**
- `patchcore` rejected localization: maps **3/3 (100.00%)**; masks **3/3 (100.00%)**

## GOOD-vs-DEFECT localization

- `patchcore` GOOD: map coverage **467/467 (100.00%)**; mask coverage **467/467 (100.00%)**; false-positive pixel rate **1.29%** (conditional on available GOOD masks; interpret with GOOD mask coverage); mean predicted-mask area **1320.6488 px (1.29%)**.
- `patchcore` DEFECT: map coverage **1258/1258 (100.00%)**; mask coverage **1258/1258 (100.00%)**; Pixel AUROC/AP **0.9536 / 0.5571**; precision/recall/F1/specificity/Dice/IoU **0.2058 / 0.9533 / 0.3385 / 0.8312 / 0.3385 / 0.2037**.

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.3939 / 0.3913 / 0.4456 s/image**
- Throughput: **2.54 images/s**

## Generated files

- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/anomaly_metrics.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/classification_test_summary.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/classification_validation_summary.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/confusion_matrix.png`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/confusion_matrix_with_unknown.png`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/dataset_distribution.png`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/error_analysis.csv`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/final_report.md`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/thresholds/patchcore/image_thresholds.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/qa_leakage_audit.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/localization_metrics.csv`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/thresholds/patchcore/pixel_thresholds.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/benchmark_provenance.json`
- `reports/experiments/goal_15_category_postpatch_qa/mvtec_ad/20261126T072127Z/final_summary.json`
