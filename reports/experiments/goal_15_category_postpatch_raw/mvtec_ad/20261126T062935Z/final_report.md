# Production run report: `mvtec_ad`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → product-category classification → independent PatchCore decision and localization → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/industrial-vision-mvtec-reuse/mvtec_ad`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_processed_320`
- Categories: **15**
- Classification: **enabled**
- Image decision authority: **patchcore**
- Product confidence threshold: **0.50**
- kNN neighbors: **1**

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

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.3818 / 0.3788 / 0.4149 s/image**
- Throughput: **2.62 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/classification_validation_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/localization_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_raw/mvtec_ad/20261126T062935Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/thresholds/patchcore/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/thresholds/patchcore/pixel_thresholds.json`
