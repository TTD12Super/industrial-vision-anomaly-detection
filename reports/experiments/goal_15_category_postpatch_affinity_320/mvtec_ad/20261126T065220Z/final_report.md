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
- Image AUROC / AP: **0.9229496532683332 / 0.973433579983695**
- Pixel AUROC / AP: **0.9528287550261305 / 0.5495973613063432**

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.3910 / 0.3875 / 0.4290 s/image**
- Throughput: **2.56 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/classification_validation_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/localization_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_postpatch_affinity_320/mvtec_ad/20261126T065220Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/thresholds/patchcore/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/thresholds/patchcore/pixel_thresholds.json`
