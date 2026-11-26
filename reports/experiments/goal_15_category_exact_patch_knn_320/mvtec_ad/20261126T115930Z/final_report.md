# Production run report: `mvtec_ad`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → product classification/routing → independent Exact kNN anomaly decision → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/industrial-vision-mvtec-reuse/mvtec_ad`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_processed_320`
- Categories: **15**
- Classification: **enabled**
- Orchestration: **independent**
- Configured detector: **exact_knn**
- Image decision authority: **exact_knn**
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

- Images with anomaly inference: **1722**
- Images without completed anomaly inference: **3**
- Image AUROC / AP: **0.9650460257471186 / 0.9875239963906001**
- Pixel AUROC / AP: **0.9423633599088702 / 0.4375077301045058**

## Localization coverage

- Classifier-rejected rows: **3**
- `exact_knn` rejected localization: maps **0/3 (0.00%)**; masks **0/3 (0.00%)**

## GOOD-vs-DEFECT localization

- `exact_knn` GOOD: map coverage **467/467 (100.00%)**; mask coverage **467/467 (100.00%)**; false-positive pixel rate **1.29%** (conditional on available GOOD masks; interpret with GOOD mask coverage); mean predicted-mask area **1318.6039 px (1.29%)**.
- `exact_knn` DEFECT: map coverage **1255/1258 (99.76%)**; mask coverage **1255/1258 (99.76%)**; Pixel AUROC/AP **0.9299 / 0.4407**; precision/recall/F1/specificity/Dice/IoU **0.2056 / 0.8983 / 0.3346 / 0.8433 / 0.3346 / 0.2009**.

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.4680 / 0.4304 / 0.6489 s/image**
- Throughput: **2.14 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/classification_validation_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/localization_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_exact_patch_knn_320/mvtec_ad/20261126T115930Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/variants/goal_15_exact_patch_knn_320/mvtec_ad/thresholds/exact_knn/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/variants/goal_15_exact_patch_knn_320/mvtec_ad/thresholds/exact_knn/pixel_thresholds.json`
