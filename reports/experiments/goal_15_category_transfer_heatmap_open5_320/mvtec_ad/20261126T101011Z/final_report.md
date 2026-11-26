# Production run report: `mvtec_ad`

This report was generated automatically after the one-command pipeline completed.

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
- Pixel AUROC / AP: **0.9626782293433396 / 0.5534663360264126**

## Localization coverage

- Classifier-rejected rows: **3**
- `patchcore` rejected localization: maps **3/3 (100.00%)**; masks **3/3 (100.00%)**

## GOOD-vs-DEFECT localization

- `patchcore` GOOD: map coverage **467/467 (100.00%)**; mask coverage **467/467 (100.00%)**; false-positive pixel rate **1.29%** (conditional on available GOOD masks; interpret with GOOD mask coverage); mean predicted-mask area **1322.1092 px (1.29%)**.
- `patchcore` DEFECT: map coverage **1258/1258 (100.00%)**; mask coverage **1258/1258 (100.00%)**; Pixel AUROC/AP **0.9536 / 0.5568**; precision/recall/F1/specificity/Dice/IoU **0.2048 / 0.9542 / 0.3372 / 0.8299 / 0.3372 / 0.2028**.

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.3881 / 0.3842 / 0.4225 s/image**
- Throughput: **2.58 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/classification_validation_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/localization_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_15_category_transfer_heatmap_open5_320/mvtec_ad/20261126T101011Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/variants/goal_15_heatmap_open5_320/thresholds/patchcore/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production_highres/mvtec_ad/variants/goal_15_heatmap_open5_320/thresholds/patchcore/pixel_thresholds.json`
