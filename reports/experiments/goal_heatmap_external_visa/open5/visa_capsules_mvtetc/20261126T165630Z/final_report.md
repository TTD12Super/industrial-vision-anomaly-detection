# Production run report: `visa_capsules_mvtetc`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → product classification/routing → independent PatchCore heatmap and mask evaluation → evaluation`

- Dataset root: `/private/tmp/ivision_visa_download/visa_capsules_mvtetc`
- Processed root: `/private/tmp/ivision_visa_download/visa_processed_320`
- Categories: **1**
- Classification: **disabled (one category)**
- Orchestration: **independent**
- Configured detector: **patchcore**
- Image decision authority: **patchcore**
- Product confidence threshold: **0.50**
- kNN neighbors: **1**

## Product classification

- Official test images: **160**
- Top-1 accuracy over all images: **100.00%**
- Top-1 accuracy over accepted images: **100.00%**
- UNKNOWN/rejected: **0 (0.00%)**
- Macro precision / recall / F1: **1.0000 / 1.0000 / 1.0000**
- Good-image accuracy: **100.00%**; defect-image accuracy: **100.00%**
- Accuracy drop on defect images: **0.00%**

## Defect detection

- Images with anomaly inference: **160**
- Images without completed anomaly inference: **0**
- Image AUROC / AP: **0.8371666666666666 / 0.9080197789229328**
- Pixel AUROC / AP: **0.976540268110706 / 0.2996850527988648**

## Localization coverage

- Classifier-rejected rows: **0**
- `patchcore` rejected localization: maps **0/0 (n/a)**; masks **0/0 (n/a)**

## GOOD-vs-DEFECT localization

- `patchcore` GOOD: map coverage **60/60 (100.00%)**; mask coverage **60/60 (100.00%)**; false-positive pixel rate **1.12%** (conditional on available GOOD masks; interpret with GOOD mask coverage); mean predicted-mask area **1145.1333 px (1.12%)**.
- `patchcore` DEFECT: map coverage **100/100 (100.00%)**; mask coverage **100/100 (100.00%)**; Pixel AUROC/AP **0.9705 / 0.3049**; precision/recall/F1/specificity/Dice/IoU **0.1038 / 0.8669 / 0.1854 / 0.9590 / 0.1854 / 0.1022**.

## Inference latency

- Device: **mps**
- Mean / median / p95: **0.1982 / 0.1929 / 0.2242 s/image**
- Throughput: **5.05 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/classification_validation_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/localization_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/experiments/goal_heatmap_external_visa/open5/visa_capsules_mvtetc/20261126T165630Z/heatmaps`
- `/private/tmp/ivision_visa_benchmark/artifacts/raw/visa_capsules_mvtetc/thresholds/patchcore/refiners/heatmap_open5/image_thresholds.json`
- `/private/tmp/ivision_visa_benchmark/artifacts/raw/visa_capsules_mvtetc/thresholds/patchcore/refiners/heatmap_open5/pixel_thresholds.json`
