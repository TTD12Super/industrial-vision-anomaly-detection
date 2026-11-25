# Production run report: `screw`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/tmp.u9IHUmarrw/screw`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_compare_processed`
- Categories: **1**
- Classification: **disabled (one category)**
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
- Image AUROC / AP: **0.6868210698913713 / 0.8471865604356454**
- Pixel AUROC / AP: **0.8524937872879002 / 0.051676296547163915**

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.0855 / 0.0850 / 0.0872 s/image**
- Throughput: **11.69 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085800Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/develop_screw_compare_cpu/screw/variants/patchcore/thresholds/patchcore/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/develop_screw_compare_cpu/screw/variants/patchcore/thresholds/patchcore/pixel_thresholds.json`
