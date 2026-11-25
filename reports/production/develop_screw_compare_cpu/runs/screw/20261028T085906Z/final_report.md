# Production run report: `screw`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/tmp.u9IHUmarrw/screw`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_compare_processed`
- Categories: **1**
- Classification: **disabled (one category)**
- Image decision authority: **padim**
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
- Image AUROC / AP: **0.49354375896700153 / 0.7625637366986275**
- Pixel AUROC / AP: **0.9196759058341961 / 0.026134489670110016**

## Inference latency

- Device: **cpu**
- Mean / median / p95: **0.0740 / 0.0739 / 0.0767 s/image**
- Throughput: **13.51 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/develop_screw_compare_cpu/runs/screw/20261028T085906Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/develop_screw_compare_cpu/screw/variants/padim/thresholds/padim/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/develop_screw_compare_cpu/screw/variants/padim/thresholds/padim/pixel_thresholds.json`
