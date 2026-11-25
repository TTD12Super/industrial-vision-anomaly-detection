# Production run report: `tmp.NlrVyiaSSt`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/tmp.NlrVyiaSSt`
- Processed root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/tmp.NlrVyiaSSt_processed`
- Categories: **1**
- Classification: **disabled (one category)**
- Image decision authority: **exact_knn**
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
- Image AUROC / AP: **0.9171961467513835 / 0.9695789952080885**
- Pixel AUROC / AP: **0.9158688161554936 / 0.3029501276138822**

## Inference latency

- Device: **mps**
- Mean / median / p95: **0.1357 / 0.1349 / 0.1379 s/image**
- Throughput: **7.37 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd2/runs/tmp.NlrVyiaSSt/20261122T193905Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd2/tmp.NlrVyiaSSt/variants/exact_knn/thresholds/exact_knn/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd2/tmp.NlrVyiaSSt/variants/exact_knn/thresholds/exact_knn/pixel_thresholds.json`
