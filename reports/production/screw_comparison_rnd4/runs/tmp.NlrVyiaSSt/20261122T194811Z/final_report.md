# Production run report: `tmp.NlrVyiaSSt`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`

- Dataset root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/tmp.NlrVyiaSSt`
- Processed root: `/private/var/folders/x6/6k_h0_3j0m93sxlcrq42t4100000gn/T/tmp.NlrVyiaSSt_processed`
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
- Pixel AUROC / AP: **0.9196759575730152 / 0.026134537097283753**

## Inference latency

- Device: **mps**
- Mean / median / p95: **0.0443 / 0.0429 / 0.0499 s/image**
- Throughput: **22.58 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd4/runs/tmp.NlrVyiaSSt/20261122T194811Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd2/tmp.NlrVyiaSSt/variants/padim/thresholds/padim/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd2/tmp.NlrVyiaSSt/variants/padim/thresholds/padim/pixel_thresholds.json`
