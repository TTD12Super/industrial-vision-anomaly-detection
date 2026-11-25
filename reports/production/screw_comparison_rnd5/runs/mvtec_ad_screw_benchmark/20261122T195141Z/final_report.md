# Production run report: `mvtec_ad_screw_benchmark`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`

- Dataset root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_benchmark`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_benchmark_processed`
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
- Pixel AUROC / AP: **0.8524938906119783 / 0.05167612850132668**

## Inference latency

- Device: **mps**
- Mean / median / p95: **0.0574 / 0.0569 / 0.0608 s/image**
- Throughput: **17.41 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195141Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd5/mvtec_ad_screw_benchmark/variants/patchcore/thresholds/patchcore/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd5/mvtec_ad_screw_benchmark/variants/patchcore/thresholds/patchcore/pixel_thresholds.json`
