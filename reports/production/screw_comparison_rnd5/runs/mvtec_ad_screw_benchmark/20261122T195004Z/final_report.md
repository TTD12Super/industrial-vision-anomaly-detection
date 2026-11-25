# Production run report: `mvtec_ad_screw_benchmark`

This report was generated automatically after the one-command pipeline completed.

## Pipeline

`raw validation → preprocessing → train/validation split → classification → Exact kNN decision → optional localization → evaluation`

- Dataset root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_benchmark`
- Processed root: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_benchmark_processed`
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
- Mean / median / p95: **0.1344 / 0.1340 / 0.1386 s/image**
- Throughput: **7.44 images/s**

## Generated files

- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/final_report.md`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/final_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/classification_test_summary.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/anomaly_metrics.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/per_category_metrics.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/error_analysis.csv`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/confusion_matrix.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/confusion_matrix_with_unknown.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/dataset_distribution.png`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/production/screw_comparison_rnd5/runs/mvtec_ad_screw_benchmark/20261122T195004Z/heatmaps`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd5/mvtec_ad_screw_benchmark/variants/exact_knn/thresholds/exact_knn/image_thresholds.json`
- `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/screw_comparison_rnd5/mvtec_ad_screw_benchmark/variants/exact_knn/thresholds/exact_knn/pixel_thresholds.json`
