# DINOv2 GOOD/DEFECT classifier benchmark

## Protocol

Frozen DINOv2 global embeddings + balanced logistic-regression binary head.
The original MVTec AD train/good images are training data. Official test images are split into validation and held-out test, so this is a supervised development benchmark, not the official MVTec AD protocol.

- Dataset: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/datasets/mvtec_ad_screw_compare_processed/screw`
- Device: `mps`
- DINOv2 embeddings: `3/3` split caches reused; recomputation time `0.00s`
- Threshold rule: highest threshold meeting defect recall >= 0.90

## Split

| Split | GOOD | DEFECT | Total |
|---|---:|---:|---:|
| train | 320 | 84 | 404 |
| validation | 20 | 18 | 38 |
| test | 21 | 17 | 38 |

## Metrics

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.9211 | 0.9222 | 0.8947 | 0.9444 | 0.9189 | 0.9639 | 0.9592 |
| Held-out test | 0.9737 | 0.9706 | 1.0000 | 0.9412 | 0.9697 | 0.9972 | 0.9967 |

- Selected threshold: `0.091734`
- Held-out confusion matrix `[[TN, FP], [FN, TP]]`: `[[21, 0], [1, 16]]`
- Classifier artifact: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/dinov2_binary_screw/classifier/classifier.pt`
- Split manifest: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/dinov2_binary_screw/split_manifest.csv`
