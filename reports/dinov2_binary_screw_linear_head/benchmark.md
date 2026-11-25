# DINOv2 GOOD/DEFECT trainable head benchmark

Frozen DINOv2 backbone with a trainable `LayerNorm → Dropout → Linear(768, 2)` head.

- Best epoch: `173` / `253`
- Threshold: `0.143656` (highest threshold meeting defect recall >= 0.90)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.8684 | 0.8722 | 0.8095 | 0.9444 | 0.8718 | 0.9389 | 0.9312 |
| test | 0.9737 | 0.9706 | 1.0000 | 0.9412 | 0.9697 | 0.9720 | 0.9782 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/dinov2_binary_screw_linear_head/head.pt`
- Manifest: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/reports/dinov2_binary_screw/split_manifest.csv`
- The DINOv2 backbone was not updated in this run.
