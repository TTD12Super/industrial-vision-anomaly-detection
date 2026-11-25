# DINOv2 GOOD/DEFECT trainable head benchmark

Frozen DINOv2 backbone with a trainable `LayerNorm → Dropout → Linear(768, 2)` head.

- Best epoch: `59` / `139`
- Threshold: `0.707872` (highest threshold meeting defect recall >= 0.90)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.9855 | 0.9667 | 1.0000 | 0.9333 | 0.9655 | 1.0000 | 1.0000 |
| test | 0.6795 | 0.5577 | 0.4286 | 0.2609 | 0.3243 | 0.7194 | 0.5116 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/products/screw/screw-full-dev-002/classifier_linear_head/head.pt`
- Manifest: `/private/tmp/screw-full-embedding-manifest.csv`
- The DINOv2 backbone was not updated in this run.
