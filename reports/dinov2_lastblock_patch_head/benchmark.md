# DINOv2 last-block + patch-aware GOOD/DEFECT benchmark

DINOv2 ViT-B/14 with the last Transformer block unfrozen. The classifier fuses global-token logits with top-k patch-token logits.

- Best epoch: `7` / `13`
- Threshold: `0.736703` (highest threshold meeting defect recall >= 0.90)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.9737 | 0.9722 | 1.0000 | 0.9444 | 0.9714 | 1.0000 | 1.0000 |
| test | 0.9211 | 0.9118 | 1.0000 | 0.8235 | 0.9032 | 1.0000 | 1.0000 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/production/dinov2_lastblock_patch_head/model.pt`
- YOLO was not retrained in this experiment.
