# DINOv2 last-block + patch-aware GOOD/DEFECT benchmark

DINOv2 ViT-B/14 with the last Transformer block unfrozen. The classifier fuses global-token logits with top-k patch-token logits.

- Best epoch: `1` / `5`
- Threshold: `0.534062` (highest threshold meeting defect recall >= 0.90)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.6087 | 0.7259 | 0.3500 | 0.9333 | 0.5091 | 0.7556 | 0.4726 |
| test | 0.6154 | 0.6387 | 0.4103 | 0.6957 | 0.5161 | 0.7486 | 0.6394 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/products/screw/screw-full-dev-002/classifier_lastblock_patch_head/model.pt`
- YOLO was not retrained in this experiment.
