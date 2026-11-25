# DINOv2 last-block + patch-aware GOOD/DEFECT benchmark

DINOv2 ViT-B/14 with the last Transformer block unfrozen. The classifier fuses global-token logits with top-k patch-token logits.

- Best epoch: `11` / `17`
- Threshold: `0.182649` (highest threshold meeting defect recall >= 0.90)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.7536 | 0.8185 | 0.4667 | 0.9333 | 0.6222 | 0.9346 | 0.8789 |
| test | 0.8846 | 0.8549 | 0.8182 | 0.7826 | 0.8000 | 0.9708 | 0.9345 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/products/screw/screw-full-dev-003/classifier_lastblock_patch_head/model.pt`
- YOLO was not retrained in this experiment.
