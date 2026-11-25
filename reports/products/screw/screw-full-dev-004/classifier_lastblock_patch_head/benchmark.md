# DINOv2 last-block + patch-aware GOOD/DEFECT benchmark

DINOv2 ViT-B/14 with the last Transformer block unfrozen. The classifier fuses global-token logits with top-k patch-token logits.

- Best epoch: `5` / `11`
- Threshold: `0.819638` (highest threshold meeting defect recall >= 0.90)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.9692 | 0.9567 | 0.9333 | 0.9333 | 0.9333 | 0.9933 | 0.9797 |
| test | 0.8846 | 0.8549 | 0.8182 | 0.7826 | 0.8000 | 0.9715 | 0.9377 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/products/screw/screw-full-dev-004/classifier_lastblock_patch_head/model.pt`
- YOLO was not retrained in this experiment.
