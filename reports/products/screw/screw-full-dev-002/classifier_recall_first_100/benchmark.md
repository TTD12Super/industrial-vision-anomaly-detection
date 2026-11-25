# DINOv2 last-block + patch-aware GOOD/DEFECT benchmark

DINOv2 ViT-B/14 with the last Transformer block unfrozen. The classifier fuses global-token logits with top-k patch-token logits.

- Best epoch: `1` / `1`
- Threshold: `0.512145` (highest threshold meeting defect recall >= 1.00)

| Split | Accuracy | Balanced accuracy | DEFECT precision | DEFECT recall | DEFECT F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.3188 | 0.5648 | 0.2419 | 1.0000 | 0.3896 | 0.7556 | 0.4726 |
| test | 0.4103 | 0.5692 | 0.3284 | 0.9565 | 0.4889 | 0.7486 | 0.6394 |

- Checkpoint: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/products/screw/screw-full-dev-002/classifier_recall_first_100/model.pt`
- YOLO was not retrained in this experiment.
