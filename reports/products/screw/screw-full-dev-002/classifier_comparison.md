# Screw classifier comparison

All methods use the same 333/69/78 full-pool split and the same DINOv2 ViT-B/14 backbone.

| Method | Trainable part | Test accuracy | Test balanced accuracy | DEFECT precision | DEFECT recall | AUROC | Image AP |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen global + Logistic Regression | head only | 0.6410 | 0.5431 | 0.3684 | 0.3043 | 0.6672 | 0.4732 |
| Frozen global + regularized head | LayerNorm/Dropout/Linear | 0.6795 | 0.5577 | 0.4286 | 0.2609 | 0.7194 | 0.5116 |
| Last block + global/patch-aware head | last Transformer block + head | 0.6154 | 0.6387 | 0.4103 | 0.6957 | 0.7486 | 0.6394 |

The last-block/patch-aware candidate gives the best ranking and defect recall, but it flags too many GOOD objects (`GOOD recall = 0.5818`). It is therefore retained as an experiment, not promoted to the production config. Early stopping selected epoch 1, which is consistent with the small-data overfitting risk.

Artifacts:

- `reports/products/screw/screw-full-dev-002/classifier_linear_head/summary.json`
- `reports/products/screw/screw-full-dev-002/classifier_lastblock_patch_head/summary.json`
- `artifacts/products/screw/screw-full-dev-001/classifier_metrics.json`

The comparison is supervised-development evidence only; it is not an independent production benchmark.
