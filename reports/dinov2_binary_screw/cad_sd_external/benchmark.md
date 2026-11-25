# DINOv2 external test: CAD-SD

Threshold was kept from the MVTec supervised development validation; it was not refit on CAD-SD.

- Images: 376 (210 GOOD, 166 DEFECT)
- Threshold: `0.091734`

| Metric | Result |
|---|---:|
| Accuracy | 0.4415 |
| GOOD recall | 0.0000 |
| DEFECT precision | 0.4415 |
| DEFECT recall | 1.0000 |
| DEFECT F1 | 0.6125 |
| Image AUROC | 0.5593 |
| Image AP | 0.5086 |

- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[0, 210], [0, 166]]`
- Mean latency: `0.1018s/image`

## Per class

| Class | Images | Predicted DEFECT | Mean P(DEFECT) |
|---|---:|---:|---:|
| 0_good | 210 | 210 | 0.9999 |
| 1_scratch | 41 | 41 | 1.0000 |
| 2_paint | 41 | 41 | 0.9998 |
| 3_over-coupling | 44 | 44 | 1.0000 |
| 4_lacking | 40 | 40 | 0.9913 |
