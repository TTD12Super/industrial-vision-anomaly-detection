# AnomalyCLIP + Relation Map: 15-category benchmark

**Status:** exploratory result; not an approved production method.

- Dataset scope: MVTec AD, 15 categories, 1725 test images.
- Defect images with masks: 1258; map coverage: 100%.
- Pixel AP/AUROC are macro averages across categories. Pooled micro AP/AUROC were not computed in this run.
- Operating-point precision/recall/F1 use category-specific thresholds. Earlier project use of the official test makes these results exploratory, not a fresh holdout estimate.

## Macro metrics

| Variant | Pixel AP | Pixel AUROC | Precision | Recall | Dice/F1 | Good-pixel FP | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnomalyCLIP raw | 31.68% | 88.36% | 25.72% | 54.98% | 29.75% | 0.88% | 96.33% |
| Relation Map | 21.65% | 82.54% | 12.80% | 65.27% | 19.36% | 2.21% | 74.37% |
| Fusion | 33.83% | 91.31% | 38.56% | 39.42% | 34.41% | 1.12% | 98.56% |

## Pooled operating-point metrics

| Variant | Precision | Recall | Dice/F1 | Good-pixel FP | Specificity |
|---|---:|---:|---:|---:|---:|
| AnomalyCLIP raw | 28.55% | 43.22% | 34.38% | 0.89% | 96.42% |
| Relation Map | 9.47% | 76.98% | 16.86% | 2.04% | 75.67% |
| Fusion | 45.54% | 33.62% | 38.68% | 1.08% | 98.67% |

## Reading the result

- Fusion has the highest macro Pixel AP and Pixel AUROC among these three variants, with the strongest macro precision/F1 operating point.
- Relation Map raises recall relative to raw AnomalyCLIP but has substantially lower precision and Pixel AP, consistent with broader false-positive regions.
- Full per-category metrics and sanitized provenance are in [`summary.json`](summary.json); [`heatmap_gallery.png`](heatmap_gallery.png) shows qualitative examples, and [`image_index.csv`](image_index.csv) maps gallery rows to test samples.
- Large per-image score tensors and the copied label tensor were removed after preserving aggregate/per-category metrics and the visual gallery. Those tensors are needed to recalculate alternative metrics without rerunning inference.
