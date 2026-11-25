# PatchCore heatmap refinement comparison

- Test images: **345**; validation GOOD images: **1089**

| Method | Precision | Recall | F1 | IoU | FPR |
|---|---:|---:|---:|---:|---:|
| fixed_0.50 | 0.422195 | 0.666363 | 0.516895 | 0.348522 | 0.029966 |
| fixed_0.60 | 0.547310 | 0.413144 | 0.470856 | 0.307922 | 0.011229 |
| fixed_0.70 | 0.653627 | 0.238617 | 0.349605 | 0.211831 | 0.004155 |
| fixed_0.80 | 0.706693 | 0.084960 | 0.151684 | 0.082066 | 0.001159 |
| fixed_0.70_open3_area9 | 0.653881 | 0.238307 | 0.349309 | 0.211614 | 0.004145 |
| fixed_0.70_area25 | 0.653626 | 0.238581 | 0.349566 | 0.211803 | 0.004154 |
| category_q995 | 0.218149 | 0.930576 | 0.353442 | 0.214655 | 0.109592 |
| category_q995_open3_area9 | 0.218436 | 0.930499 | 0.353814 | 0.214929 | 0.109398 |
| adaptive_0.40_0.80_area256 | 0.520877 | 0.085812 | 0.147349 | 0.079534 | 0.002594 |
| top1pct_open3_area9 | 0.444957 | 0.137857 | 0.210498 | 0.117629 | 0.005651 |

- Best method by F1/IoU: **fixed_0.50**
- Threshold range: **0.276492 – 0.509492**
- Best method selection is exploratory because it uses this 1/5 test subset.
- PaDiM: **not run**
