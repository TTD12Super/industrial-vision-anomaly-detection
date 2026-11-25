# PatchCore heatmap refinement comparison

- Test images: **345**; validation GOOD images: **1089**

| Mask | Precision | Recall | F1 | IoU | FPR |
|---|---:|---:|---:|---:|---:|
| Fixed > 0.70 | 0.653627 | 0.238617 | 0.349605 | 0.211831 | 0.004155 |
| Adaptive small > 0.40, large > 0.80 | 0.520877 | 0.085812 | 0.147349 | 0.079534 | 0.002594 |
| Calibrated q=0.995 + refine | 0.218436 | 0.930499 | 0.353814 | 0.214929 | 0.109398 |

- Adaptive F1 delta vs fixed: **-0.202256**
- Adaptive IoU delta vs fixed: **-0.132297**
- F1 delta: **+0.004208**
- IoU delta: **+0.003098**
- Threshold range: **0.276492 – 0.509492**
- PaDiM: **not run**
