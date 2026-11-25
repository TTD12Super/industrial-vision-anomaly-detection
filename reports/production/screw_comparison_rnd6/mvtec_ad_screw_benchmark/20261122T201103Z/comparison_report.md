# Detector comparison: `mvtec_ad_screw_benchmark`

All three detectors were run independently on the same deterministic test split.

| Detector | Product routing | Image AUROC | Pixel AUROC | Pixel AP | Pixel F1 | Mean s/image | p95 s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_knn | 1.0000 | 0.9172 | 0.9159 | 0.3030 | 0.1097 | 0.1396 | 0.1444 |
| patchcore | 1.0000 | 0.6868 | 0.8525 | 0.0517 | 0.0720 | 0.0553 | 0.0562 |
| padim | 1.0000 | 0.4935 | 0.9197 | 0.0261 | 0.0661 | 0.0405 | 0.0411 |

- Image anomaly baseline: **exact_knn**
- Heatmap baseline: **patchcore**
- Combined mode: **sequential_independent_detectors**
- Combined inference time: **37.661 s**
- Official fusion: **not enabled; requires a measured validation/reliability gain**
