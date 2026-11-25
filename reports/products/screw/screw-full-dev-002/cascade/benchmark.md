# Screw cascade fusion benchmark

Independent experiment branch: `codex/screw-cascade-fusion`.

Protocol: recall-first candidate union (`max(DINOv2 probability, AnomalyCLIP score)`) followed by a supervised logistic verifier using DINOv2 distances, AnomalyCLIP score, heatmap statistics, and a hard-negative weight for GOOD candidates.

| Split | Accuracy | DEFECT recall | GOOD false-positive rate | Image AUROC | Image AP |
|---|---:|---:|---:|---:|---:|
| Validation | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| Test | 0.705 | 0.348 | 0.145 | 0.644 | 0.496 |

Target on test: DEFECT recall >= 0.90; GOOD false-positive rate <= 0.25.
Result: NOT MET.

The validation result is not used as the final claim: it is perfect here, but the test result shows a domain/generalization gap. Production configuration was not changed.

Artifacts: `/Users/trantrungdung/Documents/523H0130_523H0038_DACNTT/industrial-vision-anomaly-detection/artifacts/products/screw/screw-full-dev-002/cascade/cascade_fusion.pkl` and cached AnomalyCLIP features under `artifacts/products/screw/screw-full-dev-002/cascade_cache/`.
