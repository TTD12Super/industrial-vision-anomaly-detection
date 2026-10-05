# Baseline reproduction

Run from the repository root with the project environment:

```bash
PYTHONPATH=src DAenv/bin/python reports/baseline_reproduction/reproduce_baseline.py
```

The command reads the existing `artifacts/dinov2_knn/final_run` scores/maps and
the read-only `datasets/mvtec_ad_processed` masks. It recomputes ranking
metrics with the refactored evaluator and writes
`baseline_reproduction.json`. It does not rebuild DINOv2, banks, or thresholds.

The report records dataset paths, category-level results, macro references,
deltas, and whether saved image scores match the maximum saved patch score
within a small floating-point tolerance.
