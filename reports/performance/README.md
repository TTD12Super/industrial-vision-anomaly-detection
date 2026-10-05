# Performance measurement

Run:

```bash
PYTHONPATH=src DAenv/bin/python reports/performance/measure_performance.py \
  --devices cpu,mps,cuda
```

The JSON report measures one local `bottle/good` image without optimization:
DINOv2, classification, routing distance, bank loading, kNN, total inference,
RSS high-water mark, and bank file size. A backend that is unavailable or
errors is recorded with its status instead of receiving a fabricated timing.
