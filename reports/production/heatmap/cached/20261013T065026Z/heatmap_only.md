# Cached PatchCore heatmap-only evaluation

- Test images: **345** (every 5th sorted image)
- Heatmap pixel AUROC / AP: **0.949772 / 0.474177**
- Threshold: **heatmap > 0.70**
- Thresholded mask precision / recall / F1 / IoU: **0.653627 / 0.238617 / 0.349605 / 0.211831**
- False-positive rate: **0.004155**
- Image-score AUROC / AP: **0.922939 / 0.973783**
- Mean / median / p95 latency: **0.1532 / 0.1445 / 0.2050 s/image**
- PaDiM evaluation: **not run**; PatchCore thresholded mask: **evaluated**
