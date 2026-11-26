# Inference Pipeline (DINOv2 + exact kNN)

Reference / source of truth: `notebooks/02_first_model.ipynb` (baseline).
This package refactors that notebook into importable modules **without
changing the algorithm**, ready to be used later by a FastAPI backend.

## How the pipeline works

1. **Preprocessing** (`preprocessing.py`): input (file path, PIL image, or
   numpy array) → RGB → resize 256×256 (LANCZOS) → tensor → ImageNet
   mean/std normalization.
2. **Patch padding** (`preprocessing.py`): 256 is not divisible by the DINOv2
   patch size 14, so the batch is replicate-padded to **266×266**
   (pad right/bottom = 10). Expected grid **19×19 = 361 patches/image**.
3. **Feature extraction** (`feature_extractor.py`): frozen `dinov2_vitb14`
   (revision `7764ea0…`, `trust_repo=True`), `eval()` mode, final-layer
   `x_norm_patchtokens`, finite-value checks, L2 normalization.
4. **Feature bank** (`feature_bank.py`): every patch of every `train/good`
   image of one category. Never built from test images. Stored as
   `{"features": Tensor, "metadata": dict}` in `.pt` files (git-ignored).
5. **Exact kNN** (`knn.py`): Euclidean `torch.cdist`, k=1, chunked over
   queries (512) and bank rows (16384). No FAISS, no approximation.
6. **Scoring**: image anomaly score = **maximum** patch distance.
   Continuous scores only — **no Normal/Anomaly threshold** exists yet.
7. **Anomaly map** (`anomaly_map.py`): scores → 19×19 grid → bilinear
   interpolate to 266×266 → crop to 256×256. Heatmap/overlay helpers are
   display-only and never affect scores.

## Module layout

```text
model/inference/
├── __init__.py          # public API re-exports
├── config.py            # all constants, categories, device selection
├── preprocessing.py     # image loading, normalization, padding math
├── feature_extractor.py # DINOv2FeatureExtractor
├── feature_bank.py      # build / save / load normal banks
├── knn.py               # exact kNN (torch.cdist, chunked)
├── anomaly_map.py       # map reconstruction + heatmap/overlay
└── detector.py          # AnomalyDetector (end-to-end, one category)
```

## Dependencies

`torch`, `torchvision`, `Pillow`, `numpy`, `matplotlib`
(see `requirements.txt`). Weights are fetched/cached by `torch.hub` on first
use and are never committed.

## Supported categories (15)

bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill,
screw, tile, toothbrush, transistor, wood, zipper.

## Build a feature bank

```python
from model.inference.feature_extractor import DINOv2FeatureExtractor
from model.inference.feature_bank import build_feature_bank, list_train_images

extractor = DINOv2FeatureExtractor()
train_paths = list_train_images("datasets/mvtec_ad_processed/bottle/train/good")
bank, metadata = build_feature_bank(
    "bottle", train_paths, extractor,
    save_path="artifacts/dinov2_knn/banks/bottle/feature_bank.pt",
    dataset_root="datasets/mvtec_ad_processed",
)
```

Or via CLI (same logic, no algorithm change):

```bash
python -m model.inference.build_feature_bank --category bottle
python -m model.inference.build_feature_bank --all
```

Real-inference smoke test (requires genuine data, refuses otherwise):

```bash
python scripts/smoke_test_inference.py --category bottle
```

## Run inference

```python
from model.inference.detector import AnomalyDetector

detector = AnomalyDetector(
    category="bottle",
    bank_dir="artifacts/dinov2_knn/banks",  # .../bottle/feature_bank.pt
)
result = detector.predict("path/to/image.png")
print(result["anomaly_score"])  # float, higher = more anomalous
# result also has: patch_scores, anomaly_map (256x256),
# heatmap, overlay, patch_grid, inference_seconds, ...
```

Or with an explicit bank file:

```python
detector = AnomalyDetector(category="bottle", bank_path=".../feature_bank.pt")
```

## Result schema

| key | type | meaning |
|---|---|---|
| `category` | str | requested category |
| `anomaly_score` | float | max patch distance (continuous) |
| `aggregation` | str | `"max"` |
| `patch_scores` | np (361,) | per-patch distances |
| `anomaly_map` | np (256,256) | continuous localization map |
| `heatmap` / `overlay` | np uint8 RGB | display-only visualization |
| `patch_grid` / `padded_size` | tuple | e.g. `(19,19)` / `(266,266)` |
| `inference_seconds` | float | wall-clock time |

No `is_anomaly` field is returned: choosing a threshold requires separate
calibration data and is explicitly out of scope for this phase.

## Tests

```bash
py -m unittest discover -s tests -t .
```

Pure tests (config, padding math, bank metadata) always run. Tests needing
`torch`/`Pillow`/`numpy` skip with a clear message when those packages are
missing. Synthetic tiny tensors are used; no dataset, weights, or bank is
needed. Heavy-model tests (real DINOv2 forward) are intentionally absent.

## Not included (by design, Phase 1)

FastAPI, frontend, database, Docker, auth, thresholds, FAISS/ANN,
re-training, or any change to the notebook algorithm.
