# Industrial Vision Anomaly Detection

An industrial computer vision project for detecting and localizing defects
in manufacturing products using deep learning and anomaly detection techniques.

## Overview

This project focuses on automated visual inspection in industrial environments.
The system analyzes product images to identify abnormal regions, detect defects,
and provide visual localization of the detected anomalies.

Pipeline: **frozen DINOv2 ViT-B/14 → patch features → exact kNN vs normal
bank → anomaly score/map → FastAPI → MongoDB history** (no threshold, no
Normal/Anomaly labels — continuous scores only).

## Setup

Requires Python 3.13 (Windows dev supported) and a running MongoDB for history.

```bash
py -m pip install -r requirements.txt
copy .env.example .env
```

`requirements.txt` covers AI (`torch`, `torchvision`, `Pillow`, `numpy`,
`matplotlib`, ...), API (`fastapi`, `uvicorn`, `python-multipart`) and
history (`pymongo`). CPU-only torch also works:
`py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## MongoDB

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=industrial_anomaly_detection
MONGODB_COLLECTION=detections
```

Without MongoDB the API starts degraded (health reports `disconnected`,
history endpoints return 503). Only metadata/paths are stored — never image
binaries. See `backend/README.md`.

## Dataset

The project uses **MVTec AD** (15 categories). The genuine dataset is NOT in
the repo (requires the official request form). Expected layout:

```text
datasets/mvtec_ad/<category>/{train/good, test/<defect>, ground_truth/...}
datasets/mvtec_ad_processed/<category>/{train/good, test/..., ground_truth/...}
```

`datasets/` is git-ignored. Preprocessing (256×256 RGB + masks) follows
`notebooks/01_data_exploration.ipynb`. Never substitute a different dataset
or commit dataset files.

## Feature bank

Banks live under `artifacts/dinov2_knn/banks/<category>/feature_bank.pt`
(git-ignored). Build from `train/good` only:

```bash
python -m model.inference.build_feature_bank --category bottle
python -m model.inference.build_feature_bank --all
```

See `model/inference/README.md` for the algorithm reference.

## DINOv2

Weights (`dinov2_vitb14`, revision `7764ea0…`) are fetched once by
`torch.hub` into the user cache (`~/.cache/torch/hub`) and never committed.
The backbone stays frozen; inference auto-selects device CUDA → MPS → CPU.

## Run backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`. Key endpoints: `GET /api/health`,
`GET /api/categories`, `POST /api/detect`, `GET /api/detections`,
`GET /api/detections/{id}`, `DELETE /api/detections/{id}`.

## Test

```bash
py -m unittest discover -s tests -t .          # Phase 1 (AI units)
py -m unittest discover -s backend/tests -t .  # API + history (mocked AI)
```

GPU/model tests are absent by design; `test_real_inference_e2e.py` runs the
full real flow only when a genuine bank + dataset image + MongoDB exist,
otherwise it skips with reasons.

## Real inference

With dataset + bank present:

```bash
python scripts/smoke_test_inference.py --category bottle
python scripts/smoke_test_inference.py --category bottle --image path/to/real.png
```

The script refuses to run without genuine data (no synthetic fallback, no
mock). Status (this machine): real DINOv2 loads and full pipeline machinery
is verified; genuine MVTec images are still required for true anomaly
validation — see the latest phase report.

## Objectives

- Detect defective products automatically.
- Identify and localize abnormal regions in images.
- Reduce the dependency on manual visual inspection.
- Evaluate different anomaly detection approaches.
- Build a reusable pipeline for industrial image inspection.

## Technologies

- Python
- PyTorch (DINOv2 ViT-B/14, exact kNN)
- FastAPI + MongoDB (API + detection history)
- Pillow, NumPy, Matplotlib
- Scikit-learn, Anomalib (evaluation references)
