# Backend — Industrial Vision Anomaly Detection API

FastAPI REST API wrapping the Phase 1 inference pipeline
(`model/inference`, frozen DINOv2 + exact kNN). The backend never
re-implements AI logic: it validates requests, manages uploads, reuses
`AnomalyDetector` instances, stores visualizations, and formats responses.

## Architecture

```text
Frontend (Phase 4)
  →  FastAPI (backend/main.py, backend/api/*)
    →  DetectionService (orchestration)
      →  DetectorService (per-category AnomalyDetector cache, lazy-load)
        →  AnomalyDetector.predict()  (Phase 1, unchanged)
      →  FileStorage (artifacts/predictions/<id>/input|heatmap|overlay.png)
      →  DetectionRepository → MongoDB `detections` (metadata/history only)
```

Detectors are lazy-loaded per category on first request and reused. Nothing
is preloaded at startup.

## Installation

```bash
py -m pip install -r requirements.txt
```

Adds over Phase 1: `fastapi`, `uvicorn`, `python-multipart`, `pymongo`.

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `APP_HOST` | `0.0.0.0` | Bind host |
| `APP_PORT` | `8000` | Bind port |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload limit |
| `BANK_DIR` | `artifacts/dinov2_knn/banks` | Per-category `feature_bank.pt` folders |
| `PREDICTIONS_DIR` | `artifacts/predictions` | Stored detection artifacts |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `industrial_anomaly_detection` | Database name |
| `MONGODB_COLLECTION` | `detections` | Collection name |

Copy `.env.example` to `.env` for local development (never commit `.env`).

## MongoDB setup

Local development (no credentials needed):

```text
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=industrial_anomaly_detection
MONGODB_COLLECTION=detections
```

Start the server with MongoDB running and indexes are created automatically
(`prediction_id` unique, plus `created_at`, `category`, `anomaly_score`).
Without MongoDB the API still starts but reports degraded health, and
history/persistence endpoints return 503. MongoDB stores metadata/paths/URLs
only — images stay in `FileStorage`, no GridFS.

## Run

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger: `http://localhost:8000/docs` · OpenAPI: `http://localhost:8000/openapi.json`

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness + MongoDB status (`connected`/`disconnected`) |
| GET | `/api/categories` | 15 MVTec categories (from Phase 1 config) |
| POST | `/api/detect` | Multipart `category` + `file` → score + artifact URLs + history record |
| GET | `/api/detections` | History (pagination, category/score filters, asc/desc sort) |
| GET | `/api/detections/{id}` | One detection (no `_id`/paths exposed) |
| DELETE | `/api/detections/{id}` | Delete history record + artifacts |
| GET | `/api/detections/{id}/input` | Stored input image (PNG) |
| GET | `/api/detections/{id}/heatmap` | Anomaly heatmap (PNG) |
| GET | `/api/detections/{id}/overlay` | Heatmap overlay (PNG) |

Query examples:

```text
GET /api/detections?page=1&page_size=20
GET /api/detections?category=bottle
GET /api/detections?min_score=0.1&max_score=0.5
GET /api/detections?sort_order=asc
```

Example:

```bash
curl -X POST http://localhost:8000/api/detect \
  -F category=bottle \
  -F file=@image.png
```

```json
{
  "prediction_id": "550e8400e29b41d4a716446655440000",
  "category": "bottle",
  "anomaly_score": 0.123456,
  "processing_time_ms": 1234,
  "result": {
    "input_url": "/api/detections/<id>/input",
    "heatmap_url": "/api/detections/<id>/heatmap",
    "overlay_url": "/api/detections/<id>/overlay"
  }
}
```

No `is_anomaly` field is returned: the project has no calibrated threshold.

## Feature bank requirement

`POST /api/detect` needs `<BANK_DIR>/<category>/feature_bank.pt`
(resolved via Phase 1 `default_bank_path`). Banks are git-ignored local
artifacts; build them with `model.inference.feature_bank.build_feature_bank`
(see `model/inference/README.md`). Missing bank → HTTP 503 with a clear
message. The API never builds banks, downloads datasets, or trains models.

## Error cases

| Case | Status |
|---|---|
| Unknown category | 400 |
| Empty / undecodable / wrong-type file | 400 |
| File over size limit | 413 |
| Missing multipart file | 422 |
| Missing feature bank or AI deps | 503 |
| MongoDB unavailable (persistence/history) | 503 |
| Inference failure | 500 (no traceback leaked) |
| Unknown prediction/artifact | 404 |

## Tests (mock AI + fake history)

```bash
py -m unittest discover -s backend/tests -t .
```

Inference is mocked (`MockDetector`) and history uses the real repository
code over an in-memory fake collection — no DINOv2 download, dataset, or
MongoDB server needed. MongoDB integration tests
(`test_mongo_integration.py`, isolated `..._test` database) run only when a
server is reachable, otherwise they skip with a clear message.

## Current limitations (by design)

No auth, no Redis/Celery (sync inference blocks the worker while DINOv2
runs), no threshold/binary labels, no Docker.
