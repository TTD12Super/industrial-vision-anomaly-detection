# Inspectra frontend

This frontend is a dependency-free static web app. It uses the existing FastAPI API and does not include a model, score threshold, or local inference path.

## Run locally

From the repository root, start the backend in one terminal:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend static server in another terminal:

```powershell
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173`. The frontend expects the API at `http://localhost:8000` by default. Change it in **System status → Backend API address** if needed. The configured URL is stored in the browser's local storage.

## API contract used

- `GET /api/health`
- `GET /api/categories`
- `POST /api/detect` with multipart `category` and `file`
- `GET /api/detections` with pagination and filter query parameters
- `GET /api/detections/{prediction_id}`
- `DELETE /api/detections/{prediction_id}`
- `GET /api/detections/{prediction_id}/{input|heatmap|overlay}` for stored artifacts

The UI displays the returned continuous anomaly score. The API does not expose a binary prediction or production threshold.
