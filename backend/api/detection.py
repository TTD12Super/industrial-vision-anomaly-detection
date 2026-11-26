"""Detection endpoints: upload -> inference -> history -> artifacts."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ..config import Settings
from ..database.mongodb import DatabaseUnavailableError
from ..database.repository import serialize_detection
from ..dependencies import (
    get_app_settings,
    get_detection_repository,
    get_detection_service,
    get_optional_repository,
    get_storage,
)
from ..schemas.detection import (
    DeleteDetectionResponse,
    DetectionDetail,
    DetectionListResponse,
    DetectionResponse,
    DetectionSummary,
)
from ..services.detection_service import (
    DetectionService,
    PersistenceError,
    UploadTooLargeError,
)
from ..services.detector_service import BankNotAvailableError, ModelNotAvailableError
from ..storage.file_storage import FileStorage

try:
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover - fallback when driver is missing

    class PyMongoError(Exception):
        """Fallback so routes import without pymongo installed."""


logger = logging.getLogger(__name__)

router = APIRouter(tags=["detection"])

_MEDIA_TYPES = {"input": "image/png", "heatmap": "image/png", "overlay": "image/png"}


@router.post(
    "/api/detect",
    response_model=DetectionResponse,
    summary="Detect anomalies in an uploaded image",
    description=(
        "Multipart upload with `category` (one of the 15 MVTec categories) and "
        "`file` (JPEG/PNG/WEBP). Returns a continuous `anomaly_score` plus URLs "
        "for the stored input, heatmap, and overlay images, and saves the "
        "detection to history. No binary Normal/Anomaly label is returned "
        "(no calibrated threshold exists)."
    ),
)
async def detect_anomaly(
    category: str = Form(description="MVTec category, e.g. 'bottle'."),
    file: UploadFile = File(description="Image file (JPEG/PNG/WEBP, max 10 MB)."),
    service: DetectionService = Depends(get_detection_service),
    settings: Settings = Depends(get_app_settings),
) -> DetectionResponse:
    data = await file.read()
    try:
        result = service.run_detection(
            category=(category or "").strip(),
            data=data,
            filename=file.filename,
            content_type=file.content_type,
            max_bytes=settings.max_upload_bytes,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (BankNotAvailableError, ModelNotAvailableError, PersistenceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Detection failed (category=%s).", category)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DetectionResponse(**result)


@router.get(
    "/api/detections",
    response_model=DetectionListResponse,
    summary="List detection history (newest first)",
)
def list_detections(
    page: int = Query(1, ge=1, description="Page number (1-based)."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)."),
    category: str | None = Query(None, description="Filter by MVTec category."),
    min_score: float | None = Query(None, description="Minimum anomaly score."),
    max_score: float | None = Query(None, description="Maximum anomaly score."),
    sort_order: str = Query("desc", description="'asc' or 'desc' by creation time."),
    repository=Depends(get_detection_repository),
) -> DetectionListResponse:
    if category is not None:
        from model.inference.config import validate_category

        try:
            validate_category(category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported category: {category}")
    if min_score is not None and max_score is not None and min_score > max_score:
        raise HTTPException(status_code=400, detail="min_score must be <= max_score.")
    if sort_order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="sort_order must be 'asc' or 'desc'.")
    try:
        items, total = repository.list_detections(
            category=category,
            min_score=min_score,
            max_score=max_score,
            page=page,
            page_size=page_size,
            sort_order=sort_order,
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Detection history is unavailable.") from exc
    return DetectionListResponse(
        items=[DetectionSummary(**serialize_detection(doc)) for doc in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/api/detections/{prediction_id}",
    response_model=DetectionDetail,
    summary="Get one detection",
    responses={404: {"description": "Detection not found."}},
)
def get_detection(
    prediction_id: str,
    repository=Depends(get_detection_repository),
) -> DetectionDetail:
    try:
        document = repository.get_detection(prediction_id)
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Detection history is unavailable.") from exc
    serialized = serialize_detection(document)
    if serialized is None:
        raise HTTPException(status_code=404, detail="Detection not found.")
    return DetectionDetail(**serialized)


@router.delete(
    "/api/detections/{prediction_id}",
    response_model=DeleteDetectionResponse,
    summary="Delete a detection and its artifacts",
    responses={404: {"description": "Detection not found."}},
)
def delete_detection(
    prediction_id: str,
    service: DetectionService = Depends(get_detection_service),
) -> DeleteDetectionResponse:
    try:
        deleted_id = service.delete_detection(prediction_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Detection not found.") from exc
    except PersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Detection history is unavailable.") from exc
    return DeleteDetectionResponse(
        message="Detection deleted successfully", prediction_id=deleted_id
    )


@router.get(
    "/api/detections/{prediction_id}/{kind}",
    summary="Download a stored detection artifact",
    description="`kind` is one of `input`, `heatmap`, `overlay`.",
    responses={404: {"description": "Prediction or artifact not found."}},
)
def get_artifact(
    prediction_id: str,
    kind: str,
    storage: FileStorage = Depends(get_storage),
    repository=Depends(get_optional_repository),
) -> FileResponse:
    if repository is not None:
        try:
            if not repository.exists(prediction_id):
                raise HTTPException(status_code=404, detail="Prediction not found.")
        except HTTPException:
            raise
        except (PyMongoError, DatabaseUnavailableError) as exc:
            # Degraded mode (DB down): fall back to filesystem-only serving.
            logger.warning("History lookup failed for %s (%s); serving from disk.", prediction_id, type(exc).__name__)
    try:
        path = storage.get_file(prediction_id, kind)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Prediction not found.") from exc
    return FileResponse(path, media_type=_MEDIA_TYPES[kind])
