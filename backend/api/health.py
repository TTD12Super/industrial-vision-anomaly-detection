"""Health probe: backend liveness plus MongoDB status (never touches AI)."""

from fastapi import APIRouter, Depends, Response

from ..dependencies import get_database_status
from ..schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse, summary="Service health")
def get_health(
    response: Response, database: str = Depends(get_database_status)
) -> HealthResponse:
    if database != "connected":
        response.status_code = 503
        return HealthResponse(
            status="degraded",
            service="industrial-vision-anomaly-detection",
            database="disconnected",
        )
    return HealthResponse(
        status="ok",
        service="industrial-vision-anomaly-detection",
        database="connected",
    )
