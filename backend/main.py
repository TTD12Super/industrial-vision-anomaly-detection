"""FastAPI application: Industrial Vision Anomaly Detection API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import categories, detection, health
from .config import get_settings
from .database.mongodb import DatabaseUnavailableError

logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init MongoDB (indexes) without touching the AI pipeline.

    The app starts in degraded mode when MongoDB is down; detectors stay
    lazy-loaded per category on first inference request.
    """
    settings = get_settings()
    try:
        from .database.mongodb import connect
        from .database.repository import ensure_indexes

        client = connect(settings)
        ensure_indexes(client[settings.mongodb_database][settings.mongodb_collection])
        logger.info("Startup complete (database=connected).")
    except DatabaseUnavailableError as exc:
        logger.warning("Startup complete in degraded mode: %s", exc)
    logger.info(
        "Starting API (bank_dir=%s, predictions_dir=%s, cors=%s).",
        settings.bank_dir,
        settings.predictions_dir,
        settings.cors_origins,
    )
    yield
    try:
        from .database.mongodb import close_client

        close_client()
    except Exception:  # shutdown must not fail
        logger.exception("Error closing MongoDB client.")


app = FastAPI(
    title="Industrial Vision Anomaly Detection API",
    description=(
        "REST API for industrial visual anomaly detection. "
        "Upload an image plus its MVTec category to get a continuous "
        "anomaly score (frozen DINOv2 + exact kNN, see `model/inference`) "
        "and visualization URLs, with MongoDB-backed detection history. "
        "No binary Normal/Anomaly label is returned: "
        "the project has no calibrated threshold yet."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(health.router)
app.include_router(categories.router)
app.include_router(detection.router)


@app.get("/", tags=["health"], summary="API root")
def read_root() -> dict[str, str]:
    return {"message": "Industrial Vision Anomaly Detection API. See /docs."}
