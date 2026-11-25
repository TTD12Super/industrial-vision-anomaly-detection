"""FastAPI still-image API with async job state and safe artifact retrieval."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .artifacts import InspectionStore
from .pipeline import InspectionPipeline
from .runtime import ProductRegistry, build_pipeline, product_readiness


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@dataclass
class Job:
    inspection_id: str
    future: Future[dict]
    pipeline: InspectionPipeline | None


class InspectionService:
    """One bounded worker prevents simultaneous API requests from racing GPU models."""

    def __init__(
        self,
        registry: ProductRegistry,
        pipeline_factory: Callable[[object], InspectionPipeline] = build_pipeline,
    ) -> None:
        self.registry = registry
        self.pipeline_factory = pipeline_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def submit(self, product_key: str, image_bytes: bytes, suffix: str) -> str:
        config = self.registry.get(product_key)
        inspection_id = f"insp_{uuid4().hex}"
        placeholder: Future[dict] = Future()
        with self.lock:
            job = Job(inspection_id, placeholder, None)
            self.jobs[inspection_id] = job
        future = self.executor.submit(self._run, job, config, image_bytes, suffix)
        with self.lock:
            job.future = future
        return inspection_id

    def _run(self, job: Job, config: object, image_bytes: bytes, suffix: str) -> dict:
        pipeline = self.pipeline_factory(config)
        with self.lock:
            job.pipeline = pipeline
        return pipeline.inspect_bytes(image_bytes, suffix, job.inspection_id)

    def result(self, inspection_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(inspection_id)
        if job is None:
            raise KeyError(inspection_id)
        if not job.future.done():
            return {"inspection_id": inspection_id, "status": "processing"}
        try:
            result = job.future.result()
        except Exception as error:
            return {"inspection_id": inspection_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        reporter = job.pipeline.reporter if job.pipeline is not None else None
        if reporter is not None:
            for item in result.get("objects", []):
                report_id = item.get("vlm_report_id")
                if report_id:
                    item["vlm_status"] = reporter.status(report_id)
            if result.get("status") == "completed_with_pending_reports" and not any(
                item.get("vlm_status") == "pending" for item in result["objects"]
            ):
                result["status"] = "completed"
        return result

    def artifact_store(self, inspection_id: str) -> InspectionStore:
        with self.lock:
            job = self.jobs.get(inspection_id)
        if job is None:
            raise KeyError(inspection_id)
        if job.pipeline is None:
            raise RuntimeError("Inspection worker has not initialized its pipeline")
        return InspectionStore(job.pipeline.runs_root, inspection_id)


def create_app(config_dir: str | Path = "configs/products", service: InspectionService | None = None) -> FastAPI:
    registry = ProductRegistry(config_dir)
    service = service or InspectionService(registry)
    app = FastAPI(title="Industrial Visual Inspection", version="1.0")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        try:
            products = registry.configs()
            details = [product_readiness(config) for config in products.values()]
            return {"ready": bool(details) and all(item["ready"] for item in details), "products": details}
        except Exception as error:
            return {"ready": False, "reason": f"{type(error).__name__}: {error}"}

    @app.get("/api/v1/products")
    def products() -> list[dict[str, str]]:
        try:
            return [
                {"product_key": config.product_key, "model_bundle_version": str(config.policy.get("model_bundle_version", "unversioned"))}
                for config in registry.configs().values()
            ]
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/v1/inspections", status_code=202)
    async def create_inspection(product_key: str = Form(...), file: UploadFile = File(...)) -> dict[str, str]:
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image MIME type")
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image must be non-empty and no larger than 20 MiB")
        try:
            inspection_id = await run_in_threadpool(service.submit, product_key, image_bytes, Path(file.filename or "upload.png").suffix)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Inspection unavailable: {error}") from error
        return {"inspection_id": inspection_id, "status": "processing"}

    @app.get("/api/v1/inspections/{inspection_id}")
    def inspection(inspection_id: str) -> dict:
        try:
            return service.result(inspection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Inspection not found") from error

    @app.get("/api/v1/inspections/{inspection_id}/artifacts")
    def artifacts(inspection_id: str) -> list[dict[str, str]]:
        try:
            store = service.artifact_store(inspection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Inspection not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return [
            {"artifact_id": artifact_id, "name": details["path"], "sha256": details["sha256"]}
            for artifact_id, details in store.manifest().items()
        ]

    @app.get("/api/v1/inspections/{inspection_id}/artifacts/{artifact_id}")
    def artifact(inspection_id: str, artifact_id: str) -> FileResponse:
        try:
            store = service.artifact_store(inspection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Inspection not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        details = store.manifest().get(artifact_id)
        if details is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(store.path(details["path"]), filename=Path(details["path"]).name)

    return app
