"""Detection orchestration: upload -> Phase 1 inference -> artifacts + MongoDB.

Route handlers stay thin; all business logic lives here. Raises:
* ``ValueError`` for client errors (bad category/image) -> HTTP 400
* ``BankNotAvailableError`` / ``ModelNotAvailableError`` -> HTTP 503
* ``RuntimeError`` for inference failures -> HTTP 500
* ``PersistenceError`` when MongoDB persistence fails -> HTTP 503

MongoDB insert happens only after inference + artifact generation succeed.
On insert failure the artifact directory is cleaned up so no orphan files
(and no fake database record) remain.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model.inference.config import validate_category

from .detector_service import DetectorService
from ..storage.file_storage import FileStorage, new_prediction_id

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class UploadTooLargeError(ValueError):
    """Uploaded file exceeds the configured limit (-> HTTP 413)."""


class PersistenceError(RuntimeError):
    """Detection metadata could not be persisted (-> HTTP 503)."""


def validate_upload(filename: str | None, content_type: str | None, size_bytes: int, max_bytes: int) -> str:
    """Validate an upload, returning the normalized extension (``.png`` ...)."""
    if size_bytes <= 0:
        raise ValueError("Uploaded file is empty.")
    if size_bytes > max_bytes:
        raise UploadTooLargeError(
            f"Uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit."
        )
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {filename!r}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type!r}.")
    return suffix


class DetectionService:
    """Runs detections with injected detector service, storage, and repository."""

    def __init__(
        self,
        detectors: DetectorService,
        storage: FileStorage,
        repository: Any | None = None,
    ) -> None:
        self.detectors = detectors
        self.storage = storage
        self.repository = repository

    def run_detection(
        self,
        category: str,
        data: bytes,
        filename: str | None,
        content_type: str | None,
        max_bytes: int,
    ) -> dict:
        started = time.perf_counter()
        try:
            validate_category(category)  # ValueError -> HTTP 400
        except ValueError:
            raise ValueError(f"Unsupported category: {category}") from None
        suffix = validate_upload(filename, content_type, len(data), max_bytes)

        prediction_id = new_prediction_id()
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            # Validates the bytes are a decodable image before inference.
            try:
                input_path = self.storage.save_upload(prediction_id, tmp_path)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Invalid image: {exc}.") from exc

            detector = self.detectors.get_detector(category)
            try:
                inference = detector.predict(str(input_path))
            except (ValueError, FileNotFoundError) as exc:
                raise ValueError(f"Invalid image: {exc}.") from exc
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError(f"Inference failed: {exc}.") from exc
            except Exception as exc:  # never leak tracebacks to clients
                logger.exception("Inference failed for prediction %s.", prediction_id)
                raise RuntimeError("Inference failed.") from exc

            self.storage.save_rgb_array(prediction_id, "heatmap", inference["heatmap"])
            self.storage.save_rgb_array(prediction_id, "overlay", inference["overlay"])
            processing_ms = int((time.perf_counter() - started) * 1000)
            result_urls = {
                "input_url": f"/api/detections/{prediction_id}/input",
                "heatmap_url": f"/api/detections/{prediction_id}/heatmap",
                "overlay_url": f"/api/detections/{prediction_id}/overlay",
            }
            self.storage.save_metadata(
                prediction_id,
                {
                    "prediction_id": prediction_id,
                    "category": category,
                    "anomaly_score": float(inference["anomaly_score"]),
                    "processing_time_ms": processing_ms,
                },
            )
            self._persist_metadata(
                prediction_id=prediction_id,
                category=category,
                anomaly_score=float(inference["anomaly_score"]),
                processing_ms=processing_ms,
                result_urls=result_urls,
                input_path=input_path,
            )
            logger.info(
                "Detection %s category=%s score=%.4f took=%dms.",
                prediction_id,
                category,
                float(inference["anomaly_score"]),
                processing_ms,
            )
            return {
                "prediction_id": prediction_id,
                "category": category,
                "anomaly_score": float(inference["anomaly_score"]),
                "processing_time_ms": processing_ms,
                "result": result_urls,
            }
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def _persist_metadata(
        self,
        *,
        prediction_id: str,
        category: str,
        anomaly_score: float,
        processing_ms: int,
        result_urls: dict[str, str],
        input_path: Path,
    ) -> None:
        """Insert the detection document; cleanup artifacts on failure."""
        if self.repository is None:
            logger.warning(
                "No repository configured; skipping MongoDB persistence for %s.",
                prediction_id,
            )
            return
        prediction_dir = self.storage.prediction_dir(prediction_id)
        document = {
            "prediction_id": prediction_id,
            "category": category,
            "anomaly_score": anomaly_score,
            "processing_time_ms": processing_ms,
            "created_at": datetime.now(timezone.utc),
            "result": result_urls,
            "artifacts": {
                "input_path": str(prediction_dir / "input.png"),
                "heatmap_path": str(prediction_dir / "heatmap.png"),
                "overlay_path": str(prediction_dir / "overlay.png"),
                "result_json_path": str(prediction_dir / "result.json"),
            },
        }
        try:
            self.repository.create_detection(document)
        except Exception as exc:
            logger.exception("MongoDB insert failed for %s; cleaning up.", prediction_id)
            shutil.rmtree(prediction_dir, ignore_errors=True)
            raise PersistenceError(
                "Detection could not be saved; please retry."
            ) from exc

    def delete_detection(self, prediction_id: str) -> str:
        """Delete history record + artifacts; raises ``LookupError`` (-> 404)."""
        if self.repository is None:
            raise PersistenceError("Detection history is unavailable.")
        document = self.repository.get_detection(prediction_id)
        if document is None:
            raise LookupError(f"Detection not found: {prediction_id}.")
        self.repository.delete_detection(prediction_id)
        try:
            shutil.rmtree(self.storage.prediction_dir(prediction_id), ignore_errors=True)
        except (OSError, ValueError) as exc:
            logger.warning("Artifact cleanup failed for %s: %s.", prediction_id, exc)
        logger.info("Deleted detection %s.", prediction_id)
        return prediction_id
