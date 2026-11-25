"""The still-image vertical slice: detect, crop, classify, localize, report and store."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .artifacts import InspectionStore
from .geometry import OrientedCrop, oriented_crop, project_crop_map
from .interfaces import (
    KNOWN_PRODUCT,
    REVIEW_PRODUCT,
    UNKNOWN_PRODUCT,
    BinaryQualityClassifier,
    Detector,
    Localization,
    Localizer,
    ProductRecognizer,
    RecognitionResult,
)
from .policy import DEFECT, GOOD, NO_DETECTION, NOT_ASSESSED, REVIEW_REQUIRED, STAGE_FAILED, DecisionPolicy
from .reporting import AsyncReportManager, render_summary_pdf


STATUS_COLORS = {
    GOOD: (37, 139, 68),
    DEFECT: (202, 53, 53),
    REVIEW_REQUIRED: (222, 142, 31),
    NOT_ASSESSED: (102, 102, 102),
    STAGE_FAILED: (92, 92, 92),
}


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)


def _status_crop(crop: np.ndarray, status: str) -> np.ndarray:
    border = 14
    color = STATUS_COLORS[status]
    panel = np.full((crop.shape[0] + 2 * border, crop.shape[1] + 2 * border, 3), color, dtype=np.uint8)
    panel[border:-border, border:-border] = crop
    return panel


def _overlay_heatmap(crop: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(heatmap.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    if color.shape[:2] != crop.shape[:2]:
        color = cv2.resize(color, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(crop, 0.58, color, 0.42, 0.0)


class InspectionPipeline:
    """Composable single-image runner; heavy backends remain independent from API code."""

    def __init__(
        self,
        product_key: str,
        model_bundle_version: str,
        detector: Detector,
        classifier: BinaryQualityClassifier,
        policy: DecisionPolicy,
        localizer: Localizer,
        runs_root: str | Path,
        reporter: AsyncReportManager | None = None,
        product_recognizer: ProductRecognizer | None = None,
    ) -> None:
        self.product_key = product_key
        self.model_bundle_version = model_bundle_version
        self.detector = detector
        self.classifier = classifier
        self.policy = policy
        self.localizer = localizer
        self.runs_root = Path(runs_root)
        self.reporter = reporter
        self.product_recognizer = product_recognizer
        self.routing_mode = "product_recognizer" if product_recognizer is not None else "closed_world"

    def inspect_bytes(self, image_bytes: bytes, suffix: str, inspection_id: str) -> dict[str, Any]:
        store = InspectionStore(self.runs_root, inspection_id)
        source_path = store.path(f"input/original{suffix.lower() if suffix else '.png'}")
        source_path.write_bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = np.asarray(source.convert("RGB"))
        return self._inspect_image(image, store, hashlib.sha256(image_bytes).hexdigest())

    def inspect_path(self, path: str | Path, inspection_id: str) -> dict[str, Any]:
        source = Path(path)
        return self.inspect_bytes(source.read_bytes(), source.suffix or ".png", inspection_id)

    def _inspect_image(self, image: np.ndarray, store: InspectionStore, source_sha256: str) -> dict[str, Any]:
        started = time.perf_counter()
        inspection: dict[str, Any] = {
            "schema_version": "1.0",
            "inspection_id": store.inspection_id,
            "product_key": self.product_key,
            "source_type": "image",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "model_bundle_version": self.model_bundle_version,
            "source_sha256": source_sha256,
            "routing_mode": self.routing_mode,
            "objects": [],
        }
        detector_started = time.perf_counter()
        try:
            detections = self.detector.detect(image)
        except Exception as error:
            inspection["status"] = STAGE_FAILED
            inspection["stage_error"] = f"detector: {type(error).__name__}: {error}"
            detections = []
        inspection["latency_ms"] = {"detector": round((time.perf_counter() - detector_started) * 1000, 3)}

        ordered = sorted(
            detections,
            key=lambda item: (float(item.corners[:, 1].mean()), float(item.corners[:, 0].mean()), -item.score),
        )
        overview = image.copy()
        for index, detection in enumerate(ordered, 1):
            object_id = f"{self.product_key}_{index:02d}"
            object_started = time.perf_counter()
            result = self._inspect_object(image, detection.corners, detection.score, object_id, store)
            result.setdefault("latency_ms", {})["total"] = round((time.perf_counter() - object_started) * 1000, 3)
            inspection["objects"].append(result)
            polygon = np.rint(detection.corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overview, [polygon], True, STATUS_COLORS[result["operational_status"]], 2)
            anchor = tuple(np.rint(detection.corners[0]).astype(int))
            cv2.putText(overview, object_id, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, STATUS_COLORS[result["operational_status"]], 1)

        _save_rgb(store.path("overview/objects_labeled.png"), overview)
        statuses = [item["operational_status"] for item in inspection["objects"]]
        inspection["summary"] = {
            "detected_count": len(ordered),
            "good_count": statuses.count(GOOD),
            "defect_count": statuses.count(DEFECT),
            "review_required_count": statuses.count(REVIEW_REQUIRED),
            "not_assessed_count": statuses.count(NOT_ASSESSED),
            "no_detection": not ordered,
        }
        inspection["overview_image"] = "overview/objects_labeled.png"
        inspection["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000, 3)
        if any(item.get("p_defect") is None for item in inspection["objects"]):
            # The current PDF renderer requires numeric p_defect; never invent one for unassessed objects.
            inspection["summary_pdf"] = None
            inspection["summary_pdf_status"] = "not_rendered_unassessed"
        else:
            inspection["summary_pdf"] = "summary.pdf"
        if inspection["status"] == "processing":
            inspection["status"] = "completed_with_pending_reports" if any(
                item.get("vlm_status") == "pending" for item in inspection["objects"]
            ) else "completed"
        store.write_json("inspection.json", inspection)
        if inspection["summary_pdf"] is not None:
            render_summary_pdf(store.path("summary.pdf"), inspection)
        store.write_json("manifest.json", {"artifacts": store.manifest()})
        return inspection

    @staticmethod
    def _recognition_fields(recognition: RecognitionResult) -> dict[str, Any]:
        if recognition.status not in {KNOWN_PRODUCT, UNKNOWN_PRODUCT, REVIEW_PRODUCT}:
            raise ValueError(f"Unsupported product recognition status: {recognition.status}")
        if recognition.status == KNOWN_PRODUCT and not recognition.product_key:
            raise ValueError("KNOWN_PRODUCT requires product_key")
        if recognition.status in {UNKNOWN_PRODUCT, REVIEW_PRODUCT} and recognition.product_key is not None:
            raise ValueError(f"{recognition.status} must not carry product_key")
        return {
            "routing_mode": "product_recognizer",
            "product_recognition_status": recognition.status,
            "recognized_product_key": recognition.product_key,
            "routed_product_key": recognition.product_key if recognition.status == KNOWN_PRODUCT else None,
            "recognition_score": recognition.score,
            "recognition_margin": recognition.margin,
            "recognition_model_version": recognition.model_version,
            "recognition_reason": recognition.reason,
        }

    def _inspect_object(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        detection_score: float,
        object_id: str,
        store: InspectionStore,
    ) -> dict[str, Any]:
        relative = f"objects/{object_id}"
        try:
            crop = oriented_crop(image, corners)
            crop_path = store.path(f"{relative}/crop_original.png")
            _save_rgb(crop_path, crop.image)
            crop_sha256 = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        except Exception as error:
            return {
                "object_id": object_id,
                "detection_score": detection_score,
                "obb_xyxyxyxy": corners.tolist(),
                "crop_sha256": None,
                "routing_mode": self.routing_mode,
                "product_recognition_status": "NOT_ATTEMPTED",
                "quality_status": NOT_ASSESSED,
                "classifier_label": None,
                "p_defect": None,
                "operational_status": STAGE_FAILED,
                "needs_review": True,
                "stage_error": f"crop: {type(error).__name__}: {error}",
                "localization_status": "not_requested",
                "vlm_status": "not_requested",
            }
        recognition_fields: dict[str, Any] = {
            "routing_mode": "closed_world",
            "product_recognition_status": "NOT_CONFIGURED",
            "recognized_product_key": None,
            "routed_product_key": self.product_key,
            "recognition_score": None,
                "recognition_margin": None,
            "recognition_model_version": None,
            "recognition_reason": "product_recognizer_not_configured",
        }
        if self.product_recognizer is not None:
            try:
                recognition = self.product_recognizer.recognize(crop.image)
                recognition_fields = self._recognition_fields(recognition)
                if recognition.status != KNOWN_PRODUCT or recognition.product_key != self.product_key:
                    if recognition.status == KNOWN_PRODUCT:
                        recognition_fields["product_recognition_status"] = REVIEW_PRODUCT
                        recognition_fields["recognized_product_key"] = None
                        recognition_fields["routed_product_key"] = None
                        recognition_fields["recognition_reason"] = "recognized_product_bundle_unavailable"
                    operational_status = NOT_ASSESSED if recognition.status == UNKNOWN_PRODUCT else REVIEW_REQUIRED
                    result = {
                        "object_id": object_id,
                        "detection_score": detection_score,
                        "obb_xyxyxyxy": corners.tolist(),
                        "crop_sha256": crop_sha256,
                        **recognition_fields,
                        "classifier_label": None,
                        "p_defect": None,
                        "quality_status": NOT_ASSESSED,
                        "operational_status": operational_status,
                        "needs_review": True,
                        "decision_reason": recognition_fields["recognition_reason"] or "product_recognition_not_accepted",
                        "status_crop": f"{relative}/crop_status.png",
                        "localization_status": "not_requested",
                        "vlm_status": "not_requested",
                    }
                    _save_rgb(store.path(result["status_crop"]), _status_crop(crop.image, operational_status))
                    store.write_json(f"{relative}/result.json", result)
                    return result
            except Exception as error:
                _save_rgb(store.path(f"{relative}/crop_status.png"), _status_crop(crop.image, STAGE_FAILED))
                return {
                    "object_id": object_id,
                    "detection_score": detection_score,
                    "obb_xyxyxyxy": corners.tolist(),
                    "crop_sha256": crop_sha256,
                    **recognition_fields,
                    "classifier_label": None,
                    "p_defect": None,
                    "quality_status": NOT_ASSESSED,
                    "operational_status": STAGE_FAILED,
                    "needs_review": True,
                    "status_crop": f"{relative}/crop_status.png",
                    "stage_error": f"product_recognition: {type(error).__name__}: {error}",
                    "localization_status": "not_requested",
                    "vlm_status": "not_requested",
                }
        try:
            prediction = self.classifier.classify(crop.image)
            decision = self.policy.decide(prediction.p_defect)
        except Exception as error:
            _save_rgb(store.path(f"{relative}/crop_status.png"), _status_crop(crop.image, STAGE_FAILED))
            return {
                "object_id": object_id,
                "detection_score": detection_score,
                "obb_xyxyxyxy": corners.tolist(),
                **recognition_fields,
                "classifier_label": None,
                "p_defect": None,
                "quality_status": NOT_ASSESSED,
                "operational_status": STAGE_FAILED,
                "needs_review": True,
                "status_crop": f"{relative}/crop_status.png",
                "stage_error": f"classifier: {type(error).__name__}: {error}",
                "localization_status": "not_requested",
                "vlm_status": "not_requested",
            }

        result: dict[str, Any] = {
            "object_id": object_id,
            "detection_score": detection_score,
            "obb_xyxyxyxy": corners.tolist(),
            "crop_sha256": crop_sha256,
            **recognition_fields,
            "classifier_label": decision.classifier_label,
            "p_defect": decision.p_defect,
            "quality_status": decision.operational_status,
            "operational_status": decision.operational_status,
            "needs_review": decision.needs_review,
            "decision_reason": decision.reason,
            "classifier_model_version": prediction.model_version,
            "status_crop": f"{relative}/crop_status.png",
            "localization_status": "not_requested",
            "decision_source": prediction.model_version,
            "heatmap_source": None,
            "mask_source": None,
            "vlm_source": None,
            "vlm_status": "not_requested",
        }
        _save_rgb(store.path(result["status_crop"]), _status_crop(crop.image, decision.operational_status))
        if decision.operational_status in {DEFECT, REVIEW_REQUIRED}:
            self._localize(result, crop, image.shape[:2], relative, store)
        if decision.operational_status == DEFECT and self.reporter is not None:
            report_path = store.path(f"{relative}/vlm_report.json")
            evidence_paths = [
                path
                for path in sorted((store.directory / "input").glob("original.*"))
                if path.is_file()
            ]
            evidence_paths.append(store.path(f"{relative}/crop_original.png"))
            evidence_paths.extend(
                path
                for path in (
                    store.directory / relative / "heatmap_overlay.png",
                    store.directory / relative / "binary_mask.png",
                )
                if path.is_file()
            )
            evidence = [path.read_bytes() for path in evidence_paths]
            context = {
                "inspection_id": store.inspection_id,
                "object_id": object_id,
                "product_key": self.product_key,
                "classifier_label": result["classifier_label"],
                "p_defect": result["p_defect"],
                "decision_reason": result["decision_reason"],
                "localization_status": result["localization_status"],
                "heatmap_path": result.get("heatmap_overlay"),
                "binary_mask_path": result.get("binary_mask"),
                "_evidence_mime_types": [
                    "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                    for path in evidence_paths
                ],
            }
            pending = self.reporter.submit(
                context, evidence, report_path
            )
            result["vlm_report"] = f"{relative}/vlm_report.json"
            result["vlm_report_id"] = pending.report_id
            result["vlm_status"] = self.reporter.status(pending.report_id)
            result["vlm_source"] = getattr(self.reporter.reporter, "model_version", "unknown")
        store.write_json(f"{relative}/result.json", result)
        return result

    def _localize(
        self,
        result: dict[str, Any],
        crop: OrientedCrop,
        image_shape: tuple[int, int],
        relative: str,
        store: InspectionStore,
    ) -> None:
        try:
            localization: Localization = self.localizer.localize(crop.image)
        except Exception as error:
            result["localization_status"] = "failed"
            result["localization_error"] = f"{type(error).__name__}: {error}"
            return
        result["localization_status"] = localization.status
        result["heatmap_source"] = localization.model_version if localization.heatmap is not None else None
        result["mask_source"] = localization.model_version if localization.binary_mask is not None else None
        if localization.reason:
            result["localization_reason"] = localization.reason
        if localization.anomaly_score is not None:
            result["localization_anomaly_score"] = localization.anomaly_score
        if localization.heatmap is not None:
            heatmap = localization.heatmap.astype(np.float32)
            if heatmap.shape != crop.image.shape[:2]:
                heatmap = cv2.resize(heatmap, (crop.image.shape[1], crop.image.shape[0]), interpolation=cv2.INTER_LINEAR)
            _save_gray(store.path(f"{relative}/heatmap.png"), cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX))
            _save_rgb(store.path(f"{relative}/heatmap_overlay.png"), _overlay_heatmap(crop.image, heatmap))
            result["heatmap"] = f"{relative}/heatmap.png"
            result["heatmap_overlay"] = f"{relative}/heatmap_overlay.png"
            projected = project_crop_map(heatmap, crop, image_shape)
            _save_gray(store.path(f"{relative}/heatmap_full_image.png"), cv2.normalize(projected, None, 0, 255, cv2.NORM_MINMAX))
            result["heatmap_full_image"] = f"{relative}/heatmap_full_image.png"
        if localization.binary_mask is not None:
            mask = localization.binary_mask.astype(bool)
            if mask.shape != crop.image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (crop.image.shape[1], crop.image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            _save_gray(store.path(f"{relative}/binary_mask.png"), mask.astype(np.uint8) * 255)
            result["binary_mask"] = f"{relative}/binary_mask.png"
            projected = project_crop_map(mask, crop, image_shape)
            _save_gray(store.path(f"{relative}/binary_mask_full_image.png"), projected.astype(np.uint8) * 255)
            result["binary_mask_full_image"] = f"{relative}/binary_mask_full_image.png"
