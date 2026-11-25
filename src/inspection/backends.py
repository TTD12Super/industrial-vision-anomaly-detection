"""Concrete optional backends. Imports of heavy model packages stay lazy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from .geometry import order_corners
from .interfaces import (
    KNOWN_PRODUCT,
    REVIEW_PRODUCT,
    UNKNOWN_PRODUCT,
    BinaryQualityClassifier,
    Classification,
    Detection,
    Detector,
    Localization,
    Localizer,
    ProductRecognizer,
    RecognitionResult,
)
from .mask_postprocess import postprocess_mask
from .relation_map import fuse_maps, relation_scores, scores_to_map


class BackendUnavailable(RuntimeError):
    """A configured backend cannot run and must not silently produce GOOD."""


def clip_detection_corners(corners: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    """Keep detector geometry inside the source image before any crop warp."""

    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Detector OBB corners must be four finite points")
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Detector image shape must be positive")
    clipped = corners.astype(np.float32, copy=True)
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
    clipped = order_corners(clipped)
    if cv2.contourArea(clipped.astype(np.float32)) < 1.0:
        return None
    return clipped


class YoloObbDetector:
    """YOLO-OBB detector returning original-image corner coordinates."""

    def __init__(self, weights: str | Path, confidence: float = 0.5, image_size: int = 1024, device: str | None = None) -> None:
        if not weights:
            raise BackendUnavailable("YOLO-OBB weights are required")
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise BackendUnavailable(f"YOLO-OBB checkpoint is missing: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise BackendUnavailable("ultralytics is not installed") from error
        self.model = YOLO(str(self.weights))
        self.confidence = confidence
        if image_size < 32:
            raise BackendUnavailable("YOLO-OBB image_size must be at least 32")
        self.image_size = image_size
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        kwargs: dict[str, Any] = {"source": image, "conf": self.confidence, "imgsz": self.image_size, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        obb = result.obb
        if obb is None or len(obb) == 0:
            return []
        corners = obb.xyxyxyxy.cpu().numpy()
        scores = obb.conf.cpu().numpy()
        detections: list[Detection] = []
        for points, score in zip(corners, scores):
            clipped = clip_detection_corners(points, image.shape[:2])
            if clipped is not None:
                detections.append(Detection(clipped, float(score)))
        return detections


class StaticDetector:
    """Deterministic in-memory detector for tests and example pipelines."""

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detections


class DinoV2LogRegClassifier:
    """Frozen DINOv2 global embedding with the existing binary logistic head."""

    def __init__(
        self,
        artifact_dir: str | Path,
        encoder_artifact_root: str | Path,
        image_size: tuple[int, int] = (256, 256),
        device: str | None = None,
        model_version: str = "dinov2-logreg",
    ) -> None:
        import torch

        from mvtec_system.classification.classifier import predict_linear_classifier
        from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device

        artifact_dir = Path(artifact_dir)
        classifier_path = artifact_dir / "classifier" / "classifier.pt"
        mapping_path = artifact_dir / "classifier" / "class_mapping.json"
        if not classifier_path.is_file() or not mapping_path.is_file():
            raise BackendUnavailable(f"DINOv2 classifier artifact is incomplete: {artifact_dir}")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8")).get("category_to_index")
        if mapping != {"GOOD": 0, "DEFECT": 1}:
            raise BackendUnavailable("DINOv2 classifier must use the binary GOOD/DEFECT class mapping")
        self.classifier = torch.load(classifier_path, map_location="cpu", weights_only=True)
        self.encoder = SharedDINOv2(
            DINOv2Config(image_size=image_size),
            artifact_root=encoder_artifact_root,
            device=device or str(select_device()),
        )
        self.image_size = image_size
        self.model_version = model_version

    def classify(self, crop: np.ndarray) -> Classification:
        import torch

        from mvtec_system.classification.classifier import predict_linear_classifier

        resized = cv2.resize(crop, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.encoder.config.mean).view(3, 1, 1)
        std = torch.tensor(self.encoder.config.std).view(3, 1, 1)
        normalized = ((tensor - mean) / std).unsqueeze(0)
        features = self.encoder.forward(normalized).global_embedding.cpu()
        _, probabilities = predict_linear_classifier(self.classifier, features)
        return Classification(float(probabilities[0, 1]), self.model_version)


class DinoV2PatchMemoryClassifier:
    """DINOv2 patch-memory anomaly classifier kept behind an experimental backend.

    The stored nearest-neighbour distance is converted to a bounded score only
    so it can satisfy the existing ``p_defect`` contract.  The 0.5 decision
    point is exactly the calibrated raw-distance threshold; the artifact, not
    the production policy, owns that threshold.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        encoder_artifact_root: str | Path = "artifacts/dinov2_knn",
        device: str | None = None,
        model_version: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        from .dinov2_patchcore_lite import load_patch_memory_artifact

        try:
            artifact = load_patch_memory_artifact(artifact_path)
        except (FileNotFoundError, OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise BackendUnavailable(f"DINOv2 patch-memory artifact is not ready: {error}") from error
        self.memory_bank = artifact["memory_bank"]
        self.raw_threshold = float(artifact["threshold"])
        self.k = int(artifact["k"])
        self.aggregation = str(artifact["aggregation"])
        self.model_version = model_version or str(artifact.get("model_version", "dinov2-patch-memory"))
        self.temperature = max(abs(self.raw_threshold) * 0.1, 1e-6)
        if encoder is None:
            from mvtec_system.core.dinov2 import SharedDINOv2, select_device
            from .product_recognition import _encoder_config

            config = _encoder_config(artifact["encoder"])
            encoder = SharedDINOv2(
                config,
                artifact_root=encoder_artifact_root,
                device=device or str(select_device()),
            )
        self.encoder = encoder
        embedding_dimension = int(self.encoder.metadata()["embedding_dimension"])
        if embedding_dimension != self.memory_bank.shape[1]:
            raise BackendUnavailable(
                "DINOv2 patch-memory encoder does not match memory-bank embedding dimension"
            )
        expected_patch_count = artifact["encoder"].get("patch_count")
        if expected_patch_count is not None and int(self.encoder.patch_count) != int(expected_patch_count):
            raise BackendUnavailable("DINOv2 patch-memory encoder does not match patch-grid contract")
        if int(self.encoder.patch_count) <= 0:
            raise BackendUnavailable("DINOv2 patch-memory encoder has no spatial patches")

    def classify(self, crop: np.ndarray) -> Classification:
        from mvtec_system.anomaly.knn import nearest_neighbor_scores
        from .dinov2_patchcore_lite import aggregate_patch_scores
        from .product_recognition import _crop_tensor

        try:
            tensor = _crop_tensor(crop, self.encoder.config).unsqueeze(0)
            features = self.encoder.forward(tensor)
            patch_scores = nearest_neighbor_scores(
                features.patch_embeddings[0].float(), self.memory_bank.float(), k=self.k
            ).numpy()
            raw_score = aggregate_patch_scores(patch_scores, self.aggregation)
            bounded_score = 0.5 + 0.5 * np.tanh(
                (raw_score - self.raw_threshold) / self.temperature
            )
        except Exception as error:
            raise BackendUnavailable(f"DINOv2 patch-memory inference failed: {error}") from error
        return Classification(float(np.clip(bounded_score, 0.0, 1.0)), self.model_version)


class StaticClassifier:
    """Deterministic classifier for contract and API tests."""

    def __init__(self, probabilities: list[float] | float) -> None:
        self.probabilities = probabilities if isinstance(probabilities, list) else [probabilities]
        self.index = 0

    def classify(self, crop: np.ndarray) -> Classification:
        probability = self.probabilities[min(self.index, len(self.probabilities) - 1)]
        self.index += 1
        return Classification(float(probability), "static-test-classifier")


class StaticProductRecognizer:
    """Deterministic recognizer for routing contract and pipeline tests."""

    def __init__(self, results: list[RecognitionResult] | RecognitionResult) -> None:
        self.results = results if isinstance(results, list) else [results]
        if not self.results:
            raise ValueError("at least one static recognition result is required")
        self.index = 0

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        result = self.results[min(self.index, len(self.results) - 1)]
        self.index += 1
        return result


class DinoV2GalleryRecognizer:
    """Nearest-product recognizer backed by a calibrated DINOv2 gallery."""

    def __init__(
        self,
        gallery_path: str | Path,
        calibration_path: str | Path,
        encoder_artifact_root: str | Path = "artifacts/dinov2_knn",
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        from mvtec_system.core.dinov2 import SharedDINOv2, select_device
        from .product_recognition import (
            _encoder_config,
            load_calibration_artifact,
            load_gallery_artifact,
        )

        try:
            gallery = load_gallery_artifact(gallery_path)
            calibration = load_calibration_artifact(calibration_path, gallery_path)
        except (FileNotFoundError, OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise BackendUnavailable(f"Product recognizer artifacts are not ready: {error}") from error
        self.gallery_embeddings = gallery["embeddings"]
        self.product_keys = gallery["product_keys"]
        self.threshold = float(calibration["known_threshold"])
        self.min_product_margin = (
            float(calibration["min_product_margin"])
            if len(set(self.product_keys)) > 1
            else None
        )
        self.model_version = str(gallery.get("model_version", "dinov2-gallery"))
        if encoder is None:
            config = _encoder_config(gallery["encoder"])
            encoder = SharedDINOv2(
                config,
                artifact_root=encoder_artifact_root,
                device=device or str(select_device()),
            )
        self.encoder = encoder
        if int(self.encoder.metadata()["embedding_dimension"]) != self.gallery_embeddings.shape[1]:
            raise BackendUnavailable("Product recognizer encoder does not match gallery embedding dimension")

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        import torch
        import torch.nn.functional as F

        from .product_recognition import _crop_tensor, _product_score_matrix

        try:
            tensor = _crop_tensor(crop, self.encoder.config).unsqueeze(0)
            query = self.encoder.forward(tensor).global_embedding.float().cpu()
            query = F.normalize(query, dim=1)
            product_keys, product_scores = _product_score_matrix(
                query, self.gallery_embeddings, self.product_keys
            )
            order = torch.argsort(product_scores, dim=1, descending=True)[0]
            top_index = int(order[0].item())
            score = float(product_scores[0, top_index].item())
            margin = None
            if len(product_keys) > 1:
                margin = score - float(product_scores[0, int(order[1].item())].item())
        except Exception as error:
            raise BackendUnavailable(f"Product recognizer inference failed: {error}") from error
        if score < self.threshold:
            return RecognitionResult(
                None,
                score,
                self.model_version,
                UNKNOWN_PRODUCT,
                "below_calibrated_threshold",
                margin,
            )
        if self.min_product_margin is not None and margin is not None and margin < self.min_product_margin:
            return RecognitionResult(
                None,
                score,
                self.model_version,
                REVIEW_PRODUCT,
                "below_calibrated_product_margin",
                margin,
            )
        return RecognitionResult(
            product_keys[top_index],
            score,
            self.model_version,
            KNOWN_PRODUCT,
            "nearest_gallery_product_above_calibrated_threshold",
            margin,
        )


class AnomalyClipLocalizer:
    """In-process AnomalyCLIP localizer with an explicit subprocess fallback script."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.model_version = str(settings.get("model_version", "anomalyclip"))
        self.session: AnomalyClipSession | None = None
        self.supervised_refiner: SupervisedPatchMaskRefiner | None = None
        self.dino_refiner: DinoAffinityRefiner | None = None
        if settings.get("supervised_refiner_enabled"):
            from .supervised_refinement import SupervisedPatchMaskRefiner

            artifact = settings.get("supervised_refiner_path")
            calibration_path = settings.get("supervised_refiner_calibration_path")
            if not artifact or not Path(artifact).is_file() or not calibration_path or not Path(calibration_path).is_file():
                raise BackendUnavailable("supervised refiner artifact or calibration is missing")
            calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            self.settings = dict(settings)
            self.settings["mask_threshold"] = float(calibration["mask_threshold"])
            self.settings["mask_postprocess"] = str(calibration.get("mask_postprocess", "raw"))
            self.supervised_refiner = SupervisedPatchMaskRefiner.load(artifact)
        self.prototype: np.ndarray | None = None
        relation_calibration_path = settings.get("relation_calibration_path")
        if settings.get("relation_fusion_enabled") and relation_calibration_path and Path(relation_calibration_path).is_file():
            relation_calibration = json.loads(Path(relation_calibration_path).read_text(encoding="utf-8"))
            settings = dict(settings)
            settings["mask_threshold"] = float(relation_calibration["mask_threshold"])
            self.settings = settings
        prototype_path = settings.get("relation_prototype_path")
        if settings.get("relation_fusion_enabled") and prototype_path and Path(prototype_path).is_file():
            with np.load(prototype_path) as data:
                self.prototype = np.asarray(data["prototype"], dtype=np.float32)

    def localize(self, crop: np.ndarray) -> Localization:
        runner = self.settings.get("runner")
        checkpoint = self.settings.get("checkpoint")
        repository = self.settings.get("repository")
        if not runner or not checkpoint or not repository:
            return Localization(None, None, None, "unavailable", "ANOMALYCLIP_NOT_CONFIGURED")
        paths = {"runner": Path(runner), "checkpoint": Path(checkpoint), "repository": Path(repository)}
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            return Localization(None, None, None, "unavailable", f"ANOMALYCLIP_MISSING_{'_'.join(missing).upper()}")
        try:
            from .anomalyclip import AnomalyClipSession

            self.session = self.session or AnomalyClipSession(self.settings)
            if self.settings.get("relation_fusion_enabled") and self.prototype is None:
                return Localization(None, None, None, "unavailable", "RELATION_PROTOTYPE_NOT_CONFIGURED")
            heatmap, anomaly_score, attention, features = self.session.infer(crop)
            if self.supervised_refiner is not None:
                from .dino_refinement import DinoAffinityRefiner
                from .supervised_refinement import patch_inputs

                dino_settings = dict(self.settings)
                dino_settings["artifact_root"] = dino_settings.get("dino_artifact_root", "artifacts/production/mvtec_ad")
                dino_settings["top_k"] = dino_settings.get("dino_top_k", 128)
                dino_settings["temperature"] = dino_settings.get("dino_temperature", 0.1)
                dino_settings["eta"] = dino_settings.get("dino_eta", 1.0)
                self.dino_refiner = self.dino_refiner or DinoAffinityRefiner(dino_settings)
                dino_features, side = self.dino_refiner.patch_features(crop)
                heatmap = self.supervised_refiner.predict_map(
                    patch_inputs(
                        dino_features,
                        heatmap,
                        side,
                        self.supervised_refiner.feature_mode,
                    ),
                    side,
                    crop.shape[:2],
                )
            if self.settings.get("relation_fusion_enabled"):
                relation = scores_to_map(relation_scores(attention, features, self.prototype), crop.shape[:2])
                baseline = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
                heatmap = fuse_maps(baseline, relation, float(self.settings.get("relation_beta", 1.0)))
        except Exception as error:
            return Localization(None, None, None, "failed", f"ANOMALYCLIP_RUNNER_FAILED: {type(error).__name__}: {error}")
        if heatmap.ndim != 2 or heatmap.size == 0 or not np.isfinite(heatmap).all():
            return Localization(None, None, None, "failed", "ANOMALYCLIP_INVALID_HEATMAP")
        threshold = self.settings.get("mask_threshold")
        mask = postprocess_mask(heatmap, float(threshold), str(self.settings.get("mask_postprocess", "raw"))) if threshold is not None else None
        return Localization(heatmap, mask, anomaly_score, "completed", model_version=self.model_version)


class PatchCorePaDiMLocalizer:
    """One DINOv2 forward pass with PatchCore heatmap and PaDiM mask."""

    @staticmethod
    def _single_map(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError("anomaly map batch must contain one object")
            array = array[0]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("anomaly map must be a finite 2-D array")
        return array

    def __init__(self, settings: dict[str, Any]) -> None:
        import torch

        from mvtec_system.anomaly.detectors import PaDiMDetector, PatchCoreDetector
        from mvtec_system.anomaly.multi_layer import MultiLayerConfig
        from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device
        from .product_recognition import _crop_tensor

        patchcore_dir = Path(settings.get("patchcore_artifact_dir", ""))
        padim_dir = Path(settings.get("padim_artifact_dir", ""))
        if not (patchcore_dir / "coreset.pt").is_file() or not (padim_dir / "statistics.pt").is_file():
            raise BackendUnavailable("PatchCore or PaDiM artifact is missing")
        try:
            patch_payload = torch.load(patchcore_dir / "coreset.pt", map_location="cpu", weights_only=True)
            padim_payload = torch.load(padim_dir / "statistics.pt", map_location="cpu", weights_only=True)
        except (EOFError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise BackendUnavailable(f"PatchCore/PaDiM artifact cannot be loaded: {error}") from error
        if not isinstance(patch_payload, dict) or not isinstance(padim_payload, dict):
            raise BackendUnavailable("PatchCore/PaDiM artifacts must contain dictionaries")
        patch_config = self._config(patch_payload.get("config"))
        padim_config = self._config(padim_payload.get("config"))
        if patch_config != padim_config:
            raise BackendUnavailable("PatchCore and PaDiM feature contracts do not match")
        self.patchcore = PatchCoreDetector(patch_config).load(patchcore_dir)
        self.padim = PaDiMDetector(padim_config).load(padim_dir)
        image_size = tuple(int(value) for value in settings.get("image_size", (256, 256)))
        if len(image_size) != 2 or min(image_size) < 32:
            raise BackendUnavailable("patchcore_padim image_size must contain height and width")
        self.encoder = SharedDINOv2(
            DINOv2Config(image_size=image_size, feature_layers=patch_config.layers),
            artifact_root=settings.get("encoder_artifact_root", "artifacts/production/mvtec_ad"),
            device=settings.get("device") or str(select_device()),
        )
        self.mask_threshold = float(settings.get("mask_threshold", 0.5))
        if not np.isfinite(self.mask_threshold):
            raise BackendUnavailable("patchcore_padim mask_threshold must be finite")
        self.mask_postprocess = str(settings.get("mask_postprocess", "raw"))
        self.model_version = str(settings.get("model_version", "patchcore-heatmap+padim-mask"))

    @staticmethod
    def _config(raw: Any) -> MultiLayerConfig:
        from mvtec_system.anomaly.multi_layer import MultiLayerConfig

        if not isinstance(raw, dict):
            raise BackendUnavailable("Multi-layer detector config is missing")
        values = dict(raw)
        values["layers"] = tuple(int(value) for value in values.get("layers", ()))
        try:
            return MultiLayerConfig(**values)
        except (TypeError, ValueError) as error:
            raise BackendUnavailable("Multi-layer detector config is invalid") from error

    def localize(self, crop: np.ndarray) -> Localization:
        from .product_recognition import _crop_tensor

        try:
            features = self.encoder.forward(_crop_tensor(crop, self.encoder.config).unsqueeze(0))
            patchcore = self.patchcore.predict(features)
            padim = self.padim.predict(features)
            heatmap = self._single_map(patchcore.anomaly_map)
            mask = postprocess_mask(
                self._single_map(padim.anomaly_map),
                self.mask_threshold,
                self.mask_postprocess,
            )
        except Exception as error:
            return Localization(None, None, None, "failed", f"PATCHCORE_PADIM_FAILED: {type(error).__name__}: {error}", self.model_version)
        if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
            return Localization(None, None, None, "failed", "PATCHCORE_INVALID_HEATMAP", self.model_version)
        return Localization(heatmap, mask, patchcore.image_score, "completed", model_version=self.model_version)


class StaticLocalizer:
    """Test localizer that returns a supplied map without influencing classification."""

    def __init__(self, heatmap: np.ndarray | None = None, mask: np.ndarray | None = None) -> None:
        self.heatmap = heatmap
        self.mask = mask

    def localize(self, crop: np.ndarray) -> Localization:
        if self.heatmap is None:
            return Localization(None, self.mask, None, "empty")
        return Localization(self.heatmap, self.mask, float(np.max(self.heatmap)), "completed")


def build_detector(settings: dict[str, Any]) -> Detector:
    if settings["backend"] != "yolo_obb":
        raise BackendUnavailable(f"Unsupported detector backend: {settings['backend']}")
    return YoloObbDetector(
        settings.get("weights"),
        float(settings.get("confidence", 0.5)),
        int(settings.get("image_size", 1024)),
        settings.get("device"),
    )


def build_classifier(settings: dict[str, Any]) -> BinaryQualityClassifier:
    if settings["backend"] == "dinov2_patch_memory":
        artifact_path = settings.get("artifact_path") or settings.get("patch_memory_artifact_path")
        if not artifact_path:
            raise BackendUnavailable("patch-memory classifier artifact_path is required")
        return DinoV2PatchMemoryClassifier(
            artifact_path,
            settings.get("encoder_artifact_root", "artifacts/dinov2_knn"),
            settings.get("device"),
            settings.get("model_version"),
        )
    if settings["backend"] != "dinov2_logreg":
        raise BackendUnavailable(f"Unsupported classifier backend: {settings['backend']}")
    if not settings.get("artifact_dir") or not settings.get("encoder_artifact_root"):
        raise BackendUnavailable("classifier.artifact_dir and encoder_artifact_root are required")
    image_size = tuple(settings.get("image_size", (256, 256)))
    if len(image_size) != 2:
        raise BackendUnavailable("classifier.image_size must be [height, width]")
    return DinoV2LogRegClassifier(
        settings.get("artifact_dir"),
        settings.get("encoder_artifact_root"),
        (int(image_size[0]), int(image_size[1])),
        settings.get("device"),
        settings.get("model_version", "dinov2-logreg"),
    )


def build_product_recognizer(settings: dict[str, Any]) -> ProductRecognizer:
    if settings.get("backend") == "gallery":
        gallery_path = settings.get("gallery_path") or settings.get("artifact_path")
        calibration_path = settings.get("calibration_path")
        if not gallery_path or not calibration_path:
            raise BackendUnavailable(
                "Gallery product recognizer requires gallery_path and calibration_path"
            )
        return DinoV2GalleryRecognizer(
            gallery_path,
            calibration_path,
            settings.get("encoder_artifact_root", "artifacts/dinov2_knn"),
            settings.get("device"),
        )
    raise BackendUnavailable(f"Unsupported product recognizer backend: {settings.get('backend')}")


def build_localizer(settings: dict[str, Any]) -> Localizer:
    if settings["backend"] == "patchcore_padim":
        return PatchCorePaDiMLocalizer(settings)
    if settings["backend"] != "anomalyclip":
        raise BackendUnavailable(f"Unsupported localizer backend: {settings['backend']}")
    calibration_path = settings.get("calibration_path")
    if calibration_path and Path(calibration_path).is_file():
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if "mask_threshold" in calibration:
            settings = dict(settings)
            settings["mask_threshold"] = float(calibration["mask_threshold"])
    return AnomalyClipLocalizer(settings)
