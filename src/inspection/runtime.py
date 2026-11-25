"""Product registry and runtime construction, kept outside model implementations."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from .backends import BackendUnavailable, build_classifier, build_detector, build_localizer, build_product_recognizer
from .config import ProductConfig, load_product_config
from .pipeline import InspectionPipeline
from .policy import DecisionPolicy
from .reporting import AsyncReportManager, LocalVLMReporter, MockReporter, UnavailableCloudReporter


def gallery_artifact_ready(*args: Any, **kwargs: Any) -> bool:
    if kwargs or len(args) != 1:
        raise TypeError("gallery_artifact_ready expects one path")
    return Path(args[0]).is_file()


def calibration_artifact_ready(*args: Any, **kwargs: Any) -> bool:
    if kwargs or not 1 <= len(args) <= 2:
        raise TypeError("calibration_artifact_ready expects a path and optional gallery path")
    path = Path(args[0])
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(payload["known_threshold"])
        gallery_sha256 = str(payload["gallery_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0 or not gallery_sha256:
        return False
    if len(args) == 2:
        gallery_path = Path(args[1])
        if not gallery_path.is_file():
            return False
        digest = hashlib.sha256(gallery_path.read_bytes()).hexdigest()
        if digest != gallery_sha256:
            return False
    return True


def _path_settings(config: ProductConfig, settings: dict[str, Any]) -> dict[str, Any]:
    value = dict(settings)
    for key in ("weights", "artifact_dir", "encoder_artifact_root", "checkpoint", "runner", "repository", "clip_cache", "calibration_path", "gallery_path", "artifact_path", "patch_memory_artifact_path", "patchcore_artifact_dir", "padim_artifact_dir", "relation_prototype_path", "relation_calibration_path", "dino_artifact_root", "supervised_refiner_path", "supervised_refiner_calibration_path"):
        if value.get(key):
            value[key] = str(config.project_path(value[key]))
    return value


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _configured_path(config: ProductConfig, value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    return config.project_path(value)


def _configured_file(config: ProductConfig, value: Any) -> bool:
    path = _configured_path(config, value)
    return path is not None and path.is_file()


def _configured_directory(config: ProductConfig, value: Any) -> bool:
    path = _configured_path(config, value)
    return path is not None and path.is_dir()


def _finite_number(payload: dict[str, Any] | None, key: str) -> bool:
    if payload is None:
        return False
    try:
        return math.isfinite(float(payload[key]))
    except (KeyError, TypeError, ValueError):
        return False


def _detector_settings(config: ProductConfig) -> dict[str, Any]:
    settings = _path_settings(config, config.model("detector"))
    calibration_value = settings.get("calibration_path")
    if not isinstance(calibration_value, str) or not calibration_value:
        raise BackendUnavailable("detector.calibration_path is required for inspection")
    calibration_path = config.project_path(calibration_value)
    if not calibration_path.is_file():
        raise BackendUnavailable(f"Detector calibration is missing: {calibration_path}")
    calibration = _json_object(calibration_path)
    if calibration is None:
        raise BackendUnavailable(f"Invalid detector calibration: {calibration_path}")
    try:
        confidence = float(calibration["confidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise BackendUnavailable(f"Invalid detector calibration: {calibration_path}") from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise BackendUnavailable(f"Invalid detector confidence in calibration: {calibration_path}")
    settings["confidence"] = confidence
    return settings


def load_policy(config: ProductConfig) -> DecisionPolicy:
    calibration_value = config.policy.get("calibration_path")
    if not isinstance(calibration_value, str):
        raise BackendUnavailable("policy.calibration_path is required for inspection")
    path = config.project_path(calibration_value)
    if not path.is_file():
        raise BackendUnavailable(f"Calibrated policy is missing: {path}")
    calibration = _json_object(path)
    if calibration is None:
        raise BackendUnavailable(f"Invalid calibration policy: {path}")
    try:
        return DecisionPolicy(float(calibration["defect_threshold"]), float(calibration["review_low_threshold"]))
    except (KeyError, TypeError, ValueError) as error:
        raise BackendUnavailable(f"Invalid calibration policy: {path}") from error


def build_pipeline(config: ProductConfig, runs_root: str | Path | None = None) -> InspectionPipeline:
    policy = load_policy(config)
    readiness = product_readiness(config)
    if not readiness["ready"]:
        missing = ", ".join(readiness["missing_components"])
        raise BackendUnavailable(f"Product {config.product_key!r} is not ready: {missing}")
    detector = build_detector(_detector_settings(config))
    classifier = build_classifier(_path_settings(config, config.model("classifier")))
    localizer = build_localizer(_path_settings(config, config.model("localizer")))
    recognizer_settings = config.models.get("product_recognizer")
    product_recognizer = build_product_recognizer(_path_settings(config, recognizer_settings)) if recognizer_settings else None
    reporter_settings = config.model("reporter")
    backend = reporter_settings["backend"]
    async_enabled = bool(reporter_settings.get("async_enabled", True))
    if backend == "mock":
        reporter = AsyncReportManager(MockReporter(), async_enabled=async_enabled)
    elif backend == "local_vlm":
        base_url = reporter_settings.get("base_url")
        model = reporter_settings.get("model")
        if not isinstance(base_url, str) or not isinstance(model, str):
            raise BackendUnavailable("local_vlm requires reporter.base_url and reporter.model")
        reporter = AsyncReportManager(
            LocalVLMReporter(
                base_url,
                model,
                float(reporter_settings.get("timeout_seconds", 120.0)),
                int(reporter_settings.get("max_tokens", 512)),
            ),
            workers=int(reporter_settings.get("workers", 1)),
                async_enabled=async_enabled,
        )
    elif backend == "cloud_vlm":
        reporter = AsyncReportManager(UnavailableCloudReporter(), async_enabled=async_enabled)
    else:
        raise BackendUnavailable(f"Unsupported reporter backend: {backend}")
    destination = Path(runs_root) if runs_root is not None else config.project_path("runs")
    return InspectionPipeline(
        product_key=config.product_key,
        model_bundle_version=str(config.policy.get("model_bundle_version", "unversioned")),
        detector=detector,
        classifier=classifier,
        policy=policy,
        localizer=localizer,
        runs_root=destination,
        reporter=reporter,
        product_recognizer=product_recognizer,
    )


class ProductRegistry:
    """Discover only YAML configuration; products never require core if/else branches."""

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)

    def configs(self) -> dict[str, ProductConfig]:
        products: dict[str, ProductConfig] = {}
        for path in sorted(self.config_dir.glob("*.yaml")):
            config = load_product_config(path)
            if config.product_key in products:
                raise ValueError(f"Duplicate product key: {config.product_key}")
            products[config.product_key] = config
        return products

    def get(self, product_key: str) -> ProductConfig:
        try:
            return self.configs()[product_key]
        except KeyError as error:
            raise KeyError(f"Unknown product: {product_key}") from error


def product_readiness(config: ProductConfig) -> dict[str, Any]:
    """Check every configured runtime prerequisite without loading model weights."""

    detector = config.model("detector")
    classifier = config.model("classifier")
    localizer = config.model("localizer")
    reporter = config.model("reporter")
    detector_calibration = _json_object(config.project_path(detector["calibration_path"])) if isinstance(detector.get("calibration_path"), str) else None
    detector_confidence = detector_calibration.get("confidence") if detector_calibration else None
    detector_calibration_ready = (
        detector_calibration is not None
        and isinstance(detector_confidence, (int, float))
        and math.isfinite(float(detector_confidence))
        and 0.0 <= float(detector_confidence) <= 1.0
    )
    policy_calibration = _json_object(config.project_path(config.policy["calibration_path"])) if isinstance(config.policy.get("calibration_path"), str) else None
    policy_calibration_ready = False
    if policy_calibration is not None:
        try:
            DecisionPolicy(float(policy_calibration["defect_threshold"]), float(policy_calibration["review_low_threshold"]))
            policy_calibration_ready = True
        except (KeyError, TypeError, ValueError):
            pass
    classifier_dir = _configured_path(config, classifier.get("artifact_dir"))
    patch_memory_artifact = _configured_path(
        config,
        classifier.get("artifact_path") or classifier.get("patch_memory_artifact_path"),
    )
    classifier_mapping = _json_object(classifier_dir / "classifier" / "class_mapping.json") if classifier_dir is not None else None
    classifier_mapping_ready = classifier_mapping is not None and classifier_mapping.get("category_to_index") == {"GOOD": 0, "DEFECT": 1}
    localizer_calibration = _json_object(config.project_path(localizer["calibration_path"])) if isinstance(localizer.get("calibration_path"), str) else None
    localizer_calibration_ready = not localizer.get("calibration_path") or _finite_number(localizer_calibration, "mask_threshold")
    patchcore_ready = _configured_file(config, localizer.get("patchcore_artifact_dir", "") and Path(localizer["patchcore_artifact_dir"]) / "coreset.pt")
    padim_ready = _configured_file(config, localizer.get("padim_artifact_dir", "") and Path(localizer["padim_artifact_dir"]) / "statistics.pt")
    refiner_calibration = _json_object(config.project_path(localizer["supervised_refiner_calibration_path"])) if isinstance(localizer.get("supervised_refiner_calibration_path"), str) else None
    refiner_ready = not localizer.get("supervised_refiner_enabled") or (
        _configured_file(config, localizer.get("supervised_refiner_path"))
        and _configured_file(config, localizer.get("supervised_refiner_calibration_path"))
        and _configured_directory(config, localizer.get("dino_artifact_root"))
        and _finite_number(refiner_calibration, "mask_threshold")
    )
    relation_calibration = _json_object(config.project_path(localizer["relation_calibration_path"])) if isinstance(localizer.get("relation_calibration_path"), str) else None
    relation_ready = not localizer.get("relation_fusion_enabled") or (
        _configured_file(config, localizer.get("relation_prototype_path"))
        and _configured_file(config, localizer.get("relation_calibration_path"))
        and _finite_number(relation_calibration, "mask_threshold")
    )
    components = {
        "detector_backend": detector.get("backend") == "yolo_obb",
        "detector_weights": _configured_file(config, detector.get("weights")),
        "detector_calibration": detector_calibration_ready,
        "classifier_backend": classifier.get("backend") in {"dinov2_logreg", "dinov2_patch_memory"},
        "classifier_weights": (
            classifier_dir is not None and (classifier_dir / "classifier" / "classifier.pt").is_file()
        ) or (patch_memory_artifact is not None and patch_memory_artifact.is_file()),
        "classifier_mapping": (
            classifier_mapping_ready if classifier.get("backend") == "dinov2_logreg" else patch_memory_artifact is not None and patch_memory_artifact.is_file()
        ),
        "classifier_encoder": _configured_directory(config, classifier.get("encoder_artifact_root")),
        "classifier_calibration": policy_calibration_ready,
        "localizer_backend": localizer.get("backend") in {"anomalyclip", "patchcore_padim"},
        "localizer_runner": (
            _configured_file(config, localizer.get("runner"))
            if localizer.get("backend") == "anomalyclip"
            else patchcore_ready
        ),
        "localizer_checkpoint": (
            _configured_file(config, localizer.get("checkpoint"))
            if localizer.get("backend") == "anomalyclip"
            else padim_ready
        ),
        "localizer_repository": (
            _configured_directory(config, localizer.get("repository"))
            if localizer.get("backend") == "anomalyclip"
            else _configured_directory(config, localizer.get("encoder_artifact_root"))
        ),
        "localizer_clip_cache": (
            not localizer.get("clip_cache") or _configured_directory(config, localizer.get("clip_cache"))
            if localizer.get("backend") == "anomalyclip"
            else True
        ),
        "localizer_calibration": localizer_calibration_ready,
        "localizer_refiner": refiner_ready if localizer.get("backend") == "anomalyclip" else True,
        "localizer_relation": relation_ready if localizer.get("backend") == "anomalyclip" else True,
        "reporter": (
            reporter.get("backend") == "mock"
            or (
                reporter.get("backend") == "local_vlm"
                and isinstance(reporter.get("base_url"), str)
                and reporter.get("base_url", "").startswith(("http://", "https://"))
                and isinstance(reporter.get("model"), str)
                and bool(reporter.get("model", "").strip())
            )
        ),
    }
    recognizer_settings = config.models.get("product_recognizer")
    if recognizer_settings is not None:
        gallery_value = recognizer_settings.get("gallery_path") or recognizer_settings.get("artifact_path")
        calibration_value = recognizer_settings.get("calibration_path")
        encoder_value = recognizer_settings.get("encoder_artifact_root")
        gallery_path = _configured_path(config, gallery_value)
        calibration_path = _configured_path(config, calibration_value)
        gallery_ready = gallery_path is not None and gallery_artifact_ready(gallery_path)
        calibration_ready = (
            calibration_path is not None
            and calibration_artifact_ready(calibration_path, gallery_path if gallery_ready else None)
        )
        encoder_ready = encoder_value is None or _configured_directory(config, encoder_value)
        components.update(
            {
                "product_recognizer_backend_implemented": recognizer_settings.get("backend") == "gallery",
                "product_recognizer_artifact": gallery_ready,
                "product_recognizer_calibration": calibration_ready,
                "product_recognizer_encoder": encoder_ready,
            }
        )
    missing_components = [name for name, ready in components.items() if not ready]
    return {
        "product_key": config.product_key,
        "routing_mode": "product_recognizer" if recognizer_settings is not None else "closed_world",
        "ready": not missing_components,
        "components": components,
        "missing_components": missing_components,
    }
