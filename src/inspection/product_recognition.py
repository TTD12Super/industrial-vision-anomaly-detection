"""DINOv2 gallery artifacts and validation-only open-set calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, select_device

from .config import ProductConfig
from .dataset import load_dataset_records
from .geometry import oriented_crop


GALLERY_ARTIFACT_TYPE = "dino_product_gallery"
GALLERY_SCHEMA_VERSION = "1.0"
CALIBRATION_ARTIFACT_TYPE = "dino_product_recognizer_calibration"
CALIBRATION_SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop_tensor(crop: np.ndarray, config: DINOv2Config) -> torch.Tensor:
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("Product recognizer crops must be RGB arrays")
    image = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize(
        (config.image_size[1], config.image_size[0]), Image.Resampling.LANCZOS
    )
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=pixels.dtype).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=pixels.dtype).view(3, 1, 1)
    return (pixels - mean) / std


def _encoder_config(metadata: dict[str, Any]) -> DINOv2Config:
    try:
        image_size = tuple(int(value) for value in metadata["image_size"])
        mean = tuple(float(value) for value in metadata["mean"])
        std = tuple(float(value) for value in metadata["std"])
        return DINOv2Config(
            model_name=str(metadata["model_name"]),
            model_revision=str(metadata["model_revision"]),
            patch_feature_key=str(metadata.get("patch_feature_key", "x_norm_patchtokens")),
            global_feature_key=str(metadata.get("global_feature_key", "x_norm_clstoken")),
            image_size=image_size,  # type: ignore[arg-type]
            mean=mean,  # type: ignore[arg-type]
            std=std,  # type: ignore[arg-type]
            l2_normalize_patches=bool(metadata.get("l2_normalize_patches", True)),
            feature_layers=tuple(int(value) for value in metadata.get("feature_layers", [])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Gallery encoder metadata is incomplete") from error


def load_gallery_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a CPU DINO gallery without loading the DINO model."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Product recognizer gallery is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("artifact_type") != GALLERY_ARTIFACT_TYPE:
        raise ValueError(f"Invalid product recognizer gallery artifact: {path}")
    if payload.get("schema_version") != GALLERY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported product recognizer gallery schema: {path}")
    embeddings = payload.get("embeddings")
    product_keys = payload.get("product_keys")
    encoder = payload.get("encoder")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] == 0
        or not torch.isfinite(embeddings).all().item()
        or not isinstance(product_keys, list)
        or len(product_keys) != embeddings.shape[0]
        or any(not isinstance(key, str) or not key for key in product_keys)
        or not isinstance(encoder, dict)
    ):
        raise ValueError(f"Product recognizer gallery payload is malformed: {path}")
    normalized = F.normalize(embeddings.float().cpu(), dim=1)
    if not torch.isfinite(normalized).all().item() or (normalized.norm(dim=1) == 0).any().item():
        raise ValueError(f"Product recognizer gallery contains zero or invalid embeddings: {path}")
    return {
        **payload,
        "embeddings": normalized,
        "product_keys": product_keys,
    }


def load_calibration_artifact(
    path: str | Path,
    gallery_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a calibrated threshold and reject stale or incomplete artifacts."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Product recognizer calibration is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid product recognizer calibration: {path}") from error
    if not isinstance(payload, dict) or payload.get("artifact_type") != CALIBRATION_ARTIFACT_TYPE:
        raise ValueError(f"Invalid product recognizer calibration: {path}")
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported product recognizer calibration schema: {path}")
    try:
        threshold = float(payload["known_threshold"])
        if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError
        if payload.get("score_type") != "cosine_similarity":
            raise ValueError
        if int(payload["known_validation_count"]) < 1 or int(payload["unknown_validation_count"]) < 1:
            raise ValueError
        if not isinstance(payload["gallery_sha256"], str) or not payload["gallery_sha256"]:
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Product recognizer calibration payload is malformed: {path}") from error
    if gallery_path is not None:
        gallery_path = Path(gallery_path)
        if not gallery_path.is_file() or payload["gallery_sha256"] != _sha256(gallery_path):
            raise ValueError("Product recognizer calibration does not match the gallery artifact")
        gallery = load_gallery_artifact(gallery_path)
        if len(set(gallery["product_keys"])) > 1:
            try:
                margin = float(payload["min_product_margin"])
                if not np.isfinite(margin) or not 0.0 <= margin <= 2.0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Multi-product calibration requires a valid min_product_margin") from error
    return payload


def gallery_artifact_ready(path: str | Path) -> bool:
    try:
        load_gallery_artifact(path)
    except (FileNotFoundError, OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError):
        return False
    return True


def calibration_artifact_ready(path: str | Path, gallery_path: str | Path | None = None) -> bool:
    try:
        load_calibration_artifact(path, gallery_path)
    except (FileNotFoundError, OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError):
        return False
    return True


def _records(
    configs: Sequence[ProductConfig],
    split: str,
    include_defects: bool = False,
) -> list[dict[str, Any]]:
    """Collect object crops; galleries stay GOOD-only, routing calibration does not."""

    records: list[dict[str, Any]] = []
    for config in configs:
        images, objects = load_dataset_records(config)
        image_by_id = {row["image_id"]: row for row in images}
        for object_row in objects:
            image = image_by_id[object_row["image_id"]]
            if image["split"] != split or (
                not include_defects and object_row.get("quality_label") != "GOOD"
            ):
                continue
            records.append(
                {
                    "product_key": config.product_key,
                    "image_path": config.dataset_path(Path(image["relative_path"])),
                    "corners": object_row["obb_xyxyxyxy"],
                    "object_id": object_row["object_id"],
                }
            )
    return records


def _manifest_provenance(config: ProductConfig, split: str, label: str) -> dict[str, Any]:
    """Record the exact canonical manifests used to build or calibrate an artifact."""

    image_manifest = config.dataset_path(config.image_manifest)
    object_manifest = config.dataset_path(config.object_manifest)
    return {
        "product_key": config.product_key,
        "split": split,
        "label": label,
        "image_manifest": str(image_manifest),
        "image_manifest_sha256": _sha256(image_manifest),
        "object_manifest": str(object_manifest),
        "object_manifest_sha256": _sha256(object_manifest),
        "split_policy": config.split_policy,
    }


def _encoder_for(
    config: ProductConfig,
    encoder: SharedDINOv2 | Any | None,
    device: str | None,
) -> SharedDINOv2 | Any:
    if encoder is not None:
        return encoder
    recognizer = config.models.get("product_recognizer", {})
    encoder_root = recognizer.get("encoder_artifact_root") or config.model("classifier").get(
        "encoder_artifact_root", "artifacts/dinov2_knn"
    )
    return SharedDINOv2(
        DINOv2Config(image_size=(256, 256)),
        artifact_root=config.project_path(encoder_root),
        device=device or str(select_device()),
    )


def _encode_records(
    records: Sequence[dict[str, Any]],
    encoder: Any,
    batch_size: int,
) -> torch.Tensor:
    if not records:
        raise ValueError("Product recognizer calibration/gallery records are empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoded: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        tensors = []
        for record in records[start : start + batch_size]:
            with Image.open(record["image_path"]) as source:
                crop = oriented_crop(np.asarray(source.convert("RGB")), record["corners"]).image
            tensors.append(_crop_tensor(crop, encoder.config))
        features = encoder.forward(torch.stack(tensors)).global_embedding.float().cpu()
        if features.ndim != 2 or not torch.isfinite(features).all().item():
            raise ValueError("DINOv2 produced invalid product gallery embeddings")
        encoded.append(F.normalize(features, dim=1))
    return torch.cat(encoded, dim=0)


def _nearest_scores(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
) -> torch.Tensor:
    _, product_scores = _product_score_matrix(query_embeddings, gallery_embeddings, None)
    return product_scores.max(dim=1).values


def _product_score_matrix(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    product_keys: Sequence[str] | None,
) -> tuple[list[str], torch.Tensor]:
    """Aggregate exemplar similarities to one score per product key."""

    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("Product recognizer embeddings must be two-dimensional")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1] or gallery_embeddings.shape[0] == 0:
        raise ValueError("Product recognizer embedding dimensions do not match")
    if product_keys is None:
        product_keys = [str(index) for index in range(gallery_embeddings.shape[0])]
    if len(product_keys) != gallery_embeddings.shape[0]:
        raise ValueError("Product recognizer product keys do not match gallery rows")
    query = F.normalize(query_embeddings.float().cpu(), dim=1)
    gallery = F.normalize(gallery_embeddings.float().cpu(), dim=1)
    exemplar_scores = query @ gallery.t()
    unique_keys = list(dict.fromkeys(product_keys))
    product_scores = torch.stack(
        [
            exemplar_scores[:, [index for index, key in enumerate(product_keys) if key == product_key]].max(dim=1).values
            for product_key in unique_keys
        ],
        dim=1,
    )
    return unique_keys, product_scores


def calibrate_gallery_threshold(
    gallery_embeddings: torch.Tensor,
    known_embeddings: torch.Tensor,
    unknown_embeddings: torch.Tensor,
    target_known_recall: float,
    known_product_keys: Sequence[str] | None = None,
    gallery_product_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Choose the highest observed cosine threshold meeting an explicit recall target."""

    if not 0.0 < target_known_recall <= 1.0:
        raise ValueError("target_known_recall must be in the interval (0, 1]")
    if known_embeddings.shape[0] == 0 or unknown_embeddings.shape[0] == 0:
        raise ValueError("Known and unknown validation examples are both required for calibration")
    known_scores = _nearest_scores(known_embeddings, gallery_embeddings)
    unknown_scores = _nearest_scores(unknown_embeddings, gallery_embeddings)
    candidates = torch.unique(
        torch.cat((torch.tensor([-1.0, 1.0]), known_scores, unknown_scores))
    ).sort().values
    eligible = [
        float(candidate)
        for candidate in candidates
        if float((known_scores >= candidate).float().mean()) >= target_known_recall
    ]
    if not eligible:
        raise ValueError("Target known-product recall is unattainable on calibration data")
    threshold = max(eligible)
    result = {
        "known_threshold": threshold,
        "known_validation_count": int(known_scores.numel()),
        "unknown_validation_count": int(unknown_scores.numel()),
        "known_validation_recall": float((known_scores >= threshold).float().mean()),
        "unknown_validation_rejection_rate": float((unknown_scores < threshold).float().mean()),
        "target_known_recall": target_known_recall,
        "score_type": "cosine_similarity",
        "calibration_protocol": "highest observed threshold meeting explicit known validation recall target",
    }
    if known_product_keys is not None or gallery_product_keys is not None:
        if known_product_keys is None or gallery_product_keys is None or len(known_product_keys) != known_embeddings.shape[0]:
            raise ValueError("Known and gallery product keys are required for product-margin calibration")
        product_keys, product_scores = _product_score_matrix(
            known_embeddings, gallery_embeddings, gallery_product_keys
        )
        if len(product_keys) > 1:
            order = torch.argsort(product_scores, dim=1, descending=True)
            top_scores = product_scores.gather(1, order[:, :1]).squeeze(1)
            second_scores = product_scores.gather(1, order[:, 1:2]).squeeze(1)
            margins = top_scores - second_scores
            top_products = [product_keys[int(index)] for index in order[:, 0]]
            expected = list(known_product_keys)
            correct = torch.tensor(
                [top == truth for top, truth in zip(top_products, expected)], dtype=torch.bool
            )
            candidates = torch.unique(torch.cat((torch.tensor([0.0, 2.0]), margins))).sort().values
            eligible = [
                float(candidate)
                for candidate in candidates
                if float(
                    (correct & (top_scores >= threshold) & (margins >= candidate)).float().mean()
                )
                >= target_known_recall
            ]
            if not eligible:
                raise ValueError("Target known-product recall is unattainable with product margin calibration")
            min_product_margin = max(eligible)
            result.update(
                {
                    "min_product_margin": min_product_margin,
                    "known_product_validation_recall": float(
                        (correct & (top_scores >= threshold) & (margins >= min_product_margin)).float().mean()
                    ),
                    "product_margin_calibration_protocol": "highest observed top-1/top-2 product score margin meeting explicit known validation recall target",
                }
            )
    return result


def build_product_gallery(
    configs: Sequence[ProductConfig],
    output_path: str | Path,
    batch_size: int = 16,
    device: str | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Build a train/GOOD-only gallery from one or more canonical product manifests."""

    if not configs:
        raise ValueError("At least one product config is required")
    if len({config.product_key for config in configs}) != len(configs):
        raise ValueError("Gallery configs must have unique product_key values")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite product gallery: {output_path}")
    records = _records(configs, "train")
    if not records:
        raise ValueError("No train/GOOD objects are available for the product gallery")
    active_encoder = _encoder_for(configs[0], encoder, device)
    embeddings = _encode_records(records, active_encoder, batch_size)
    metadata = active_encoder.metadata()
    if embeddings.shape[1] != int(metadata["embedding_dimension"]):
        raise ValueError("DINOv2 embedding dimension does not match encoder metadata")
    payload = {
        "artifact_type": GALLERY_ARTIFACT_TYPE,
        "schema_version": GALLERY_SCHEMA_VERSION,
        "model_version": "dinov2-gallery",
        "product_keys": [record["product_key"] for record in records],
        "object_ids": [record["object_id"] for record in records],
        "embeddings": embeddings,
        "encoder": metadata,
        "source_split": "train",
        "source_label": "GOOD",
        "source_manifests": [_manifest_provenance(config, "train", "GOOD") for config in configs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "artifact_type": GALLERY_ARTIFACT_TYPE,
        "schema_version": GALLERY_SCHEMA_VERSION,
        "gallery_path": str(output_path.resolve()),
        "product_keys": sorted(set(payload["product_keys"])),
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "source_split": "train",
        "source_label": "GOOD",
    }


def calibrate_product_recognizer(
    known_configs: Sequence[ProductConfig],
    unknown_configs: Sequence[ProductConfig],
    gallery_path: str | Path,
    output_path: str | Path,
    target_known_recall: float,
    batch_size: int = 16,
    device: str | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Calibrate only from validation crops, requiring an explicit unknown set."""

    if not known_configs or not unknown_configs:
        raise ValueError("Known and unknown product configs are both required for calibration")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite product recognizer calibration: {output_path}")
    gallery_path = Path(gallery_path)
    gallery = load_gallery_artifact(gallery_path)
    gallery_keys = set(gallery["product_keys"])
    if any(config.product_key not in gallery_keys for config in known_configs):
        raise ValueError("Known calibration configs must be represented in the gallery")
    if any(config.product_key in gallery_keys for config in unknown_configs):
        raise ValueError("Unknown calibration configs must not be represented in the gallery")
    known_records = _records(known_configs, "validation", include_defects=True)
    unknown_records = _records(unknown_configs, "validation")
    if not known_records or not unknown_records:
        raise ValueError("Known and unknown validation sets must both contain object crops")
    active_encoder = _encoder_for(known_configs[0], encoder, device)
    expected_metadata = gallery["encoder"]
    actual_metadata = active_encoder.metadata()
    if actual_metadata.get("embedding_dimension") != expected_metadata.get("embedding_dimension"):
        raise ValueError("Calibration encoder does not match gallery embedding dimension")
    known_embeddings = _encode_records(known_records, active_encoder, batch_size)
    unknown_embeddings = _encode_records(unknown_records, active_encoder, batch_size)
    calibration = calibrate_gallery_threshold(
        gallery["embeddings"],
        known_embeddings,
        unknown_embeddings,
        target_known_recall,
        [record["product_key"] for record in known_records],
        gallery["product_keys"],
    )
    calibration.update(
        {
            "artifact_type": CALIBRATION_ARTIFACT_TYPE,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "gallery_sha256": _sha256(gallery_path),
            "known_product_keys": sorted({record["product_key"] for record in known_records}),
            "unknown_product_keys": sorted({record["product_key"] for record in unknown_records}),
            "calibration_split": "validation",
            "unknown_examples_used": True,
            "known_source_manifests": [_manifest_provenance(config, "validation", "GOOD") for config in known_configs],
            "unknown_source_manifests": [_manifest_provenance(config, "validation", "GOOD") for config in unknown_configs],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return {"calibration_path": str(output_path.resolve()), **calibration}


__all__ = [
    "CALIBRATION_ARTIFACT_TYPE",
    "CALIBRATION_SCHEMA_VERSION",
    "GALLERY_ARTIFACT_TYPE",
    "GALLERY_SCHEMA_VERSION",
    "build_product_gallery",
    "calibrate_gallery_threshold",
    "calibrate_product_recognizer",
    "calibration_artifact_ready",
    "gallery_artifact_ready",
    "load_calibration_artifact",
    "load_gallery_artifact",
]
