"""Product configuration loading with explicit, portable paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a product configuration cannot support the common pipeline."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _relative_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{name} must stay within the configured dataset root")
    return path


@dataclass(frozen=True)
class ProductConfig:
    """The small configuration surface shared by training and inference."""

    source_path: Path
    schema_version: int
    product_key: str
    dataset_root: Path
    image_manifest: Path
    object_manifest: Path
    split_policy: str
    models: dict[str, dict[str, Any]]
    policy: dict[str, Any]
    outputs: dict[str, Any]

    @property
    def config_dir(self) -> Path:
        return self.source_path.parent

    @property
    def project_root(self) -> Path:
        return self.source_path.parents[2]

    def model(self, name: str) -> dict[str, Any]:
        try:
            return self.models[name]
        except KeyError as error:
            raise ConfigError(f"models.{name} is required") from error

    def dataset_path(self, relative_path: Path) -> Path:
        return self.dataset_root / relative_path

    def project_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_product_config(path: str | Path) -> ProductConfig:
    """Load and validate a product YAML without embedding product-specific logic."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigError(f"Product config does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _mapping(raw, "product config")
    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    product_key = raw.get("product_key")
    if not isinstance(product_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", product_key):
        raise ConfigError("product_key must contain only letters, numbers, '_' or '-'")

    dataset = _mapping(raw.get("dataset"), "dataset")
    dataset_root_value = dataset.get("root")
    if not isinstance(dataset_root_value, str) or not dataset_root_value.strip():
        raise ConfigError("dataset.root must be a non-empty path")
    dataset_root = Path(dataset_root_value)
    if not dataset_root.is_absolute():
        dataset_root = (source_path.parents[2] / dataset_root).resolve()
    image_manifest = _relative_path(dataset.get("image_manifest"), "dataset.image_manifest")
    object_manifest = _relative_path(dataset.get("object_manifest"), "dataset.object_manifest")
    split_policy = dataset.get("split_policy")
    if not isinstance(split_policy, str) or not split_policy:
        raise ConfigError("dataset.split_policy is required")

    models = _mapping(raw.get("models"), "models")
    normalized_models: dict[str, dict[str, Any]] = {}
    for name in ("detector", "classifier", "localizer", "reporter"):
        model = _mapping(models.get(name), f"models.{name}")
        backend = model.get("backend")
        if not isinstance(backend, str) or not backend:
            raise ConfigError(f"models.{name}.backend is required")
        normalized_models[name] = model
    if "product_recognizer" in models:
        recognizer = _mapping(models.get("product_recognizer"), "models.product_recognizer")
        backend = recognizer.get("backend")
        if not isinstance(backend, str) or not backend:
            raise ConfigError("models.product_recognizer.backend is required")
        normalized_models["product_recognizer"] = recognizer

    policy = _mapping(raw.get("policy"), "policy")
    if policy.get("quality_labels") != ["GOOD", "DEFECT"]:
        raise ConfigError("policy.quality_labels must be exactly [GOOD, DEFECT]")
    outputs = _mapping(raw.get("outputs"), "outputs")
    return ProductConfig(
        source_path=source_path,
        schema_version=1,
        product_key=product_key,
        dataset_root=dataset_root,
        image_manifest=image_manifest,
        object_manifest=object_manifest,
        split_policy=split_policy,
        models=normalized_models,
        policy=policy,
        outputs=outputs,
    )
