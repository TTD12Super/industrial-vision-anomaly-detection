"""Small JSON-backed configuration for product-specific experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class EggDemoConfig:
    """Paths and model settings; processing code does not depend on egg names."""

    detection_dataset: Path = Path("datasets/raw/egg_detection")
    quality_archive: Path = Path("datasets/raw/egg_quality/Original_Images_Eggs.zip")
    quality_processed: Path = Path("datasets/processed/egg_quality")
    yolo_artifact: Path = Path("artifacts/egg_demo/yolo")
    padim_artifact: Path = Path("artifacts/egg_demo/padim")
    encoder_artifact: Path = Path("artifacts/production/mvtec_ad")
    production_artifact: Path = Path("artifacts/production/mvtec_ad_egg")
    image_size: int = 256
    feature_layers: tuple[int, ...] = (5, 8, 11)
    neighborhood_size: int = 3
    padim_reduced_dim: int = 100
    covariance_regularization: float = 0.01
    target_acceptance_rate: float = 0.99
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.5
    crop_margin: float = 0.05
    seed: int = 42
    categories: tuple[str, ...] = field(default_factory=lambda: ("white-egg", "brown-egg"))

    @classmethod
    def from_json(cls, path: str | Path) -> "EggDemoConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("feature_layers", "categories"):
            if key in values:
                values[key] = tuple(values[key])
        for key in (
            "detection_dataset",
            "quality_archive",
            "quality_processed",
            "yolo_artifact",
            "padim_artifact",
            "encoder_artifact",
            "production_artifact",
        ):
            if key in values:
                values[key] = Path(values[key])
        return cls(**values)

    def to_json(self, path: str | Path) -> None:
        values = asdict(self)
        for key in values:
            if isinstance(values[key], Path):
                values[key] = str(values[key])
            elif isinstance(values[key], tuple):
                values[key] = list(values[key])
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
