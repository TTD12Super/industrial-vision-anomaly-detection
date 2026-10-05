"""Versioned paths for new MVTec system artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Stable, ignored locations for one artifact version."""

    root: Path = Path("artifacts/mvtec_system")
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.version or Path(self.version).name != self.version:
            raise ValueError("version must be a non-empty directory name.")

    @property
    def version_root(self) -> Path:
        return self.root / self.version

    @property
    def classifier(self) -> Path:
        return self.version_root / "classifier"

    @property
    def embeddings(self) -> Path:
        return self.version_root / "embeddings"

    @property
    def feature_banks(self) -> Path:
        return self.version_root / "feature_banks"

    @property
    def thresholds(self) -> Path:
        return self.version_root / "thresholds"

    @property
    def evaluation(self) -> Path:
        return self.version_root / "evaluation"

    @property
    def config(self) -> Path:
        return self.version_root / "config.json"

    def directories(self) -> tuple[Path, ...]:
        """Return directories that hold versioned artifacts."""

        return (
            self.classifier,
            self.embeddings,
            self.feature_banks,
            self.thresholds,
            self.evaluation,
        )

    def create(self) -> "ArtifactLayout":
        """Create empty versioned directories without touching old artifacts."""

        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def path_for(self, kind: str, filename: str) -> Path:
        """Return a safe path for one artifact file in a known section."""

        sections = {
            "classifier": self.classifier,
            "embeddings": self.embeddings,
            "feature_banks": self.feature_banks,
            "thresholds": self.thresholds,
            "evaluation": self.evaluation,
        }
        if kind not in sections:
            raise ValueError(f"Unknown artifact section: {kind}")
        path = (sections[kind] / filename).resolve()
        if not path.is_relative_to(sections[kind].resolve()):
            raise ValueError("Artifact filename would leave its versioned section.")
        return path


def make_artifact_layout(
    root: str | Path = "artifacts/mvtec_system",
    version: str = "v1",
) -> ArtifactLayout:
    """Build a versioned layout object without creating files."""

    return ArtifactLayout(Path(root), version)
