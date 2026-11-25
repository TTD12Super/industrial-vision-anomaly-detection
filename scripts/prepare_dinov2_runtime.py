"""Prepare a local DINOv2 torch.hub cache without reading iCloud placeholders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHECKPOINT_NAME = "dinov2_vitb14_pretrain.pth"
REPOSITORY_PREFIX = "facebookresearch_dinov2_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialized(path: Path) -> bool:
    stat = path.stat()
    return stat.st_size > 0 and getattr(stat, "st_blocks", 1) > 0


def _link(source: Path, destination: Path, is_directory: bool) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace runtime link: {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite runtime path: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=is_directory)


def prepare_runtime(source_root: str | Path, destination_root: str | Path) -> dict[str, object]:
    """Create an idempotent runtime view with a materialized trust list."""

    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    source_hub = source_root / "torch_hub"
    checkpoint = source_hub / "checkpoints" / CHECKPOINT_NAME
    if not checkpoint.is_file() or not _materialized(checkpoint):
        raise FileNotFoundError(f"Materialized DINOv2 checkpoint is required: {checkpoint}")
    repositories = sorted(
        path for path in source_hub.iterdir() if path.is_dir() and path.name.startswith(REPOSITORY_PREFIX)
    )
    if len(repositories) != 1:
        raise ValueError(f"Expected exactly one cached DINOv2 repository under {source_hub}, found {repositories}")
    repository = repositories[0]
    required_code = (
        repository / "hubconf.py",
        repository / "dinov2/models/vision_transformer.py",
        repository / "dinov2/layers/attention.py",
    )
    if any(not path.is_file() or not _materialized(path) for path in required_code):
        raise FileNotFoundError(f"DINOv2 repository contains an unmaterialized required file: {repository}")

    destination_hub = destination_root / "torch_hub"
    destination_hub.mkdir(parents=True, exist_ok=True)
    trusted_list = destination_hub / "trusted_list"
    if trusted_list.exists() and not trusted_list.is_symlink():
        current_trust = trusted_list.read_text(encoding="utf-8")
        if current_trust not in {"", "facebookresearch\n"}:
            raise FileExistsError(f"Refusing to overwrite runtime trust list: {trusted_list}")
        if current_trust != "facebookresearch\n":
            trusted_list.write_text("facebookresearch\n", encoding="utf-8")
    else:
        trusted_list.write_text("facebookresearch\n", encoding="utf-8")
    _link(repository, destination_hub / repository.name, is_directory=True)
    _link(source_hub / "checkpoints", destination_hub / "checkpoints", is_directory=True)
    return {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "repository": repository.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "trusted_list": str(trusted_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("artifacts/dinov2_knn"))
    parser.add_argument("--destination-root", type=Path, default=Path("artifacts/runtime/dinov2_clean"))
    args = parser.parse_args()
    print(json.dumps(prepare_runtime(args.source_root, args.destination_root), indent=2))


if __name__ == "__main__":
    main()
