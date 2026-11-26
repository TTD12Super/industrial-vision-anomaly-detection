#!/usr/bin/env python3
"""Leakage-guarded original Dinomaly ViT-Small vs category-routed PatchCore pilot.

This file intentionally imports no third-party packages at module load time so
``--self-check`` can run without importing the locally broken Torch runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPSTREAM_COMMIT = "1f252be03a918789b19848f0ca37166e7c28dada"
EXPECTED_MANIFEST_SHA256 = "fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432"
EXPECTED_DATASET_INVENTORY_SHA256 = "76ecec7e1a11c95e244f345e7fe4c542cc9778fec5fa839e1f709229cc9df6eb"
EXPECTED_TEST_INVENTORY_SHA256 = "a88785e62f357832c19188baddb3df95ca252e1257dc6050c26f728c23ac32be"
EXPECTED_TRAIN_VALIDATION_INVENTORY_SHA256 = "ec86b00363eabe9ae981ef39015a92a00207c99c02ec4bcff11ce728c7007c63"
DEFAULT_DECODED_PIXEL_AUDIT = Path("reports/experiments/goal_15_category_heatmap_methods_320/decoded_pixel_split_audit.json")
EXPECTED_TRAIN_ROWS = 2540
EXPECTED_VALIDATION_ROWS = 1089
EXPECTED_TEST_IMAGES = 1725
EXPECTED_DATASET_IMAGES = 5354
CATEGORY_NAMES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)
IMAGE_SIZE = 320
PADDED_SIZE = 322
SEED = 42
TRAIN_ITERATIONS = 5000
OFFICIAL_TOTAL_ITERATIONS = 10000
WARMUP_ITERATIONS = 100
OFFICIAL_BASE_LR = 2e-3
OFFICIAL_FINAL_LR = 2e-4
BATCH_SIZE = 8
CALIBRATION_QUANTILE = 0.99
PATCHCORE_LAYERS = (5, 8, 11)
PATCHCORE_RATIO = 0.10
PATCHCORE_MAX_POINTS = 50_000
PATCHCORE_NEIGHBORHOOD = 3

UPSTREAM_SOURCE_FILES = (
    "dinomaly_mvtec_uni.py",
    "models/uad.py",
    "models/vit_encoder.py",
    "models/vision_transformer.py",
    "dinov1/utils.py",
    "optimizers/StableAdamW.py",
    "utils.py",
)
PROJECT_SOURCE_FILES = (
    "src/mvtec_system/core/dinov2.py",
    "src/mvtec_system/anomaly/detectors.py",
    "src/mvtec_system/anomaly/multi_layer.py",
    "src/mvtec_system/anomaly/knn.py",
    "src/mvtec_system/anomaly/maps.py",
    "src/mvtec_system/classification/classifier.py",
    "src/mvtec_system/classification/router.py",
)


class PilotError(RuntimeError):
    """Raised when an experiment precondition is not met."""


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    category: str
    split: str


@dataclass(frozen=True)
class TestSample:
    image_path: Path
    category: str
    defect_type: str
    mask_path: Path | None


class RunGate:
    """State guard preventing official-test access before both fits/calibration."""

    def __init__(self) -> None:
        self.dinomaly_fitted = False
        self.patchcore_fitted = False
        self.checkpoints_saved = False
        self.thresholds_calibrated = False
        self.test_inference_started = False

    def open_test(self, factory: Callable[[], Any]) -> Any:
        if not (
            self.dinomaly_fitted
            and self.patchcore_fitted
            and self.checkpoints_saved
            and self.thresholds_calibrated
        ):
            raise PilotError("Official test access is blocked until both fits, checkpoints, and validation calibration finish.")
        self.test_inference_started = True
        return factory()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_parts(value: str) -> tuple[str, ...]:
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise PilotError(f"Manifest image_path must be relative and remain inside the data root: {value}")
    return pure.parts


def validate_training_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Accept only the pinned manifest's train/good rows for fitting."""

    selected = list(rows)
    if not selected:
        raise PilotError("No training rows were supplied.")
    for row in selected:
        parts = _relative_parts(row.image_path)
        if row.split != "train" or len(parts) < 4 or parts[0] != row.category:
            raise PilotError(f"Non-train or category-mismatched row cannot enter fit: {row}")
        if parts[1:3] != ("train", "good") or any(part.lower() in {"test", "ground_truth"} for part in parts):
            raise PilotError(f"Only train/good images may enter fit: {row.image_path}")
    return selected


def validate_manifest(manifest_path: Path, data_root: Path) -> tuple[list[ManifestRow], list[ManifestRow], str]:
    if not manifest_path.is_file():
        raise PilotError(f"Manifest not found: {manifest_path}")
    digest = sha256_file(manifest_path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise PilotError(
            f"Manifest SHA-256 mismatch: expected {EXPECTED_MANIFEST_SHA256}, got {digest}."
        )
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_path", "category", "split"]:
            raise PilotError(f"Unexpected manifest columns: {reader.fieldnames}")
        rows = [ManifestRow(row["image_path"], row["category"], row["split"]) for row in reader]

    if len(rows) != EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS:
        raise PilotError(f"Expected 3629 manifest rows, got {len(rows)}.")
    categories = {row.category for row in rows}
    if categories != set(CATEGORY_NAMES):
        raise PilotError(f"Manifest category set differs from the pinned 15 categories: {sorted(categories)}")
    train_rows = [row for row in rows if row.split == "train"]
    validation_rows = [row for row in rows if row.split == "validation"]
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(validation_rows) != EXPECTED_VALIDATION_ROWS:
        raise PilotError(
            f"Expected train/validation counts {EXPECTED_TRAIN_ROWS}/{EXPECTED_VALIDATION_ROWS}, "
            f"got {len(train_rows)}/{len(validation_rows)}."
        )
    if len({row.image_path for row in rows}) != len(rows):
        raise PilotError("Manifest contains duplicate image paths.")
    for row in rows:
        parts = _relative_parts(row.image_path)
        if row.category != parts[0] or parts[1:3] != ("train", "good"):
            raise PilotError(f"Manifest row is not a category-matched train/good image: {row}")
        path = data_root.joinpath(*parts)
        if not path.is_file():
            raise PilotError(f"Manifest image is missing: {path}")
    validate_training_rows(train_rows)
    return train_rows, validation_rows, digest


def _validate_decoded_audit_identity(audit_path: Path, manifest_digest: str) -> dict[str, Any]:
    if not audit_path.is_file():
        raise PilotError(f"Saved decoded-pixel audit is missing: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotError(f"Saved decoded-pixel audit is unreadable: {audit_path}: {error}") from error
    expected = {
        "protocol": "SHA-256 over decoded RGBA pixels and dimensions",
        "manifest_sha256": manifest_digest,
        "dataset_inventory_sha256": EXPECTED_DATASET_INVENTORY_SHA256,
        "test_image_inventory_sha256": EXPECTED_TEST_INVENTORY_SHA256,
        "image_count": EXPECTED_DATASET_IMAGES,
        "files_by_split": {"test": EXPECTED_TEST_IMAGES, "train": EXPECTED_TRAIN_ROWS, "validation": EXPECTED_VALIDATION_ROWS},
        "unique_decoded_pixel_hashes": EXPECTED_DATASET_IMAGES,
        "decoded_pixel_hash_failures": 0,
        "cross_split_identical_pixel_pair_count": 0,
    }
    mismatches = [key for key, value in expected.items() if audit.get(key) != value]
    if mismatches:
        raise PilotError("Saved decoded-pixel audit identity mismatch: " + ", ".join(mismatches))
    return {
        "path": str(audit_path.resolve()),
        "file_sha256": sha256_file(audit_path),
        "protocol": audit["protocol"],
        "audit_date": audit.get("audit_date"),
        "decoder": audit.get("decoder"),
        "manifest_sha256": audit["manifest_sha256"],
        "expected_dataset_inventory_sha256": audit["dataset_inventory_sha256"],
        "expected_test_image_inventory_sha256": audit["test_image_inventory_sha256"],
        "image_count": audit["image_count"],
        "files_by_split": audit["files_by_split"],
        "unique_decoded_pixel_hashes": audit["unique_decoded_pixel_hashes"],
        "cross_split_identical_pixel_pair_count": audit["cross_split_identical_pixel_pair_count"],
    }


def _decoded_pixel_sha256(path: Path, Image: Any) -> str:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        digest = hashlib.sha256()
        digest.update(rgba.width.to_bytes(4, "big"))
        digest.update(rgba.height.to_bytes(4, "big"))
        digest.update(rgba.tobytes())
        return digest.hexdigest()


def _inventory_records(rows: Iterable[ManifestRow], data_root: Path, Image: Any) -> list[tuple[str, str, str, str]]:
    records = []
    for row in rows:
        parts = _relative_parts(row.image_path)
        path = data_root.joinpath(*parts)
        records.append((row.split, row.category, "/".join(parts), _decoded_pixel_sha256(path, Image)))
    return records


def _inventory_sha256(records: Iterable[tuple[str, str, str, str]]) -> str:
    lines = sorted(f"{split}\t{category}\t{relative}\t{pixel_hash}" for split, category, relative, pixel_hash in records)
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _test_inventory_records(paths: list[Path], data_root: Path, Image: Any) -> tuple[list[tuple[str, str, str, str]], str]:
    records = []
    test_lines = []
    for path in paths:
        parts = path.relative_to(data_root).parts
        if len(parts) < 4 or parts[1] != "test" or parts[0] not in CATEGORY_NAMES:
            raise PilotError(f"Unexpected official test image path during inventory validation: {path}")
        pixel_hash = _decoded_pixel_sha256(path, Image)
        relative = "/".join(parts)
        category = parts[0]
        records.append(("test", category, relative, pixel_hash))
        test_lines.append(f"{relative}\t{category}\t{pixel_hash}")
    test_digest = hashlib.sha256(("\n".join(sorted(test_lines)) + "\n").encode("utf-8")).hexdigest()
    return records, test_digest


def _verify_live_dataset_inventory(
    train_validation_records: list[tuple[str, str, str, str]], test_paths: list[Path],
    data_root: Path, Image: Any,
) -> dict[str, Any]:
    test_records, test_digest = _test_inventory_records(test_paths, data_root, Image)
    all_records = train_validation_records + test_records
    full_digest = _inventory_sha256(all_records)
    pixel_hashes: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for split, _, _, pixel_hash in all_records:
        pixel_hashes[split].add(pixel_hash)
    cross_split_pairs = sum(
        len(left & right)
        for left, right in (
            (pixel_hashes["train"], pixel_hashes["validation"]),
            (pixel_hashes["train"], pixel_hashes["test"]),
            (pixel_hashes["validation"], pixel_hashes["test"]),
        )
    )
    if full_digest != EXPECTED_DATASET_INVENTORY_SHA256:
        raise PilotError(f"Live decoded dataset inventory differs from audit: {full_digest}")
    if test_digest != EXPECTED_TEST_INVENTORY_SHA256:
        raise PilotError(f"Live decoded test inventory differs from audit: {test_digest}")
    if cross_split_pairs != 0 or len({record[3] for record in all_records}) != EXPECTED_DATASET_IMAGES:
        raise PilotError("Live decoded pixels contain exact cross-split duplicates or unexpected counts.")
    return {
        "actual_dataset_inventory_sha256": full_digest,
        "actual_test_image_inventory_sha256": test_digest,
        "actual_image_count": len(all_records),
        "actual_unique_decoded_pixel_hashes": len({record[3] for record in all_records}),
        "actual_cross_split_identical_pixel_pair_count": cross_split_pairs,
        "verified_against_saved_audit": True,
    }


def parse_cuda_device(requested: str, cuda_available: bool, device_count: int) -> int:
    """Validate explicit CUDA index; never assumes cuda:1 or falls back to CPU."""

    match = re.fullmatch(r"cuda(?::(\d+))?", requested.strip().lower())
    if not match:
        raise PilotError("--device must be 'cuda', 'cuda:0', or another explicit cuda:N index.")
    index = int(match.group(1) or 0)
    if not cuda_available:
        raise PilotError("CUDA is unavailable; this pilot requires a successful CUDA smoke check.")
    if index >= device_count:
        raise PilotError(f"Requested {requested}, but torch reports only {device_count} CUDA device(s).")
    return index


def check_smoke_identity(smoke: dict[str, Any], expected: dict[str, Any]) -> None:
    if smoke.get("status") != "passed":
        raise PilotError("CUDA smoke-check marker is missing or did not pass.")
    observed = smoke.get("run_identity")
    if observed != expected:
        raise PilotError("CUDA smoke-check marker does not match the current source, data, weights, device, and runtime.")


def _self_check() -> None:
    train = ManifestRow("screw/train/good/000.png", "screw", "train")
    assert validate_training_rows([train]) == [train]
    for unsafe in (
        ManifestRow("screw/test/crack/000.png", "screw", "test"),
        ManifestRow("screw/train/good/000.png", "screw", "validation"),
        ManifestRow("bottle/train/good/000.png", "screw", "train"),
        ManifestRow("screw/ground_truth/crack/000_mask.png", "screw", "train"),
    ):
        try:
            validate_training_rows([unsafe])
        except PilotError:
            pass
        else:
            raise AssertionError(f"Unsafe row passed training-index validation: {unsafe}")

    gate = RunGate()
    called: list[bool] = []
    try:
        gate.open_test(lambda: called.append(True))
    except PilotError:
        pass
    else:
        raise AssertionError("Test factory was allowed before fit and calibration.")
    assert not called
    gate.dinomaly_fitted = gate.patchcore_fitted = gate.checkpoints_saved = gate.thresholds_calibrated = True
    assert gate.open_test(lambda: "opened") == "opened"
    assert gate.test_inference_started

    class OptimizerStub:
        param_groups = [{"lr": -1.0}]

    optimizer = OptimizerStub()
    _set_official_lr(optimizer, 0, OFFICIAL_TOTAL_ITERATIONS)
    assert optimizer.param_groups[0]["lr"] == 0.0
    _set_official_lr(optimizer, WARMUP_ITERATIONS - 1, OFFICIAL_TOTAL_ITERATIONS)
    assert optimizer.param_groups[0]["lr"] == OFFICIAL_BASE_LR
    _set_official_lr(optimizer, WARMUP_ITERATIONS, OFFICIAL_TOTAL_ITERATIONS)
    assert optimizer.param_groups[0]["lr"] == OFFICIAL_BASE_LR
    _set_official_lr(optimizer, TRAIN_ITERATIONS - 1, OFFICIAL_TOTAL_ITERATIONS)
    assert OFFICIAL_FINAL_LR < optimizer.param_groups[0]["lr"] < OFFICIAL_BASE_LR

    assert parse_cuda_device("cuda", True, 1) == 0
    assert parse_cuda_device("cuda:0", True, 1) == 0
    try:
        parse_cuda_device("cuda:1", True, 1)
    except PilotError:
        pass
    else:
        raise AssertionError("Invalid cuda:1 was accepted on a single-GPU host.")
    try:
        check_smoke_identity({"status": "passed", "run_identity": {"x": 1}}, {"x": 2})
    except PilotError:
        pass
    else:
        raise AssertionError("Mismatched CUDA smoke marker was accepted.")
    print("SELF-CHECK PASS: train-only index, test gate, official LR schedule, CUDA index validation, smoke identity")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run stdlib-only protocol checks and exit.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/production_highres/mvtec_ad/classification_split.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/mvtec_ad_processed_320"))
    parser.add_argument("--decoded-pixel-audit", type=Path, default=DEFAULT_DECODED_PIXEL_AUDIT)
    parser.add_argument("--upstream-root", type=Path, help="Local original CVPR 2025 Dinomaly checkout at the pinned commit.")
    parser.add_argument("--dinomaly-weights", type=Path, help="Directory containing dinov2_vits14_reg4_pretrain.pth for original ViTill.")
    parser.add_argument("--patchcore-artifact-root", type=Path, default=Path("artifacts/production_highres/mvtec_ad"))
    parser.add_argument("--output-dir", type=Path, help="New or smoke-check-only output directory; existing experiment outputs are never overwritten.")
    parser.add_argument("--device", default="cuda:0", help="Validated CUDA index; default cuda:0 for a single RTX 4070.")
    parser.add_argument("--cuda-smoke-check", action="store_true", help="Run CUDA model-forward checks and write a marker; do not train or read test data.")
    return parser.parse_args()


def _require_args(args: argparse.Namespace) -> None:
    missing = [name for name in ("upstream_root", "dinomaly_weights", "output_dir") if getattr(args, name) is None]
    if missing:
        raise PilotError("Required for a full run or CUDA smoke check: " + ", ".join("--" + name.replace("_", "-") for name in missing))


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise PilotError(f"Cannot read Git revision for {root}: {error}") from error


def _validate_upstream(root: Path) -> dict[str, str]:
    root = root.resolve()
    if _git_revision(root) != EXPECTED_UPSTREAM_COMMIT:
        raise PilotError(f"Original CVPR 2025 Dinomaly checkout must be exactly {EXPECTED_UPSTREAM_COMMIT}.")
    try:
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise PilotError(f"Cannot check original Dinomaly tracked source state: {error}") from error
    if dirty:
        raise PilotError("Original Dinomaly tracked source files are modified; use a clean pinned checkout.")
    files: dict[str, str] = {}
    for relative in UPSTREAM_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise PilotError(f"Pinned original Dinomaly source is missing: {path}")
        files[relative] = sha256_file(path)
    return files


def _validate_weight_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    small = args.dinomaly_weights.resolve() / "dinov2_vits14_reg4_pretrain.pth"
    if not small.is_file():
        raise PilotError(f"DINOv2 ViT-Small pretrained checkpoint missing; refusing network download: {small}")
    patchcore_root = args.patchcore_artifact_root.resolve()
    hub = patchcore_root / "torch_hub"
    base = hub / "checkpoints" / "dinov2_vitb14_pretrain.pth"
    revision = hub / "facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    if not base.is_file() or not (revision / "hubconf.py").is_file():
        raise PilotError(
            "PatchCore's project DINOv2-B/14 cache is incomplete. Expected its checkpoint and "
            f"Torch Hub source under {hub}; the runner will not substitute another backbone."
        )
    return small, base


def _validate_classifier_artifacts(artifact_root: Path, manifest_digest: str) -> tuple[dict[str, Path], dict[str, str]]:
    classifier_root = artifact_root / "classifier"
    paths = {
        "classifier": classifier_root / "classifier.pt",
        "class_mapping": classifier_root / "class_mapping.json",
        "training_config": classifier_root / "classifier_training_config.json",
        "calibration": artifact_root / "calibration.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PilotError("The category-routed PatchCore baseline requires the project's classifier artifacts: " + ", ".join(missing))
    try:
        mapping_payload = json.loads(paths["class_mapping"].read_text(encoding="utf-8"))
        training_config = json.loads(paths["training_config"].read_text(encoding="utf-8"))
        calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotError(f"Project classifier metadata is unreadable: {error}") from error
    if mapping_payload.get("category_to_index") != {category: index for index, category in enumerate(CATEGORY_NAMES)}:
        raise PilotError("Project classifier class mapping differs from the pinned 15-category order.")
    if training_config.get("embedding_manifest_sha256") != manifest_digest:
        raise PilotError("Project category classifier was not trained from this pinned manifest.")
    if training_config.get("classifier_type") != "multinomial_logistic_regression":
        raise PilotError("Project classifier artifact has an unsupported classifier type.")
    if calibration.get("class_mapping") != mapping_payload.get("category_to_index"):
        raise PilotError("Classifier calibration class mapping does not match the saved classifier mapping.")
    return paths, {key: sha256_file(path) for key, path in paths.items()}


def _load_route_classifier(runtime: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    try:
        classifier = runtime["torch"].load(paths["classifier"], map_location="cpu", weights_only=True)
        mapping_payload = json.loads(paths["class_mapping"].read_text(encoding="utf-8"))
        calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    except Exception as error:
        raise PilotError(f"Could not load the saved category router: {type(error).__name__}: {error}") from error
    mapping = mapping_payload["category_to_index"]
    weights = classifier.get("weights") if isinstance(classifier, dict) else None
    if not isinstance(weights, runtime["torch"].Tensor) or tuple(weights.shape) != (len(CATEGORY_NAMES), 768):
        raise PilotError("Saved classifier weight shape is incompatible with the DINOv2-B/14 route encoder.")
    return classifier, mapping, calibration


def _route_embeddings(global_embeddings: Any, classifier: dict[str, Any], class_mapping: dict[str, int], calibration: dict[str, Any], runtime: dict[str, Any]) -> tuple[list[str], list[bool], list[float]]:
    predicted_categories, rejected, confidences = [], [], []
    for embedding in global_embeddings:
        route = runtime["route_product"](
            embedding,
            classifier,
            None,
            class_mapping,
            calibration,
            bank_root=".",
            resolve_bank=False,
        )
        # Rejected classifications still localize through the route's raw argmax category.
        predicted_categories.append(route.predicted_category)
        rejected.append(bool(route.rejected))
        confidences.append(float(route.confidence))
    return predicted_categories, rejected, confidences


def _project_source_hashes() -> dict[str, str]:
    result = {"scripts/benchmark_dinomaly_15cat.py": sha256_file(Path(__file__).resolve())}
    for relative in PROJECT_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise PilotError(f"Project source needed for provenance is missing: {path}")
        result[relative] = sha256_file(path)
    return result


def _imports() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
        from sklearn.metrics import average_precision_score, roc_auc_score
        sys.path.insert(0, str((PROJECT_ROOT / "src").resolve()))
        from mvtec_system.anomaly.detectors import PatchCoreDetector
        from mvtec_system.anomaly.multi_layer import MultiLayerConfig
        from mvtec_system.classification.router import route_product
        from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2
        from mvtec_system.core.dinov2 import DINOv2Features
    except Exception as error:
        raise PilotError(
            "Runtime imports failed. Prepare a compatible target CUDA environment; this runner "
            f"does not install or replace dependencies. Original error: {type(error).__name__}: {error}"
        ) from error
    return locals()


def _resolve_cuda(torch: Any, requested: str) -> tuple[Any, dict[str, Any]]:
    index = parse_cuda_device(requested, torch.cuda.is_available(), torch.cuda.device_count())
    torch.cuda.set_device(index)
    device = torch.device(f"cuda:{index}")
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "requested": requested,
        "resolved": str(device),
        "device_count": int(torch.cuda.device_count()),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": sys.modules["torchvision"].__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_launch_blocking_inherited": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }


def _image_tensor(path: Path, runtime: dict[str, Any], device: Any | None = None) -> Any:
    np, torch, Image = runtime["np"], runtime["torch"], runtime["Image"]
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise PilotError(f"Expected preprocessed 320x320 input, got {image.size}: {path}")
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device) if device is not None else tensor


def _pad_normalized(images: Any, torch: Any) -> Any:
    if tuple(images.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise PilotError(f"Expected 320x320 normalized input, got {tuple(images.shape[-2:])}")
    return torch.nn.functional.pad(images, (0, 2, 0, 2), mode="replicate")


def _build_dinomaly(runtime: dict[str, Any], upstream_root: Path, weights_dir: Path, device: Any) -> Any:
    torch = runtime["torch"]
    upstream_root = upstream_root.resolve()
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="dinomaly_import_") as import_cwd:
            os.chdir(import_cwd)
            from models import vit_encoder
            from models.uad import ViTill
            from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp
            from dinov1.utils import trunc_normal_
            from optimizers import StableAdamW
    except Exception as error:
        raise PilotError(f"Cannot import original Dinomaly pinned modules: {type(error).__name__}: {error}") from error
    finally:
        os.chdir(original_cwd)
    for module_name in ("models.uad", "models.vision_transformer", "models.vit_encoder", "dinov1.utils", "optimizers"):
        module = sys.modules[module_name]
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(upstream_root):
            raise PilotError(f"An unexpected module shadowed the pinned original Dinomaly source: {source}")

    # The pinned encoder uses this module variable; set it after import to avoid cwd-relative downloads.
    vit_encoder._WEIGHTS_DIR = str(weights_dir.resolve())
    encoder = vit_encoder.load("dinov2reg_vit_small_14")
    embed_dim, heads = 384, 6
    bottleneck = torch.nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = torch.nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        for _ in range(8)
    ])
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        remove_class_token=False,
        mask_neighbor_size=0,
    )
    trainable = torch.nn.ModuleList([model.bottleneck, model.decoder])
    for module in trainable.modules():
        if isinstance(module, torch.nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    return model.to(device), StableAdamW


def _dinomaly_maps(model: Any, images: Any, runtime: dict[str, Any], gaussian: Any) -> Any:
    torch, F = runtime["torch"], runtime["torch"].nn.functional
    with torch.inference_mode():
        en, de = model(_pad_normalized(images, torch))
        maps = []
        for source, reconstructed in zip(en, de):
            discrepancy = 1.0 - F.cosine_similarity(source, reconstructed)
            maps.append(F.interpolate(discrepancy[:, None], size=(PADDED_SIZE, PADDED_SIZE), mode="bilinear", align_corners=False))
        anomaly_map = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
        anomaly_map = gaussian(anomaly_map)
        return anomaly_map[:, 0, :IMAGE_SIZE, :IMAGE_SIZE].float().cpu().numpy()


def _gaussian_filter(torch: Any, device: Any) -> Any:
    kernel_size, sigma = 5, 4.0
    coordinates = torch.arange(kernel_size, dtype=torch.float32, device=device)
    grid_x = coordinates.repeat(kernel_size).view(kernel_size, kernel_size)
    grid_y = grid_x.t()
    grid = torch.stack((grid_x, grid_y), dim=-1)
    mean = (kernel_size - 1) / 2.0
    kernel = torch.exp(-torch.sum((grid - mean) ** 2, dim=-1) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    layer = torch.nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False).to(device)
    with torch.no_grad():
        layer.weight.copy_(kernel.view(1, 1, kernel_size, kernel_size))
    layer.weight.requires_grad_(False)
    return layer


def _official_loss(en: list[Any], de: list[Any], p: float, factor: float, F: Any, torch: Any) -> Any:
    loss = 0.0
    for source, reconstructed in zip(en, de):
        source = source.detach()
        with torch.no_grad():
            point_distance = 1.0 - F.cosine_similarity(source, reconstructed).unsqueeze(1)
            count = int(point_distance.numel() * (1.0 - p))
            count = min(max(count, 1), point_distance.numel())
            threshold = point_distance.reshape(-1).topk(count).values[-1]
            low_error = point_distance < threshold
        loss = loss + (1.0 - F.cosine_similarity(source.flatten(1), reconstructed.flatten(1))).mean()
        mask = low_error.expand_as(reconstructed)

        def scale_low_error_gradient(gradient: Any, mask: Any = mask, factor: float = factor) -> Any:
            gradient = gradient.clone()
            gradient[mask] *= factor
            return gradient

        reconstructed.register_hook(scale_low_error_gradient)
    return loss / max(len(en), 1)


def _set_official_lr(optimizer: Any, step: int, total_steps: int) -> None:
    if total_steps <= WARMUP_ITERATIONS or not 0 <= step < total_steps:
        raise PilotError("Invalid official scheduler step or schedule length.")
    if step < WARMUP_ITERATIONS:
        lr = OFFICIAL_BASE_LR * step / (WARMUP_ITERATIONS - 1)
    else:
        cosine_steps = total_steps - WARMUP_ITERATIONS
        cosine_index = step - WARMUP_ITERATIONS
        lr = OFFICIAL_FINAL_LR + 0.5 * (OFFICIAL_BASE_LR - OFFICIAL_FINAL_LR) * (
            1.0 + math.cos(math.pi * cosine_index / cosine_steps)
        )
    if len(optimizer.param_groups) != 1:
        raise PilotError("Original Dinomaly uses one StableAdamW parameter group.")
    optimizer.param_groups[0]["lr"] = lr


def _load_manifest_image_batch(paths: list[Path], runtime: dict[str, Any], device: Any) -> Any:
    torch = runtime["torch"]
    return torch.stack([_image_tensor(path, runtime) for path in paths]).to(device)


def _fit_dinomaly(
    train_paths: list[Path], runtime: dict[str, Any], upstream_root: Path,
    weights_dir: Path, device: Any, output_dir: Path,
) -> tuple[Path, list[tuple[int, float]]]:
    torch = runtime["torch"]
    np = runtime["np"]
    model, StableAdamW = _build_dinomaly(runtime, upstream_root, weights_dir, device)
    dataset = _GoodDataset(train_paths, runtime)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, generator=generator,
    )
    trainable = torch.nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
        amsgrad=True, eps=1e-10,
    )
    F = torch.nn.functional
    iterator = iter(loader)
    logged: list[tuple[int, float]] = []
    recent: list[float] = []
    model.train()
    for step in range(TRAIN_ITERATIONS):
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)
        images = _pad_normalized(images.to(device), torch)
        _set_official_lr(optimizer, step, OFFICIAL_TOTAL_ITERATIONS)
        en, de = model(images)
        p = min(0.9 * step / 1000.0, 0.9)
        loss = _official_loss(en, de, p=p, factor=0.1, F=F, torch=torch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
        optimizer.step()
        recent.append(float(loss.detach().item()))
        if (step + 1) % 100 == 0:
            logged.append((step + 1, float(np.mean(recent))))
            print(f"Dinomaly iter {step + 1}/{TRAIN_ITERATIONS}: loss={logged[-1][1]:.6f}", flush=True)
            recent.clear()

    checkpoint_path = output_dir / "dinomaly_vit_small_final.pt"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "iteration": TRAIN_ITERATIONS, "seed": SEED}, checkpoint_path)
    del state, optimizer, loader, iterator, trainable
    return checkpoint_path, logged


class _GoodDataset:
    """Small PIL dataset; constructed from already-validated train paths only."""

    def __init__(self, paths: list[Path], runtime: dict[str, Any]) -> None:
        self.paths = paths
        self.runtime = runtime

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        return _image_tensor(self.paths[index], self.runtime)


def _fit_category_patchcore(
    train_rows: list[ManifestRow], data_root: Path, runtime: dict[str, Any], device: Any,
    artifact_root: Path, output_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Path]]:
    """Fit the production PatchCore implementation independently for each train category."""
    torch = runtime["torch"]
    PatchCoreDetector = runtime["PatchCoreDetector"]
    MultiLayerConfig = runtime["MultiLayerConfig"]
    DINOv2Config, SharedDINOv2 = runtime["DINOv2Config"], runtime["SharedDINOv2"]
    config = MultiLayerConfig(
        layers=PATCHCORE_LAYERS,
        neighborhood_size=PATCHCORE_NEIGHBORHOOD,
        coreset_ratio=PATCHCORE_RATIO,
        coreset_max_points=PATCHCORE_MAX_POINTS,
        seed=SEED,
    )
    encoder = SharedDINOv2(
        DINOv2Config(image_size=(IMAGE_SIZE, IMAGE_SIZE), feature_layers=PATCHCORE_LAYERS),
        artifact_root=artifact_root,
        device=device,
    )
    detectors: dict[str, Any] = {}
    checkpoints: dict[str, Path] = {}
    output_root = output_dir / "patchcore_category_banks"
    for category in CATEGORY_NAMES:
        paths = [
            data_root.joinpath(*_relative_parts(row.image_path))
            for row in train_rows if row.category == category
        ]
        if not paths:
            raise PilotError(f"No train/good images for PatchCore category {category}.")
        detector = PatchCoreDetector(config=config, k=1).fit(category, paths, encoder, batch_size=BATCH_SIZE)
        if detector.category != category:
            raise PilotError(f"PatchCore fit returned wrong category bank for {category}.")
        bank_dir = output_root / category
        detector.save(bank_dir)
        detector.coreset = detector.coreset.to(device)
        detectors[category] = detector
        checkpoint = bank_dir / "coreset.pt"
        if not checkpoint.is_file():
            raise PilotError(f"PatchCoreDetector.save did not write its expected coreset checkpoint: {checkpoint}")
        checkpoints[category] = checkpoint
        print(f"Fitted category-specific PatchCore bank {category}: {len(paths)} train/good images.", flush=True)
        torch.cuda.empty_cache()
    return encoder, detectors, checkpoints


def _load_test_paths(data_root: Path) -> list[Path]:
    """Discover image paths only; do not inspect test category labels or masks here."""

    paths: list[Path] = []
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    for category in CATEGORY_NAMES:
        test_root = data_root / category / "test"
        if not test_root.is_dir():
            raise PilotError(f"Official test category folder is missing: {test_root}")
        for folder in sorted(path for path in test_root.iterdir() if path.is_dir()):
            for image_path in sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions):
                paths.append(image_path)
    paths.sort(key=lambda path: path.as_posix())
    if len(paths) != EXPECTED_TEST_IMAGES:
        raise PilotError(f"Expected all {EXPECTED_TEST_IMAGES} official test images; discovered {len(paths)}.")
    return paths


def _group_test_samples_after_inference(paths: list[Path], data_root: Path) -> list[TestSample]:
    """Attach category/defect masks only after both methods have produced maps."""

    samples: list[TestSample] = []
    for image_path in paths:
        parts = image_path.relative_to(data_root).parts
        if len(parts) < 4 or parts[1] != "test" or parts[0] not in CATEGORY_NAMES:
            raise PilotError(f"Unexpected official test image path: {image_path}")
        category, defect_type = parts[0], parts[2]
        if defect_type == "good":
            samples.append(TestSample(image_path, category, "good", None))
        else:
            mask_path = data_root / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise PilotError(f"Missing ground-truth mask for inferred defect image: {mask_path}")
            samples.append(TestSample(image_path, category, defect_type, mask_path))
    return samples


def _infer_paths(
    paths: list[Path], runtime: dict[str, Any], device: Any, dinomaly: Any, patch_encoder: Any,
    patch_detectors: dict[str, Any], classifier: dict[str, Any], class_mapping: dict[str, int],
    classifier_calibration: dict[str, Any], batch_size: int = BATCH_SIZE,
) -> tuple[Any, Any, list[str], list[bool], list[float]]:
    np, torch = runtime["np"], runtime["torch"]
    dinomaly_maps: list[Any] = []
    patchcore_maps: list[Any] = []
    predicted_categories: list[str] = []
    rejected_routes: list[bool] = []
    route_confidences: list[float] = []
    dinomaly.eval()
    patch_encoder.backbone.eval()
    gaussian = _gaussian_filter(torch, device)
    for start in range(0, len(paths), batch_size):
        images = _load_manifest_image_batch(paths[start : start + batch_size], runtime, device)
        current_dino = _dinomaly_maps(dinomaly, images, runtime, gaussian)
        with torch.inference_mode():
            patch_features = patch_encoder.forward(images)
            if patch_features.layer_patch_embeddings is None:
                raise PilotError("Category-routed PatchCore requires the configured multi-layer DINOv2 embeddings.")
            routed_categories, routed_rejections, routed_confidences = _route_embeddings(
                patch_features.global_embedding, classifier, class_mapping, classifier_calibration, runtime
            )
            current_patch = np.empty((len(routed_categories), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            for category in set(routed_categories):
                indices = np.asarray([index for index, routed in enumerate(routed_categories) if routed == category], dtype=int)
                tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=patch_features.global_embedding.device)
                category_features = runtime["DINOv2Features"](
                    global_embedding=patch_features.global_embedding[tensor_indices],
                    patch_embeddings=patch_features.patch_embeddings[tensor_indices],
                    layer_patch_embeddings={
                        layer: embeddings[tensor_indices]
                        for layer, embeddings in patch_features.layer_patch_embeddings.items()
                    },
                )
                patch_results = patch_detectors[category].predict_batch(category_features)
                current_patch[indices] = np.stack([result.anomaly_map for result in patch_results])
        current_patch = current_patch.astype(np.float32, copy=False)
        if current_dino.shape != current_patch.shape or current_dino.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise PilotError(f"Map geometry mismatch: Dinomaly={current_dino.shape}, PatchCore={current_patch.shape}")
        if not np.isfinite(current_dino).all() or not np.isfinite(current_patch).all():
            raise PilotError("A detector produced non-finite heatmap values.")
        dinomaly_maps.append(current_dino.astype(np.float32, copy=False))
        patchcore_maps.append(current_patch)
        predicted_categories.extend(routed_categories)
        rejected_routes.extend(routed_rejections)
        route_confidences.extend(routed_confidences)
        del images, patch_features, patch_results
    return (
        np.concatenate(dinomaly_maps), np.concatenate(patchcore_maps),
        predicted_categories, rejected_routes, route_confidences,
    )


def _calibrate_threshold(scores: Any, np: Any) -> float:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise PilotError("Validation maps are empty or non-finite.")
    return float(np.quantile(flat, CALIBRATION_QUANTILE, method="linear"))


def _calibrate_validation(
    validation_rows: list[ManifestRow], data_root: Path, runtime: dict[str, Any], device: Any,
    dinomaly: Any, patch_encoder: Any, patch_detectors: dict[str, Any], classifier: dict[str, Any],
    class_mapping: dict[str, int], classifier_calibration: dict[str, Any],
) -> dict[str, Any]:
    if not validation_rows or any(row.split != "validation" for row in validation_rows):
        raise PilotError("Threshold calibration accepts validation rows only.")
    paths = [data_root.joinpath(*_relative_parts(row.image_path)) for row in validation_rows]
    dino_maps, patch_maps, routes, rejected, _ = _infer_paths(
        paths, runtime, device, dinomaly, patch_encoder, patch_detectors,
        classifier, class_mapping, classifier_calibration,
    )
    thresholds: dict[str, Any] = {
        "primary_global": {
            "dinomaly": _calibrate_threshold(dino_maps, runtime["np"]),
            "patchcore": _calibrate_threshold(patch_maps, runtime["np"]),
            "quantile": CALIBRATION_QUANTILE,
            "validation_good_images": len(paths),
            "validation_pixels_per_method": int(dino_maps.size),
        },
        "by_predicted_category": {},
        "routing": {
            "images": len(routes),
            "rejected_images_still_localized": int(sum(rejected)),
            "category_counts": {category: routes.count(category) for category in CATEGORY_NAMES},
        },
    }
    route_array = runtime["np"].asarray(routes)
    for category in CATEGORY_NAMES:
        indices = runtime["np"].flatnonzero(route_array == category)
        if len(indices):
            category_dino, category_patch = dino_maps[indices], patch_maps[indices]
            thresholds["by_predicted_category"][category] = {
                "dinomaly": _calibrate_threshold(category_dino, runtime["np"]),
                "patchcore": _calibrate_threshold(category_patch, runtime["np"]),
                "validation_good_images_routed": len(indices),
                "validation_pixels_per_method": int(category_dino.size),
                "fallback_to_global": False,
            }
        else:
            thresholds["by_predicted_category"][category] = {
                "dinomaly": None,
                "patchcore": None,
                "validation_good_images_routed": 0,
                "validation_pixels_per_method": 0,
                "fallback_to_global": True,
            }
    print(
        "Calibrated primary global q=.99 thresholds from all validation GOOD maps: "
        f"Dinomaly={thresholds['primary_global']['dinomaly']:.6g}; "
        f"category-routed PatchCore={thresholds['primary_global']['patchcore']:.6g}",
        flush=True,
    )
    del dino_maps, patch_maps
    return thresholds


def _routed_thresholds(predicted_categories: list[str], thresholds: dict[str, Any], model_name: str, np: Any) -> Any:
    global_threshold = float(thresholds["primary_global"][model_name])
    by_category = thresholds["by_predicted_category"]
    return np.asarray([
        float(by_category[category][model_name])
        if by_category[category][model_name] is not None else global_threshold
        for category in predicted_categories
    ], dtype=np.float32)


def _load_test_masks(samples: list[TestSample], runtime: dict[str, Any]) -> tuple[Any, Any]:
    np = runtime["np"]
    masks, labels = [], []
    for sample in samples:
        labels.append(0 if sample.defect_type == "good" else 1)
        if sample.mask_path is None:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        else:
            with runtime["Image"].open(sample.mask_path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise PilotError(f"Expected 320x320 ground-truth mask, got {mask.shape}: {sample.mask_path}")
            mask = mask.astype(np.uint8)
        masks.append(mask)
    return np.stack(masks), np.asarray(labels, dtype=np.uint8)


def _pixel_metrics(scores: Any, masks: Any, image_labels: Any, threshold: Any, runtime: dict[str, Any]) -> dict[str, float | int]:
    np = runtime["np"]
    score_maps = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(image_labels, dtype=np.uint8)
    if score_maps.ndim != 3 or len(score_maps) != len(labels):
        raise PilotError("Pixel scores must be [images,height,width] and align with image labels.")
    if np.isscalar(threshold):
        thresholds = np.full(len(labels), float(threshold), dtype=np.float32)
        threshold_fields: dict[str, float] = {"threshold": float(threshold)}
    else:
        thresholds = np.asarray(threshold, dtype=np.float32).reshape(-1)
        if len(thresholds) != len(labels):
            raise PilotError("Per-image routed thresholds do not align with predictions.")
        threshold_fields = {
            "threshold_mean": float(thresholds.mean()),
            "threshold_min": float(thresholds.min()),
            "threshold_max": float(thresholds.max()),
        }
    y_true = (np.asarray(masks).reshape(-1) > 0).astype(np.uint8)
    y_score = score_maps.reshape(-1)
    if y_true.size != y_score.size or np.unique(y_true).size != 2:
        raise PilotError("Pixel metric inputs must align and contain both normal and defect pixels.")
    prediction_maps = score_maps >= thresholds[:, None, None]
    prediction = prediction_maps.reshape(-1)
    tp = int(np.count_nonzero(prediction & (y_true == 1)))
    fp = int(np.count_nonzero(prediction & (y_true == 0)))
    fn = int(np.count_nonzero(~prediction & (y_true == 1)))
    tn = int(np.count_nonzero(~prediction & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    from sklearn.metrics import average_precision_score, roc_auc_score

    good_pixels = np.repeat(labels == 0, IMAGE_SIZE * IMAGE_SIZE)
    good_fp = float(np.count_nonzero(prediction & good_pixels) / np.count_nonzero(good_pixels)) if np.any(good_pixels) else 0.0
    area_by_image = prediction_maps.mean(axis=(1, 2), dtype=np.float32)
    result = {
        "test_images": int(len(labels)),
        "good_images": int(np.count_nonzero(labels == 0)),
        "defect_images": int(np.count_nonzero(labels == 1)),
        "map_images": int(len(labels)),
        "map_coverage": 1.0,
        "mask_images": int(len(labels)),
        "mask_coverage": 1.0,
        "pixel_count": int(y_true.size),
        "pixel_ap": float(average_precision_score(y_true, y_score)),
        "pixel_auroc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "specificity": float(specificity),
        "good_fp_pixel_rate": good_fp,
        "mean_predicted_area_ratio": float(area_by_image.mean()),
        "mean_predicted_area_ratio_good": float(area_by_image[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "mean_predicted_area_ratio_defect": float(area_by_image[labels == 1].mean()) if np.any(labels == 1) else 0.0,
    }
    result.update(threshold_fields)
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PilotError("No metrics rows to write.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_macro(per_category: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "pixel_ap", "pixel_auroc", "precision", "recall", "f1", "dice", "iou",
        "specificity", "good_fp_pixel_rate", "mean_predicted_area_ratio",
        "mean_predicted_area_ratio_good", "mean_predicted_area_ratio_defect",
        "route_q99_precision", "route_q99_recall", "route_q99_f1", "route_q99_dice",
        "route_q99_iou", "route_q99_specificity", "route_q99_good_fp_pixel_rate",
        "route_q99_mean_predicted_area_ratio_good", "route_q99_mean_predicted_area_ratio_defect",
    )
    return {key: sum(float(row[key]) for row in per_category.values()) / len(per_category) for key in metric_keys}


def _run_fingerprint(
    args: argparse.Namespace, manifest_digest: str, project_hashes: dict[str, str],
    upstream_hashes: dict[str, str], weights: tuple[Path, Path], runtime: dict[str, Any],
    cuda_info: dict[str, Any], audit_identity: dict[str, Any], classifier_hashes: dict[str, str],
) -> dict[str, Any]:
    torch = runtime["torch"]
    return {
        "manifest_sha256": manifest_digest,
        "decoded_pixel_audit_identity": audit_identity,
        "classifier_artifact_sha256": classifier_hashes,
        "script_and_project_source_sha256": project_hashes,
        "project_git_revision": _git_revision(PROJECT_ROOT),
        "candidate": "original CVPR 2025 Dinomaly ViTill-Small/14",
        "upstream_repository": "https://github.com/guojiajeremy/Dinomaly",
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "upstream_source_sha256": upstream_hashes,
        "pretrained_checkpoints": {
            "dinomaly_dinov2_vits14_reg4": {"path": str(weights[0]), "sha256": sha256_file(weights[0])},
            "patchcore_dinov2_vitb14": {"path": str(weights[1]), "sha256": sha256_file(weights[1])},
        },
        "runtime": cuda_info,
        "fixed_protocol": {
            "seed": SEED,
            "iterations": TRAIN_ITERATIONS,
            "official_schedule_total_iterations": OFFICIAL_TOTAL_ITERATIONS,
            "learning_rate_schedule": "official WarmCosineScheduler; base=2e-3, final=2e-4, linear warmup=100, cosine horizon=10000; pilot stops at step 5000",
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "padded_size": PADDED_SIZE,
            "pad_mode": "replicate after ImageNet normalization; bottom=2,right=2",
            "device": cuda_info["resolved"],
            "torch_deterministic_cudnn": bool(torch.backends.cudnn.deterministic),
            "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }


def _run_cuda_smoke(
    args: argparse.Namespace, runtime: dict[str, Any], device: Any, cuda_info: dict[str, Any],
    fingerprint: dict[str, Any], classifier: dict[str, Any], class_mapping: dict[str, int],
    calibration: dict[str, Any],
) -> None:
    torch = runtime["torch"]
    dinomaly, StableAdamW = _build_dinomaly(runtime, args.upstream_root, args.dinomaly_weights, device)
    dinomaly.train()
    dummy = torch.randn((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    en, de = dinomaly(_pad_normalized(dummy, torch))
    if len(en) != 2 or en[0].shape[-2:] != (23, 23):
        raise PilotError("Dinomaly CUDA forward returned unexpected 23x23 patch geometry.")
    optimizer = StableAdamW(
        [{"params": list(dinomaly.bottleneck.parameters()) + list(dinomaly.decoder.parameters())}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
        amsgrad=True, eps=1e-10,
    )
    _set_official_lr(optimizer, 0, OFFICIAL_TOTAL_ITERATIONS)
    loss = _official_loss(en, de, p=0.9, factor=0.1, F=torch.nn.functional, torch=torch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(dinomaly.bottleneck.parameters()) + list(dinomaly.decoder.parameters()), 0.1)
    optimizer.step()
    torch.cuda.synchronize(device)
    del dinomaly, optimizer, dummy, en, de, loss
    torch.cuda.empty_cache()

    from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2
    encoder = SharedDINOv2(
        DINOv2Config(image_size=(IMAGE_SIZE, IMAGE_SIZE), feature_layers=PATCHCORE_LAYERS),
        artifact_root=args.patchcore_artifact_root,
        device=device,
    )
    with torch.inference_mode():
        output = encoder.forward(torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device))
    if output.layer_patch_embeddings is None or output.patch_embeddings.shape[1] != 23 * 23:
        raise PilotError("Project DINOv2-B CUDA smoke forward did not produce the expected 23x23 patch grid.")
    route_categories, route_rejected, route_confidences = _route_embeddings(
        output.global_embedding, classifier, class_mapping, calibration, runtime
    )
    torch.cuda.synchronize(device)
    marker = {
        "status": "passed",
        "run_identity": fingerprint,
        "device": cuda_info,
        "smoke": {
            "dinomaly_train_step": "forward/backward/optimizer passed on synthetic batch 8 at 320px padded to 322px",
            "patchcore_encoder_forward": "passed on synthetic normalized 320x320 input; project pad gives 23x23 tokens",
            "project_classifier_route": {"predicted_category": route_categories[0], "rejected": route_rejected[0], "confidence": route_confidences[0]},
            "cuda_launch_blocking_set_by_runner": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = args.output_dir / "cuda_smoke_check.json"
    if marker_path.exists():
        raise PilotError(f"Refusing to overwrite existing smoke marker: {marker_path}")
    _write_json(marker_path, marker)
    print(json.dumps(marker, indent=2))


def _add_routed_metrics(primary: dict[str, Any], routed: dict[str, Any]) -> None:
    keys = (
        "precision", "recall", "f1", "dice", "iou", "specificity", "good_fp_pixel_rate",
        "mean_predicted_area_ratio", "mean_predicted_area_ratio_good", "mean_predicted_area_ratio_defect",
        "threshold_mean", "threshold_min", "threshold_max",
    )
    for key in keys:
        if key in routed:
            primary[f"route_q99_{key}"] = routed[key]


def _write_heatmap_gallery(
    samples: list[TestSample], dinomaly_maps: Any, patchcore_maps: Any, masks: Any,
    data_root: Path, output_path: Path, runtime: dict[str, Any],
) -> None:
    from PIL import ImageDraw

    np, Image = runtime["np"], runtime["Image"]
    categories = ("bottle", "cable", "screw", "zipper")
    panels: list[tuple[str, int]] = []
    for category in categories:
        indices = [index for index, sample in enumerate(samples) if sample.category == category]
        good = next((index for index in indices if samples[index].defect_type == "good"), None)
        defect = next((index for index in indices if samples[index].defect_type != "good"), None)
        if good is not None:
            panels.append((f"{category} / GOOD", good))
        if defect is not None:
            panels.append((f"{category} / {samples[defect].defect_type}", defect))
    if not panels:
        raise PilotError("Could not select representative test images for the heatmap gallery.")
    tile_w, tile_h, title_h = IMAGE_SIZE, IMAGE_SIZE, 48
    canvas = Image.new("RGB", (tile_w * 4, len(panels) * (tile_h + title_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (title, index) in enumerate(panels):
        top = row * (tile_h + title_h)
        draw.text((4, top + 4), title, fill="black")
        for column, label in enumerate(("Input", "Ground truth", "Dinomaly", "PatchCore")):
            draw.text((column * tile_w + 4, top + 25), label, fill="black")
        sample = samples[index]
        with Image.open(sample.image_path) as image:
            original = image.convert("RGB")
        gt = (np.asarray(masks[index], dtype=np.uint8) > 0).astype(np.uint8)
        gt_rgb = np.repeat((gt * 255)[:, :, None], 3, axis=2)
        rendered = [original, Image.fromarray(gt_rgb, mode="RGB")]
        for score_map in (dinomaly_maps[index], patchcore_maps[index]):
            low, high = np.quantile(score_map, [0.01, 0.99])
            normalized = np.clip((score_map - low) / max(float(high - low), 1e-8), 0.0, 1.0)
            alpha = normalized[:, :, None] * 0.55
            source = np.asarray(original, dtype=np.float32)
            red = np.zeros_like(source)
            red[:, :, 0] = 255
            overlay = np.clip(source * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)
            rendered.append(Image.fromarray(overlay, mode="RGB"))
        for column, panel in enumerate(rendered):
            canvas.paste(panel, (column * tile_w, top + title_h))
    canvas.save(output_path)


def _write_metric_charts(metrics: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    models = (("dinomaly", "Dinomaly"), ("patchcore", "PatchCore"))
    colors = ("#2563eb", "#f97316")
    macro = metrics["macro"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = (
        ("pixel_ap", "Macro Pixel AP", "AP"),
        ("pixel_auroc", "Macro Pixel AUROC", "AUROC"),
        ("f1", "Macro F1 at validation q=.99", "F1"),
        ("good_fp_pixel_rate", "Macro GOOD false-positive pixel rate", "FP rate"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        values = [float(macro[model][key]) * 100 for model, _ in models]
        bars = axis.bar([label for _, label in models], values, color=colors, width=0.58)
        axis.set_title(title)
        axis.set_ylabel(ylabel + " (%)")
        axis.set_ylim(0, max(5, max(values, default=0) * 1.18))
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    figure.suptitle("15-category localization comparison", fontsize=13)
    figure.tight_layout()
    overview_path = output_dir / "pixel_metrics_overview.png"
    figure.savefig(overview_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    categories = CATEGORY_NAMES
    y = np.arange(len(categories))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 8))
    for index, (model, label) in enumerate(models):
        values = [float(metrics["per_category"][model][category]["pixel_ap"]) * 100 for category in categories]
        offset = (index - 0.5) * width
        axis.barh(y + offset, values, height=width, label=label, color=colors[index])
    axis.set_yticks(y, categories)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Pixel AP (%)")
    axis.set_title("Pixel AP by category")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    category_path = output_dir / "pixel_ap_by_category.png"
    figure.savefig(category_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [overview_path.name, category_path.name]


def _run(args: argparse.Namespace) -> None:
    _require_args(args)
    for name in ("manifest", "data_root", "decoded_pixel_audit", "upstream_root", "dinomaly_weights", "patchcore_artifact_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    train_rows, validation_rows, manifest_digest = validate_manifest(args.manifest, args.data_root)
    train_rows = validate_training_rows(train_rows)
    audit_identity = _validate_decoded_audit_identity(args.decoded_pixel_audit, manifest_digest)
    from PIL import Image as PILImage
    train_validation_records = _inventory_records(train_rows + validation_rows, args.data_root, PILImage)
    if len(train_validation_records) != EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS:
        raise PilotError("Live train/validation content inventory has an unexpected image count.")
    split_hashes = {
        split: {record[3] for record in train_validation_records if record[0] == split}
        for split in ("train", "validation")
    }
    if split_hashes["train"] & split_hashes["validation"]:
        raise PilotError("Live train and validation images contain exact decoded-pixel duplicates.")
    audit_identity["actual_train_validation_inventory_sha256"] = _inventory_sha256(train_validation_records)
    if audit_identity["actual_train_validation_inventory_sha256"] != EXPECTED_TRAIN_VALIDATION_INVENTORY_SHA256:
        raise PilotError(
            "Live decoded train/validation inventory differs from the verified split: "
            f"{audit_identity['actual_train_validation_inventory_sha256']}"
        )
    audit_identity["expected_train_validation_inventory_sha256"] = EXPECTED_TRAIN_VALIDATION_INVENTORY_SHA256
    audit_identity["actual_train_validation_image_count"] = len(train_validation_records)
    upstream_hashes = _validate_upstream(args.upstream_root)
    weights = _validate_weight_paths(args)
    classifier_paths, classifier_hashes = _validate_classifier_artifacts(args.patchcore_artifact_root, manifest_digest)
    project_hashes = _project_source_hashes()
    runtime = _imports()
    torch = runtime["torch"]
    classifier, class_mapping, classifier_calibration = _load_route_classifier(runtime, classifier_paths)
    device, cuda_info = _resolve_cuda(torch, args.device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    runtime["np"].random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fingerprint = _run_fingerprint(
        args, manifest_digest, project_hashes, upstream_hashes, weights, runtime, cuda_info,
        audit_identity, classifier_hashes,
    )
    if args.cuda_smoke_check:
        _run_cuda_smoke(args, runtime, device, cuda_info, fingerprint, classifier, class_mapping, classifier_calibration)
        return

    marker_path = args.output_dir / "cuda_smoke_check.json"
    if not marker_path.is_file():
        raise PilotError(f"Run --cuda-smoke-check successfully on this host before fitting: {marker_path}")
    check_smoke_identity(json.loads(marker_path.read_text(encoding="utf-8")), fingerprint)
    existing = [path for path in args.output_dir.iterdir() if path.name != "cuda_smoke_check.json"]
    if existing:
        raise PilotError(f"Output directory already contains experiment files; refusing overwrite: {existing[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gate = RunGate()
    train_paths = [args.data_root.joinpath(*_relative_parts(row.image_path)) for row in train_rows]
    print(f"Fitting original Dinomaly ViTill-Small on {len(train_paths)} train/good images; {TRAIN_ITERATIONS}-step feasibility pilot.", flush=True)
    dinomaly_checkpoint, loss_log = _fit_dinomaly(
        train_paths, runtime, args.upstream_root, args.dinomaly_weights, device, args.output_dir
    )
    gate.dinomaly_fitted = True
    with (args.output_dir / "training_loss.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("iteration", "mean_loss_last_100"))
        writer.writerows(loss_log)

    print("Fitting the project's 15 category-specific PatchCore banks on train/good only.", flush=True)
    patch_encoder, patch_detectors, patchcore_checkpoints = _fit_category_patchcore(
        train_rows, args.data_root, runtime, device, args.patchcore_artifact_root, args.output_dir
    )
    gate.patchcore_fitted = True
    checkpoint_hashes = {
        "dinomaly_vi_till_small": {"path": str(dinomaly_checkpoint), "sha256": sha256_file(dinomaly_checkpoint)},
        "patchcore_category_banks": {
            category: {"path": str(path), "sha256": sha256_file(path)}
            for category, path in patchcore_checkpoints.items()
        },
    }
    gate.checkpoints_saved = all(Path(item["path"]).is_file() for item in [checkpoint_hashes["dinomaly_vi_till_small"], *checkpoint_hashes["patchcore_category_banks"].values()])
    if not gate.checkpoints_saved:
        raise PilotError("Not all model checkpoints were saved before calibration.")

    dinomaly, _ = _build_dinomaly(runtime, args.upstream_root, args.dinomaly_weights, device)
    checkpoint = torch.load(dinomaly_checkpoint, map_location="cpu", weights_only=True)
    dinomaly.load_state_dict(checkpoint["state_dict"], strict=True)
    dinomaly.eval()
    del checkpoint
    torch.cuda.empty_cache()

    provenance = {
        "status": "both_fits_and_checkpoints_saved; test_not_accessed",
        "run_fingerprint": fingerprint,
        "trained_checkpoints": checkpoint_hashes,
        "dataset_inventory_audit": {
            **audit_identity,
            "actual_dataset_inventory_sha256": None,
            "actual_test_image_inventory_sha256": None,
            "live_test_validation": "pending; executed only after fit and validation calibration",
        },
        "training": {
            "manifest_train_good_rows": len(train_rows),
            "only_train_good_rows_used": True,
            "validation_used_for_fit": False,
            "official_test_loader_created": False,
            "ground_truth_category_used_for_routing_or_calibration": False,
            "patchcore_category_models": len(patch_detectors),
        },
        "candidate": {
            "name": "original CVPR 2025 Dinomaly ViTill-Small/14",
            "implementation": "ViTill from the original Dinomaly repository; this is not Dinomaly2",
            "backbone": "dinov2reg_vit_small_14",
            "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
            "encoder_fusion_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "decoder_fusion_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "bottleneck": "one bMlp(384,1536,384,drop=0.2)",
            "decoder": "8 ViT blocks, 6 heads, LinearAttention2, LayerNorm eps=1e-8",
            "input_size": IMAGE_SIZE,
            "input_padding": "replicate after ImageNet normalization; bottom/right +2 to 322",
            "map_resize": "bilinear to 322x322, align_corners=False to match project interpolation protocol",
            "map_crop": "top-left 320x320 after 5x5 sigma-4 Gaussian filter",
            "training_iterations": TRAIN_ITERATIONS,
            "pilot_role": "feasibility only; cannot reject the candidate or establish convergence",
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "optimizer": "official StableAdamW; one trainable group; lr=2e-3, betas=(0.9,0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10",
            "lr_schedule": {
                "implementation": "official WarmCosineScheduler formula",
                "base_lr": OFFICIAL_BASE_LR,
                "final_lr": OFFICIAL_FINAL_LR,
                "warmup_iterations": WARMUP_ITERATIONS,
                "schedule_total_iterations": OFFICIAL_TOTAL_ITERATIONS,
                "pilot_stops_at_iteration": TRAIN_ITERATIONS,
                "pilot_reaches_final_lr": False,
            },
            "loss": "global_cosine_hm_percent; p ramps from 0 to 0.9 over 1000 steps; low-error gradient factor=0.1; clip norm=0.1",
        },
        "patchcore_project_baseline": {
            "bank_scope": "one project PatchCore bank per category; selected only by saved classifier top-1 prediction",
            "router_classifier_sha256": classifier_hashes["classifier"],
            "router_rejection": "rejected images are still localized using route.predicted_category; rejection does not suppress map coverage",
            "backbone": "project DINOv2 ViT-B/14",
            "image_size": IMAGE_SIZE,
            "padding": "project SharedDINOv2 bottom/right replicate pad to 322",
            "layers": list(PATCHCORE_LAYERS),
            "neighborhood_size": PATCHCORE_NEIGHBORHOOD,
            "coreset_ratio": PATCHCORE_RATIO,
            "coreset_max_points": PATCHCORE_MAX_POINTS,
            "k": 1,
            "fit_implementation": "src/mvtec_system/anomaly/detectors.py::PatchCoreDetector.fit",
            "map_interpolation": "project make_anomaly_maps; align_corners=False; crop to 320 mask geometry",
        },
        "thresholds": "primary is one global q=.99 threshold per method from pooled validation GOOD maps; secondary q=.99 thresholds are calibrated and selected by predicted category, never test ground-truth category",
        "evaluation": "Pixel AP/AUROC are threshold-free; binary metrics use global primary threshold and separately report predicted-category-routed q=.99 diagnostic; masks remain at 320x320",
        "uncertainty": {"paired_group_bootstrap_ci": None, "reason": "No physical-object/acquisition group IDs exist in the pinned manifest."},
        "visual_outputs": {
            "charts": ["pixel_metrics_overview.png", "pixel_ap_by_category.png"],
            "heatmap_gallery": "heatmap_gallery.png",
        },
        "limitations": [
            "The official MVTec test split was viewed during earlier project work; all test results are exploratory.",
            "The 5,000-iteration run is a feasibility pilot only and cannot be used to reject this candidate.",
            "The official map evaluator uses a different interpolation setting; align_corners=False is deliberate to match project evaluation geometry.",
            "Exact decoded-pixel checks do not detect perceptual near-duplicates or acquisition-session leakage.",
            "GPU nondeterministic kernels may prevent bit-identical reruns despite fixed seeds and deterministic cuDNN settings.",
        ],
    }
    _write_json(args.output_dir / "provenance.json", provenance)

    thresholds = _calibrate_validation(
        validation_rows, args.data_root, runtime, device, dinomaly, patch_encoder,
        patch_detectors, classifier, class_mapping, classifier_calibration,
    )
    gate.thresholds_calibrated = True
    _write_json(args.output_dir / "validation_thresholds.json", {
        "quantile": CALIBRATION_QUANTILE,
        "source_split": "known-good validation rows only",
        "threshold_routing": "predicted category for category-specific thresholds; pooled maps for primary global threshold",
        "manifest_sha256": manifest_digest,
        **thresholds,
    })
    provenance["status"] = "fits_saved_and_validation_calibrated; test_not_accessed"
    provenance["validation_calibration"] = thresholds
    _write_json(args.output_dir / "provenance.json", provenance)

    test_paths = gate.open_test(lambda: _load_test_paths(args.data_root))
    live_inventory = _verify_live_dataset_inventory(train_validation_records, test_paths, args.data_root, runtime["Image"])
    provenance["dataset_inventory_audit"].update(live_inventory)
    provenance["dataset_inventory_audit"]["live_test_validation"] = "passed before test model inference"
    provenance["status"] = "fits_saved_validation_calibrated_and_dataset_inventory_verified_before_test_inference"
    _write_json(args.output_dir / "provenance.json", provenance)

    dino_maps, patch_maps, predicted_categories, rejected_routes, route_confidences = _infer_paths(
        test_paths, runtime, device, dinomaly, patch_encoder, patch_detectors,
        classifier, class_mapping, classifier_calibration,
    )
    if len(dino_maps) != EXPECTED_TEST_IMAGES or len(patch_maps) != EXPECTED_TEST_IMAGES:
        raise PilotError("A detector did not produce one map for every official test image.")
    if len(predicted_categories) != EXPECTED_TEST_IMAGES:
        raise PilotError("Category router did not produce a route for every test image.")
    samples = _group_test_samples_after_inference(test_paths, args.data_root)
    del dinomaly, patch_encoder, patch_detectors
    torch.cuda.empty_cache()

    masks, image_labels = _load_test_masks(samples, runtime)
    predicted_array = runtime["np"].asarray(predicted_categories)
    rejected_array = runtime["np"].asarray(rejected_routes, dtype=bool)
    route_threshold_arrays = {
        model_name: _routed_thresholds(predicted_categories, thresholds, model_name, runtime["np"])
        for model_name in ("dinomaly", "patchcore")
    }
    grouped = {category: [i for i, sample in enumerate(samples) if sample.category == category] for category in CATEGORY_NAMES}
    per_category: dict[str, dict[str, Any]] = {"dinomaly": {}, "patchcore": {}}
    csv_rows: list[dict[str, Any]] = []
    map_dir = args.output_dir / "score_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    global_thresholds = thresholds["primary_global"]
    for category, indices in grouped.items():
        idx = runtime["np"].asarray(indices, dtype=int)
        category_samples = [samples[i] for i in indices]
        category_masks, category_labels = masks[idx], image_labels[idx]
        category_routes, category_rejected = predicted_array[idx], rejected_array[idx]
        category_dino, category_patch = dino_maps[idx], patch_maps[idx]
        image_ids = runtime["np"].asarray([sample.image_path.relative_to(args.data_root).as_posix() for sample in category_samples])
        runtime["np"].savez_compressed(
            map_dir / f"{category}.npz",
            image_ids=image_ids,
            image_labels=category_labels,
            masks=category_masks,
            predicted_categories=category_routes,
            classifier_rejected=category_rejected,
            dinomaly_maps=category_dino,
            patchcore_maps=category_patch,
        )
        for model_name, score_maps in (("dinomaly", category_dino), ("patchcore", category_patch)):
            result = _pixel_metrics(score_maps, category_masks, category_labels, global_thresholds[model_name], runtime)
            routed = _pixel_metrics(
                score_maps, category_masks, category_labels, route_threshold_arrays[model_name][idx], runtime
            )
            _add_routed_metrics(result, routed)
            result.update({
                "map_images": len(category_samples),
                "map_coverage": 1.0,
                "mask_images": len(category_samples),
                "mask_coverage": 1.0,
                "classifier_rejected_images": int(category_rejected.sum()),
                "classifier_route_top1_correct": int(sum(category_routes[i] == category for i in range(len(category_routes)))),
            })
            per_category[model_name][category] = result
            csv_rows.append({"model": model_name, "category": category, **result})

    summary_metrics: dict[str, Any] = {"per_category": per_category, "macro": {}, "micro": {}}
    for model_name, score_maps in (("dinomaly", dino_maps), ("patchcore", patch_maps)):
        category_metrics = per_category[model_name]
        summary_metrics["macro"][model_name] = {"category_count": len(category_metrics), **_aggregate_macro(category_metrics)}
        micro = _pixel_metrics(score_maps, masks, image_labels, global_thresholds[model_name], runtime)
        routed_micro = _pixel_metrics(score_maps, masks, image_labels, route_threshold_arrays[model_name], runtime)
        _add_routed_metrics(micro, routed_micro)
        summary_metrics["micro"][model_name] = micro
        csv_rows.append({"model": model_name, "category": "__macro__", **summary_metrics["macro"][model_name]})
        csv_rows.append({"model": model_name, "category": "__micro__", **micro})

    if any(row["map_images"] != row["mask_images"] for model in per_category.values() for row in model.values()):
        raise PilotError("Map and mask coverage differ.")
    _write_heatmap_gallery(samples, dino_maps, patch_maps, masks, args.data_root, args.output_dir / "heatmap_gallery.png", runtime)
    chart_paths = _write_metric_charts(summary_metrics, args.output_dir)
    route_correct = sum(predicted_categories[index] == sample.category for index, sample in enumerate(samples))
    summary = {
        "status": "completed_exploratory_feasibility_pilot",
        "protocol": "original CVPR 2025 Dinomaly ViTill-Small vs project category-specific raw PatchCore with saved top-1 classifier routing; localization only",
        "candidate_identity": "Dinomaly from guojiajeremy/Dinomaly at pinned commit; not Dinomaly2",
        "manifest_sha256": manifest_digest,
        "decoded_pixel_audit": provenance["dataset_inventory_audit"],
        "official_test_images": EXPECTED_TEST_IMAGES,
        "coverage": {
            "dinomaly_maps": f"{len(dino_maps)}/{EXPECTED_TEST_IMAGES}",
            "patchcore_maps": f"{len(patch_maps)}/{EXPECTED_TEST_IMAGES}",
            "masks": f"{len(masks)}/{EXPECTED_TEST_IMAGES}",
        },
        "routing": {
            "route_source": "saved project classifier raw argmax; predicted category used even if the confidence gate rejects",
            "top1_correct_after_inference_for_audit": route_correct,
            "top1_accuracy_after_inference_for_audit": route_correct / len(samples),
            "rejected_images_still_localized": int(rejected_array.sum()),
            "predicted_category_counts": {category: int((predicted_array == category).sum()) for category in CATEGORY_NAMES},
        },
        "thresholds": thresholds,
        "metrics": summary_metrics,
        "provenance_file": "provenance.json",
        "threshold_file": "validation_thresholds.json",
        "score_maps_directory": "score_maps",
        "heatmap_gallery": "heatmap_gallery.png",
        "uncertainty": {"paired_group_bootstrap_ci": None, "reason": "No physical-object/acquisition group IDs are present."},
        "charts": chart_paths,
        "limitations": [
            "Official MVTec test images were viewed in prior project work; results are exploratory.",
            "A 5,000-step run is a feasibility pilot only and cannot be used to reject Dinomaly-Small.",
            "Exact-pixel audit does not detect perceptual near-duplicates or acquisition-session leakage.",
        ],
    }
    provenance["status"] = "completed_exploratory_feasibility_pilot"
    _write_json(args.output_dir / "provenance.json", provenance)
    _write_json(args.output_dir / "summary.json", summary)
    _write_metrics_csv(args.output_dir / "localization_metrics.csv", csv_rows)
    print(json.dumps({
        "summary": str(args.output_dir / "summary.json"),
        "csv": str(args.output_dir / "localization_metrics.csv"),
        "gallery": str(args.output_dir / "heatmap_gallery.png"),
        "charts": [str(args.output_dir / path) for path in chart_paths],
        "coverage": summary["coverage"],
    }, indent=2))


def main() -> int:
    args = _parse_args()
    if args.self_check:
        _self_check()
        return 0
    try:
        _run(args)
    except PilotError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
