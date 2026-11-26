#!/usr/bin/env python3
"""Standalone AnomalyCLIP/relation-map experiment for the 15 MVTec categories.

This script deliberately does not import or modify the production pipeline.  It
fits only GOOD train prototypes, calibrates only on known GOOD validation
images, and evaluates only on the official MVTec test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
METHODS = ("anomalyclip_raw", "relation_map", "fusion")
IMAGE_SIZE = 518
FEATURES_LIST = (6, 12, 18, 24)
DPAM_LAYER = 20
SIGMA = 4.0
QUANTILE = 0.99
FUSION_BETA = 1.0
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def relation_features(attention: Any, features: Any) -> np.ndarray:
    """Return M=A@F for patch tokens, with no class token included."""

    attention_array = np.asarray(attention, dtype=np.float32)
    feature_array = np.asarray(features, dtype=np.float32)
    if attention_array.ndim != 2 or attention_array.shape[0] != attention_array.shape[1]:
        raise ValueError("attention must be a square 2-D patch matrix")
    if feature_array.ndim != 2 or feature_array.shape[0] != attention_array.shape[1]:
        raise ValueError("features must have one row per attention patch")
    if not np.isfinite(attention_array).all() or not np.isfinite(feature_array).all():
        raise ValueError("attention and features must be finite")
    return attention_array @ feature_array


def relation_scores(attention: Any, features: Any, prototype: Any) -> np.ndarray:
    """Return patch L2 distances from M=A@F to a GOOD prototype."""

    relation = relation_features(attention, features)
    prototype_array = np.asarray(prototype, dtype=np.float32)
    if prototype_array.shape != relation.shape:
        raise ValueError(
            f"prototype shape {prototype_array.shape} does not match relation {relation.shape}"
        )
    if not np.isfinite(prototype_array).all():
        raise ValueError("prototype must be finite")
    return np.linalg.norm(relation - prototype_array, axis=1)


def normalize_map(values: Any) -> np.ndarray:
    """Normalize one map independently to [0, 1], matching the screw helper."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("map must be a finite 2-D array")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def fuse_maps(baseline: Any, relation: Any, beta: float = FUSION_BETA) -> np.ndarray:
    """Fuse normalized per-image maps as beta*raw + relation."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    baseline_array = np.asarray(baseline, dtype=np.float32)
    relation_array = np.asarray(relation, dtype=np.float32)
    if baseline_array.shape != relation_array.shape:
        raise ValueError("baseline and relation maps must have the same shape")
    return (beta * normalize_map(baseline_array) + normalize_map(relation_array)).astype(
        np.float32
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source(root: Path) -> str:
    """Hash source text, excluding the nested git/cache/weight directories."""

    digest = hashlib.sha256()
    suffixes = {".md", ".py", ".txt", ".json", ".yaml", ".yml"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts or "checkpoints" in path.parts:
            continue
        if path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + _sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _read_manifest(path: Path, processed_root: Path) -> dict[str, dict[str, list[Path]]]:
    if not path.is_file():
        raise FileNotFoundError(f"classification manifest is missing: {path}")
    result: dict[str, dict[str, list[Path]]] = defaultdict(lambda: {"train": [], "validation": []})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {"image_path", "category", "split"}
        if not required.issubset(rows.fieldnames or set()):
            raise ValueError(f"manifest must contain {sorted(required)}: {path}")
        for row in rows:
            category = str(row["category"])
            split = str(row["split"])
            if category not in MVTEC_CATEGORIES:
                raise ValueError(f"unexpected category in manifest: {category}")
            if split not in ("train", "validation"):
                raise ValueError(f"unexpected manifest split: {split}")
            relative = Path(str(row["image_path"]).replace("\\", "/"))
            image = (processed_root / relative).resolve()
            expected = (processed_root / category / "train" / "good").resolve()
            if image.parent != expected or not image.is_file():
                raise ValueError(f"manifest image is not a train/good file: {relative}")
            result[category][split].append(image)
    for category in MVTEC_CATEGORIES:
        if not result[category]["train"] or not result[category]["validation"]:
            raise ValueError(f"manifest needs train and validation GOOD rows for {category}")
    return result


def _discover_test(processed_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for category in MVTEC_CATEGORIES:
        test_root = processed_root / category / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"missing official test directory: {test_root}")
        for defect_type in sorted(path.name for path in test_root.iterdir() if path.is_dir()):
            for image in _image_files(test_root / defect_type):
                mask = None
                if defect_type != "good":
                    mask = processed_root / category / "ground_truth" / defect_type / f"{image.stem}_mask.png"
                    if not mask.is_file():
                        raise FileNotFoundError(f"missing official test mask: {mask}")
                samples.append(
                    {
                        "category": category,
                        "defect_type": defect_type,
                        "path": image,
                        "mask": mask,
                        "is_good": defect_type == "good",
                    }
                )
    if not samples:
        raise ValueError("official MVTec test split is empty")
    return samples


def _split_hash_audit(
    manifest: dict[str, dict[str, list[Path]]],
    test: list[dict[str, Any]],
    processed_root: Path,
) -> dict[str, Any]:
    split_paths = {
        "train": [path for category in manifest.values() for path in category["train"]],
        "validation": [path for category in manifest.values() for path in category["validation"]],
        "test": [row["path"] for row in test],
    }
    split_hashes: dict[str, set[str]] = {}
    split_tree_hashes: dict[str, str] = {}
    for split, paths in split_paths.items():
        digest = hashlib.sha256()
        hashes: set[str] = set()
        for path in sorted(paths):
            relative = path.resolve().relative_to(processed_root.resolve()).as_posix()
            content_hash = _sha256_file(path)
            digest.update(relative.encode("utf-8") + b"\0" + content_hash.encode("ascii") + b"\n")
            hashes.add(content_hash)
        split_hashes[split] = hashes
        split_tree_hashes[split] = digest.hexdigest()
    test_masks = [row["mask"] for row in test if row.get("mask") is not None]
    mask_digest = hashlib.sha256()
    for path in sorted(test_masks):
        relative = path.resolve().relative_to(processed_root.resolve()).as_posix()
        content_hash = _sha256_file(path)
        mask_digest.update(relative.encode("utf-8") + b"\0" + content_hash.encode("ascii") + b"\n")
    pairs: dict[str, Any] = {}
    overlaps: list[str] = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = sorted(split_hashes[left].intersection(split_hashes[right]))
        overlaps.extend(f"{left}:{right}:{value}" for value in shared)
        pairs[f"{left}_vs_{right}"] = {
            "left_count": len(split_hashes[left]),
            "right_count": len(split_hashes[right]),
            "overlap_count": len(shared),
            "overlap_sha256": shared,
        }
    if overlaps:
        text = f"Exact byte-hash overlap was found in {len(overlaps)} split-pair/hash entries; treat the affected comparison as contaminated."
    else:
        text = "No exact byte-hash overlap was found for train-vs-validation, train-vs-test, or validation-vs-test; perceptual duplicates are not ruled out."
    return {
        "pairs": pairs,
        "split_tree_sha256": split_tree_hashes,
        "test_mask_count": len(test_masks),
        "test_masks_tree_sha256": mask_digest.hexdigest(),
        "overlap_entry_count": len(overlaps),
        "text": text,
    }


def _require_clean_split_hashes(
    manifest: dict[str, dict[str, list[Path]]],
    test: list[dict[str, Any]],
    processed_root: Path,
) -> dict[str, Any]:
    audit = _split_hash_audit(manifest, test, processed_root)
    if audit["overlap_entry_count"]:
        raise RuntimeError(audit["text"])
    return audit


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("L").resize(size, Image.Resampling.NEAREST)
        return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def _resize_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


class _AnomalyCLIPRunner:
    """Minimal copy of the screw session with hooks registered only on last 5 blocks."""

    def __init__(self, repository: Path, checkpoint: Path, clip_cache: Path, device: str) -> None:
        self.repository = repository
        self.checkpoint = checkpoint
        self.clip_cache = clip_cache
        self.requested_device = device
        self.device: str | None = None
        self.model: Any | None = None
        self.preprocess: Any | None = None
        self.text_features: Any | None = None
        self.library: Any | None = None
        self.torch: Any | None = None
        self.attention: dict[int, Any] = {}
        self.hooks: list[Any] = []
        self.captured_layer_indices: tuple[int, ...] = ()

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from PIL import Image  # noqa: F401
            from scipy.ndimage import gaussian_filter  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on the training environment
            raise RuntimeError("AnomalyCLIP runtime needs torch, torchvision, Pillow and scipy") from exc
        if not self.repository.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError("AnomalyCLIP repository or checkpoint is missing")
        self.clip_cache.mkdir(parents=True, exist_ok=True)
        device = self.requested_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        sys.path.insert(0, str(self.repository))
        compatibility = types.ModuleType("pkg_resources")
        import packaging

        compatibility.packaging = packaging
        sys.modules.setdefault("pkg_resources", compatibility)
        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

        parameters = {
            "Prompt_length": 12,
            "learnabel_text_embedding_depth": 9,
            "learnabel_text_embedding_length": 4,
        }
        model, _ = AnomalyCLIP_lib.load(
            "ViT-L/14@336px",
            device=device,
            design_details=parameters,
            download_root=str(self.clip_cache),
        )
        model.eval()
        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
        try:
            state = torch.load(self.checkpoint, map_location=device, weights_only=False)
        except TypeError:  # older torch
            state = torch.load(self.checkpoint, map_location=device)
        prompt_learner.load_state_dict(state["prompt_learner"])
        prompt_learner.to(device).eval()
        model.to(device).eval()
        model.visual.DAPM_replace(DPAM_layer=DPAM_LAYER)
        block_count = len(model.visual.transformer.resblocks)
        selected = tuple(range(max(0, block_count - 5), block_count))
        self.captured_layer_indices = selected
        for index in selected:
            attention = model.visual.transformer.resblocks[index].attn
            if not hasattr(attention, "qkv"):
                raise RuntimeError(f"AnomalyCLIP block {index} has no qkv module")
            self.hooks.append(attention.register_forward_pre_hook(self._make_capture(index)))
        prompts, tokenized, compound = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(prompts, tokenized, compound).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        args = types.SimpleNamespace(image_size=IMAGE_SIZE)
        self.preprocess, _ = get_transform(args)
        self.model = model
        self.library = AnomalyCLIP_lib
        self.torch = torch
        self.gaussian_filter = gaussian_filter

    def _make_capture(self, index: int):
        def capture(module: Any, inputs: tuple[Any, ...]) -> None:
            torch = self.torch
            if torch is None:
                raise RuntimeError("torch is not initialized")
            x = inputs[0]
            batch, tokens, channels = x.shape
            qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, channels // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, key = qkv[0], qkv[1]
            self.attention[index] = ((q @ key.transpose(-2, -1)) * module.scale).softmax(dim=-1).detach()

        return capture

    def infer(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._load()
        assert self.model is not None and self.preprocess is not None and self.text_features is not None and self.torch is not None
        from PIL import Image

        torch = self.torch
        self.attention.clear()
        with Image.fromarray(crop.astype(np.uint8), mode="RGB") as image:
            pixels = self.preprocess(image).reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
        with torch.inference_mode():
            image_features, patch_features = self.model.encode_image(
                pixels, list(FEATURES_LIST), DPAM_layer=DPAM_LAYER
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            _probabilities = (
                image_features @ self.text_features.permute(0, 2, 1) / 0.07
            ).softmax(-1)[:, 0, 1]
            maps = []
            for index, patch_feature in enumerate(patch_features):
                if index >= 0:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = self.library.compute_similarity(patch_feature, self.text_features[0])
                    similarity_map = self.library.get_similarity_map(similarity[:, 1:, :], IMAGE_SIZE)
                    maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
            if not maps:
                raise RuntimeError("AnomalyCLIP produced no selected patch maps")
            raw = torch.stack(maps).sum(dim=0)[0].float().cpu().numpy()
            if tuple(sorted(self.attention)) != self.captured_layer_indices:
                raise RuntimeError("AnomalyCLIP did not capture exactly the last five q-k layers")
            attention = torch.stack([self.attention[index].mean(dim=1)[0] for index in self.captured_layer_indices]).mean(dim=0)
            features = patch_features[-1][0]
            if features.shape[0] != attention.shape[0]:
                raise RuntimeError(f"relation token mismatch: features={features.shape} attention={attention.shape}")
            attention = attention[1:, 1:].float().cpu().numpy()
            features = features[1:].float().cpu().numpy()
        return self.gaussian_filter(raw, sigma=SIGMA).astype(np.float32), attention, features


def _maps(runner: _AnomalyCLIPRunner, image: np.ndarray, prototype: np.ndarray) -> dict[str, np.ndarray]:
    raw, attention, features = runner.infer(image)
    scores = relation_scores(attention, features, prototype)
    side = math.isqrt(int(scores.size))
    if side * side != scores.size:
        raise ValueError(f"relation patch count is not square: {scores.size}")
    relation = _resize_map(scores.reshape(side, side), (IMAGE_SIZE, IMAGE_SIZE))
    return {"anomalyclip_raw": raw, "relation_map": relation, "fusion": fuse_maps(raw, relation)}


def _build_prototypes(runner: _AnomalyCLIPRunner, manifest: dict[str, dict[str, list[Path]]]) -> dict[str, np.ndarray]:
    prototypes: dict[str, np.ndarray] = {}
    for category in MVTEC_CATEGORIES:
        total: np.ndarray | None = None
        count = 0
        for path in manifest[category]["train"]:
            _, attention, features = runner.infer(_load_rgb(path))
            relation = relation_features(attention, features)
            total = relation.copy() if total is None else total + relation
            count += 1
        if total is None or count == 0:
            raise ValueError(f"no GOOD train images for {category}")
        prototypes[category] = (total / count).astype(np.float32)
    return prototypes


def _thresholds(runner: _AnomalyCLIPRunner, manifest: dict[str, dict[str, list[Path]]], prototypes: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for category in MVTEC_CATEGORIES:
        values: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        for path in manifest[category]["validation"]:
            maps = _maps(runner, _load_rgb(path), prototypes[category])
            for method in METHODS:
                values[method].append(maps[method].reshape(-1))
        thresholds[category] = {
            method: float(np.quantile(np.concatenate(values[method]), QUANTILE))
            for method in METHODS
        }
    return thresholds


def _binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    predictions = scores >= threshold
    truth = labels.astype(bool)
    tp = int(np.logical_and(predictions, truth).sum())
    tn = int(np.logical_and(~predictions, ~truth).sum())
    fp = int(np.logical_and(predictions, ~truth).sum())
    fn = int(np.logical_and(~predictions, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    if np.unique(labels).size < 2:
        pixel_auroc = None
        pixel_ap = None
    else:
        pixel_auroc = float(roc_auc_score(labels, scores))
        pixel_ap = float(average_precision_score(labels, scores))
    return {
        "pixel_ap": pixel_ap,
        "pixel_auroc": pixel_auroc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _category_metrics(
    category: str,
    method: str,
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    good_predictions: list[np.ndarray],
    threshold: float,
    image_count: int,
    mask_images: int,
) -> dict[str, Any]:
    score_array = np.concatenate(scores).astype(np.float32, copy=False)
    label_array = np.concatenate(labels).astype(np.uint8, copy=False)
    if np.unique(label_array).size != 2:
        raise ValueError(f"{category} test masks must contain both GOOD and DEFECT pixels")
    report = _binary_metrics(score_array, label_array, threshold)
    good_pixel_count = sum(array.size for array in good_predictions)
    good_fp_pixels = sum(int(array.sum()) for array in good_predictions)
    report.update(
        {
            "scope": "category",
            "method": method,
            "category": category,
            "image_count": image_count,
            "map_images": image_count,
            "mask_images": mask_images,
            "map_coverage": 1.0,
            "mask_coverage": mask_images / image_count if image_count else 0.0,
            "threshold": threshold,
            "good_fp": good_fp_pixels / good_pixel_count if good_pixel_count else None,
            "good_fp_pixels": good_fp_pixels,
            "good_pixel_count": good_pixel_count,
            "pixel_count": int(score_array.size),
        }
    )
    return report


def _aggregate_rows(rows: list[dict[str, Any]], method: str, scope: str) -> dict[str, Any]:
    fields = ("pixel_ap", "pixel_auroc", "precision", "recall", "f1", "dice", "iou", "specificity", "good_fp")
    result: dict[str, Any] = {"scope": scope, "method": method, "category": ""}
    for field in fields:
        values = [row[field] for row in rows if row.get(field) is not None]
        result[field] = float(np.mean(values)) if values else None
    result["image_count"] = int(sum(row["image_count"] for row in rows)) if scope == "micro" else int(len(rows))
    result["macro_category_count"] = int(len(rows)) if scope == "macro" else None
    result["map_images"] = int(sum(row["map_images"] for row in rows)) if scope == "micro" else None
    result["mask_images"] = int(sum(row["mask_images"] for row in rows)) if scope == "micro" else None
    result["map_coverage"] = (result["map_images"] / result["image_count"]) if scope == "micro" and result["image_count"] else (float(np.mean([row["map_coverage"] for row in rows])) if rows else None)
    result["mask_coverage"] = (result["mask_images"] / result["image_count"]) if scope == "micro" and result["image_count"] else (float(np.mean([row["mask_coverage"] for row in rows])) if rows else None)
    result["threshold"] = None
    result["pixel_count"] = int(sum(row["pixel_count"] for row in rows)) if scope == "micro" else None
    result["good_fp_pixels"] = int(sum(row.get("good_fp_pixels", 0) for row in rows)) if scope == "micro" else None
    result["good_pixel_count"] = int(sum(row.get("good_pixel_count", 0) for row in rows)) if scope == "micro" else None
    if scope == "micro":
        for field in ("tp", "tn", "fp", "fn"):
            result[field] = int(sum(row[field] for row in rows))
        tp, tn, fp, fn = result["tp"], result["tn"], result["fp"], result["fn"]
        result["precision"] = tp / (tp + fp) if tp + fp else 0.0
        result["recall"] = tp / (tp + fn) if tp + fn else 0.0
        result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"]) if result["precision"] + result["recall"] else 0.0
        result["dice"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        result["iou"] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        result["specificity"] = tn / (tn + fp) if tn + fp else 0.0
        result["good_fp"] = result["good_fp_pixels"] / result["good_pixel_count"] if result["good_pixel_count"] else None
    else:
        result.update({"tp": None, "tn": None, "fp": None, "fn": None})
    return result


def _write_gallery(path: Path, samples: list[dict[str, Any]]) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(samples), 4, squeeze=False, figsize=(14, 4 * len(samples)))
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["image"])
        axes[row, 0].set_title(f"{sample['category']} / {sample['defect_type']}")
        axes[row, 0].axis("off")
        for column, method in enumerate(METHODS, start=1):
            axes[row, column].imshow(sample["image"])
            axes[row, column].imshow(sample["maps"][method], cmap="jet", alpha=0.48, vmin=0.0, vmax=float(np.max(sample["maps"][method])) or 1.0)
            axes[row, column].set_title(method)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    settings = summary["settings"]
    provenance = summary["provenance"]
    path.write_text(
        "# AnomalyCLIP relation-map benchmark\n\n"
        "This is an isolated experiment; it does not change the production pipeline or classifier.\n\n"
        f"- Categories: all 15 MVTec categories\n- Inference image size: {settings['image_size']}\n"
        f"- Features: `{settings['features_list']}`; DPAM layer: `{settings['dpam_layer']}`; sigma: `{settings['sigma']}`\n"
        f"- Relation: `M=A@F`, patch L2 distance to a category prototype\n- Fusion: `fuse_maps`, per-image normalization, beta={settings['fusion_beta']}\n"
        f"- Prototype fit: manifest `train` GOOD only\n- Threshold calibration: manifest `validation` GOOD only, q={settings['validation_quantile']}\n"
        f"- Evaluation: official MVTec `test` only; metric size `{settings['metric_size']}`\n- CLIP cache: `{provenance['clip_cache']}`\n\n"
        "## Localization scope\n\n"
        "This is an **oracle-category localization ablation**: each test image uses its ground-truth category to select that category's GOOD prototype. It isolates heatmap quality and is not an end-to-end routed result; classification accuracy is evaluated separately.\n\n"
        "## Frozen outputs\n\n"
        "`anomalyclip_raw` is the upstream AnomalyCLIP map. `relation_map` is evaluated separately. `fusion` is only the fixed map fusion; it does not mix detector rankings with another detector's mask.\n\n"
        "## Provenance and caveats\n\n"
        f"- Manifest SHA-256: `{provenance['manifest_sha256']}`\n- Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`\n- Source SHA-256: `{provenance['source_sha256']}`\n"
        f"- Experiment script SHA-256: `{provenance['experiment_script_sha256']}`\n"
        f"- Project revision: `{provenance['project_revision']}`; tracked worktree dirty at capture: `{provenance['project_worktree_dirty']}`\n"
        f"- {provenance['exact_hash_caveat']['text']}\n"
        f"- Processed split digests: {provenance['exact_hash_caveat']['split_tree_sha256']}\n"
        f"- Test mask count/digest: {provenance['exact_hash_caveat']['test_mask_count']} / {provenance['exact_hash_caveat']['test_masks_tree_sha256']}\n"
        f"- Upstream citation: `{provenance['upstream_readme']}:{provenance['upstream_readme_lines']}` states that the MVTec evaluation checkpoint is trained on VisA; `train.sh:11-14` maps `9_12_4_multiscale` to `--dataset visa`.\n"
        "- MVTec AD's official test split is public and has been exposed in prior research; these results are exploratory benchmark evidence, not a private production holdout.\n"
        "- Official test masks are loaded only after prototypes and GOOD-only thresholds are fixed; they are never used for fitting or tuning.\n"
        "- Pixel metrics are computed after resizing the 518px output map to the processed 320px metric grid; the inference model remains fixed at 518px.\n"
        "- The requested checkpoint's upstream training provenance is recorded as supplied; this script does not infer or rewrite its training dataset.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    processed_root = Path(args.processed_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    repository = Path(args.repository).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    clip_cache = Path(args.clip_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.metric_size <= 0:
        raise ValueError("metric-size must be positive")
    manifest = _read_manifest(manifest_path, processed_root)
    if not repository.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("AnomalyCLIP repository or checkpoint is missing")
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    manifest_sha256 = _sha256_file(manifest_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    source_sha256 = _sha256_source(repository)
    experiment_script_sha256 = _sha256_file(script_path)
    project_revision = subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True).strip()
    project_worktree_dirty = bool(subprocess.check_output(["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    source_revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip() if (repository / ".git").is_dir() else None
    if args.smoke:
        runner = _AnomalyCLIPRunner(repository, checkpoint, clip_cache, args.device)
        sample = manifest[MVTEC_CATEGORIES[0]]["train"][0]
        started = time.perf_counter()
        raw, attention, features = runner.infer(_load_rgb(sample))
        elapsed = time.perf_counter() - started
        result = {
            "mode": "smoke",
            "image": str(sample),
            "elapsed_seconds": elapsed,
            "raw_shape": list(raw.shape),
            "relation_shape": list(relation_features(attention, features).shape),
            "captured_attention_layer_indices": list(runner.captured_layer_indices),
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "source_sha256": source_sha256,
            "experiment_script_sha256": experiment_script_sha256,
            "project_revision": project_revision,
            "project_worktree_dirty": project_worktree_dirty,
            "source_revision": source_revision,
            "clip_cache": str(clip_cache),
            "exact_hash_audit": "not run in smoke mode",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    test = _discover_test(processed_root)
    if sorted({row["category"] for row in test}) != list(MVTEC_CATEGORIES):
        raise ValueError("official test data must contain exactly the 15 MVTec categories")
    split_audit = _require_clean_split_hashes(manifest, test, processed_root)
    provenance = {
        "processed_root": str(processed_root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_repository": str(repository),
        "source_revision": source_revision,
        "source_sha256": source_sha256,
        "experiment_script": str(script_path),
        "experiment_script_sha256": experiment_script_sha256,
        "project_revision": project_revision,
        "project_worktree_dirty": project_worktree_dirty,
        "clip_cache": str(clip_cache),
        "localization_protocol": "oracle_category_localization",
        "upstream_readme": str(repository / "README.md"),
        "upstream_readme_lines": "110",
        "evaluation_status": "exploratory_public_official_test",
        "exact_hash_caveat": split_audit,
    }
    runner = _AnomalyCLIPRunner(repository, checkpoint, clip_cache, args.device)
    prototypes = _build_prototypes(runner, manifest)
    thresholds = _thresholds(runner, manifest, prototypes)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "image_size": IMAGE_SIZE,
        "features_list": list(FEATURES_LIST),
        "dpam_layer": DPAM_LAYER,
        "sigma": SIGMA,
        "validation_quantile": QUANTILE,
        "fusion_beta": FUSION_BETA,
        "metric_size": args.metric_size,
        "device": args.device,
        "clip_cache": str(clip_cache),
        "captured_attention_layers": 5,
        "captured_attention_layer_indices": list(runner.captured_layer_indices),
    }
    category_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    gallery: list[dict[str, Any]] = []
    metric_width = metric_height = args.metric_size
    total_pixels = len(test) * metric_width * metric_height
    with tempfile.TemporaryDirectory(prefix="anomalyclip-relation-") as temp_dir:
        score_maps = {
            method: np.memmap(Path(temp_dir) / f"{method}.f32", dtype=np.float32, mode="w+", shape=(total_pixels,))
            for method in METHODS
        }
        label_map = np.memmap(Path(temp_dir) / "labels.u8", dtype=np.uint8, mode="w+", shape=(total_pixels,))
        offset = 0
        for category in MVTEC_CATEGORIES:
            category_samples = [row for row in test if row["category"] == category]
            local_scores: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
            local_labels: list[np.ndarray] = []
            local_good: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
            mask_images = 0
            for row in category_samples:
                image = _load_rgb(row["path"])
                maps = _maps(runner, image, prototypes[category])
                mask = _load_mask(row["mask"], (metric_width, metric_height))
                if row["mask"] is not None:
                    mask_images += 1
                labels = mask.reshape(-1)
                for method in METHODS:
                    values = _resize_map(maps[method], (metric_width, metric_height)).reshape(-1)
                    local_scores[method].append(values)
                    score_maps[method][offset : offset + values.size] = values
                    if row["is_good"]:
                        local_good[method].append(values >= thresholds[category][method])
                local_labels.append(labels)
                label_map[offset : offset + labels.size] = labels
                offset += labels.size
                if len(gallery) < args.gallery_samples and not row["is_good"] and mask.any():
                    gallery.append({"category": category, "defect_type": row["defect_type"], "image": image, "maps": maps})
            for method in METHODS:
                category_rows[method].append(
                    _category_metrics(category, method, local_scores[method], local_labels, local_good[method], thresholds[category][method], len(category_samples), mask_images)
                )
        if offset != total_pixels:
            raise RuntimeError(f"pixel writer offset mismatch: {offset} != {total_pixels}")
        metrics: dict[str, Any] = {"per_category": {}, "micro": {}, "macro": {}}
        for method in METHODS:
            metrics["per_category"][method] = category_rows[method]
            all_scores = np.asarray(score_maps[method])
            all_labels = np.asarray(label_map)
            ranking = _binary_metrics(all_scores, all_labels, 0.0)
            metrics["micro"][method] = _aggregate_rows(category_rows[method], method, "micro")
            metrics["micro"][method]["pixel_ap"] = ranking["pixel_ap"]
            metrics["micro"][method]["pixel_auroc"] = ranking["pixel_auroc"]
            metrics["macro"][method] = _aggregate_rows(category_rows[method], method, "macro")
        for mmap in score_maps.values():
            mmap.flush()
        label_map.flush()
    summary = {
        "schema_version": 1,
        "settings": settings,
        "provenance": provenance,
        "categories": list(MVTEC_CATEGORIES),
        "thresholds": thresholds,
        "metrics": metrics,
        "outputs": {"metrics_csv": "metrics.csv", "summary_json": "summary.json", "readme": "README.md", "gallery": "heatmap_gallery.png"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    fields = ["scope", "method", "category", "image_count", "macro_category_count", "map_images", "mask_images", "map_coverage", "mask_coverage", "threshold", "pixel_ap", "pixel_auroc", "precision", "recall", "f1", "dice", "iou", "specificity", "good_fp", "good_fp_pixels", "good_pixel_count", "pixel_count", "tp", "tn", "fp", "fn"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            writer.writerows(metrics["per_category"][method])
            writer.writerow(metrics["micro"][method])
            writer.writerow(metrics["macro"][method])
    _write_readme(output_dir / "README.md", summary)
    _write_gallery(output_dir / "heatmap_gallery.png", gallery)
    return summary


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default=str(root / "datasets/mvtec_ad_processed_320"))
    parser.add_argument("--manifest", default=str(root / "artifacts/production_highres/mvtec_ad/variants/goal_15_exact_patch_knn_320/mvtec_ad/classification_split.csv"))
    parser.add_argument("--repository", default=str(root / "artifacts/external/anomalyclip"))
    parser.add_argument("--checkpoint", default=str(root / "artifacts/external/anomalyclip/checkpoints/9_12_4_multiscale/epoch_15.pth"))
    parser.add_argument("--clip-cache", default=str(root / "artifacts/external/clip"))
    parser.add_argument("--output-dir", default=str(root / "reports/experiments/anomalyclip_relation_518"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--metric-size", type=int, default=320)
    parser.add_argument("--gallery-samples", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="Run one manifest train image and print runtime/shapes without writing benchmark outputs.")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
