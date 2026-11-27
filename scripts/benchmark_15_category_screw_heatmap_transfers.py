"""Benchmark Screw-branch heatmap methods on all MVTec AD categories.

The unsupervised methods use the complete official test split. Supervised
refiners use a deterministic image-disjoint development split carved from the
official test anomalies; their scores are therefore reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn

from mvtec_system.anomaly.detectors import ExactKNNDetector, PaDiMDetector, PatchCoreDetector
from mvtec_system.anomaly.maps import make_anomaly_maps
from mvtec_system.anomaly.multi_layer import fit_padim_statistics, fused_patch_features
from mvtec_system.anomaly.refinement import (
    MASK_POSTPROCESS_METHODS,
    postprocess_mask,
    refine_heatmap,
)
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
SCORE_METHODS = (
    "patchcore_raw", "patchcore_open3", "patchcore_open5", "patchcore_open7",
    "patchcore_open9", "exact_knn_k1", "exact_knn_k3", "exact_knn_k5", "padim_320",
)
MORPHOLOGY = {
    "patchcore_raw": "raw",
    "patchcore_open3": "heatmap_open3",
    "patchcore_open5": "heatmap_open5",
    "patchcore_open7": "heatmap_open7",
    "patchcore_open9": "heatmap_open9",
}
MASK_METHODS = MASK_POSTPROCESS_METHODS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )


def _classification_manifest(root: Path, output: Path, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category in CATEGORIES:
        images = _images(root / category / "train" / "good")
        shuffled = list(images)
        random.Random(seed + sum(map(ord, category))).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * 0.30))
        validation_count = min(validation_count, len(shuffled) - 1) if len(shuffled) > 1 else 0
        validation_paths = set(shuffled[:validation_count])
        for image in images:
            rows.append({
                "image_path": image.relative_to(root).as_posix(),
                "category": category,
                "split": "validation" if image in validation_paths else "train",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "category", "split"))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _defect_records(root: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for category in CATEGORIES:
        test_root = root / category / "test"
        for folder in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name != "good"):
            images = _images(folder)
            shuffled = list(images)
            group_seed = seed + sum(map(ord, category + folder.name))
            random.Random(group_seed).shuffle(shuffled)
            n = len(shuffled)
            test_count = max(1, round(n * 0.15)) if n >= 3 else (1 if n == 2 else 0)
            validation_count = max(1, round(n * 0.15)) if n >= 3 else 0
            if n == 1:
                test_count = 1
            if test_count + validation_count >= n and n >= 3:
                validation_count = max(1, n - test_count - 1)
            split_by_path: dict[Path, str] = {}
            for path in shuffled[: n - test_count - validation_count]:
                split_by_path[path] = "train"
            for path in shuffled[n - test_count - validation_count : n - test_count]:
                split_by_path[path] = "validation"
            for path in shuffled[n - test_count :]:
                split_by_path[path] = "test"
            for image in images:
                defect_type = folder.name
                mask = root / category / "ground_truth" / defect_type / f"{image.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(f"Missing ground-truth mask: {mask}")
                records.append({
                    "category": category,
                    "defect_type": defect_type,
                    "image": image,
                    "mask": mask,
                    "custom_split": split_by_path[image],
                    "is_defect": True,
                })
            counts[f"{category}/{folder.name}"] = n
    return records, counts


def _mask(record: dict[str, Any], size: int) -> np.ndarray:
    if not record["is_defect"]:
        return np.zeros((size, size), dtype=np.uint8)
    with Image.open(record["mask"]) as image:
        image.load()
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    if values.shape != (size, size):
        raise ValueError(f"Mask geometry mismatch for {record['mask']}: {values.shape}")
    return (values > 0).astype(np.uint8)


def _patch_inputs(features: np.ndarray, heatmap: np.ndarray, side: int, mode: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (side * side, values.shape[-1]):
        raise ValueError("DINO patch feature grid is malformed")
    scores = cv2.resize(np.asarray(heatmap, dtype=np.float32), (side, side), interpolation=cv2.INTER_AREA)
    if mode == "default":
        return np.concatenate((values, scores.reshape(-1, 1)), axis=1)
    smooth = cv2.GaussianBlur(scores, (3, 3), 0)
    local_max = cv2.dilate(scores, np.ones((3, 3), np.uint8))
    local_min = cv2.erode(scores, np.ones((3, 3), np.uint8))
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    xy = np.stack((xx / max(side - 1, 1), yy / max(side - 1, 1)), axis=-1)
    context = np.concatenate((
        scores[..., None], smooth[..., None], local_max[..., None], local_min[..., None],
        (scores - smooth)[..., None], xy,
    ), axis=-1).reshape(-1, 7)
    return np.concatenate((values, context), axis=1)


def _patch_labels(mask: np.ndarray, side: int) -> np.ndarray:
    return (cv2.resize(mask.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8).reshape(-1)


def _sample_rows(inputs: np.ndarray, labels: np.ndarray, rng: np.random.Generator, per_class: int = 64) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    chunks_x, chunks_y = [], []
    for values, label in ((negative, 0), (positive, 1)):
        if not len(values):
            continue
        chosen = rng.choice(values, size=min(per_class, len(values)), replace=False)
        chunks_x.append(inputs[chosen])
        chunks_y.append(np.full(len(chosen), label, dtype=np.uint8))
    if not chunks_x:
        raise ValueError("No patch samples available for the supervised refiner")
    return np.concatenate(chunks_x), np.concatenate(chunks_y)


def _quantile(maps: list[np.ndarray], q: float = 0.99) -> float:
    if not maps:
        raise ValueError("Cannot calibrate without known-good validation maps")
    return float(np.quantile(np.concatenate([item.reshape(-1) for item in maps]), q))


def _knn_triplet(queries: torch.Tensor, bank: torch.Tensor, chunk_size: int = 8192) -> dict[int, torch.Tensor]:
    """Scan the exact bank once and return mean distances for k=1, 3 and 5."""
    bank = bank.float().cpu()
    queries = queries.float().cpu()
    outputs = {1: [], 3: [], 5: []}
    for start in range(0, len(queries), 128):
        query = queries[start : start + 128]
        best = torch.full((len(query), 5), float("inf"), dtype=torch.float32)
        for bank_start in range(0, len(bank), chunk_size):
            distances = torch.cdist(query, bank[bank_start : bank_start + chunk_size])
            best = torch.cat((best, distances), dim=1).topk(5, dim=1, largest=False).values
        for k in outputs:
            outputs[k].append(best[:, :k].mean(dim=1))
    return {k: torch.cat(parts) for k, parts in outputs.items()}


class _SpatialDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.GroupNorm(1, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.GroupNorm(1, 16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def _decoder_input(features: np.ndarray, heatmap: np.ndarray, side: int) -> np.ndarray:
    values = _patch_inputs(features, heatmap, side, "default")
    return values.reshape(side, side, -1).transpose(2, 0, 1).astype(np.float32)


def _fit_decoder(train_x: list[np.ndarray], train_y: list[np.ndarray], val_x: list[np.ndarray], val_y: list[np.ndarray], device: torch.device, epochs: int) -> tuple[_SpatialDecoder, int]:
    if not train_x or not val_x:
        raise ValueError("Spatial decoder requires train and validation images")
    torch.manual_seed(42)
    x_train = torch.from_numpy(np.stack(train_x))
    y_train = torch.from_numpy(np.stack(train_y)).float()
    x_val = torch.from_numpy(np.stack(val_x)).to(device)
    y_val = torch.from_numpy(np.stack(val_y)).float().to(device)
    positives = float(y_train.sum())
    negatives = float(y_train.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("Spatial decoder training requires defect and normal patches")
    model = _SpatialDecoder(x_train.shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(30.0, negatives / positives), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    generator = torch.Generator().manual_seed(42)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 8):
            indices = order[start : start + 8]
            inputs = x_train[indices].to(device)
            targets = y_train[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            probabilities = torch.sigmoid(logits)
            bce = loss_fn(logits, targets)
            intersection = (probabilities * targets).sum(dim=(1, 2, 3))
            dice = 1 - ((2 * intersection + 1) / (probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1)).mean()
            (bce + dice).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x_val)
            probs = torch.sigmoid(logits)
            val_bce = loss_fn(logits, y_val)
            inter = (probs * y_val).sum(dim=(1, 2, 3))
            val_dice = 1 - ((2 * inter + 1) / (probs.sum(dim=(1, 2, 3)) + y_val.sum(dim=(1, 2, 3)) + 1)).mean()
            value = float((val_bce + val_dice).item())
        if value < best_loss:
            best_loss, best_epoch, stale = value, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError("Spatial decoder did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_epoch


def _predict_decoder(model: _SpatialDecoder, features: np.ndarray, heatmap: np.ndarray, side: int, output_size: int, device: torch.device) -> np.ndarray:
    inputs = torch.from_numpy(_decoder_input(features, heatmap, side)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(inputs)).squeeze().detach().cpu().numpy()
    return cv2.resize(probability.astype(np.float32), (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def _confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "negative_pixel_fpr": fp / (tn + fp) if tn + fp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _rank(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "pixel_ap": float(average_precision_score(labels.reshape(-1), scores.reshape(-1))),
        "pixel_auroc": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1))),
    }


def _thresholded_mask(raw: np.ndarray, threshold: float, method: str) -> np.ndarray:
    return postprocess_mask(raw, threshold, method)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _feature_pass(paths: list[Path], encoder: SharedDINOv2, batch_size: int) -> list[Any]:
    outputs = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([image_tensor(path, encoder.config) for path in paths[start : start + batch_size]])
        outputs.append(encoder.forward(batch))
    return outputs


def _score_batch(features: Any, patchcore: PatchCoreDetector, padim: PaDiMDetector, bank: ExactKNNDetector, encoder: SharedDINOv2) -> list[dict[str, Any]]:
    with torch.inference_mode():
        patch_results = patchcore.predict_batch(features)
        padim_results = padim.predict_batch(features)
    final_features = features.layer_patch_embeddings[11]
    rows: list[dict[str, Any]] = []
    for index, (patch_result, padim_result) in enumerate(zip(patch_results, padim_results)):
        patch_tokens = final_features[index].detach().cpu()
        distances = _knn_triplet(patch_tokens, bank.bank)
        maps = {
            "patchcore_raw": patch_result.anomaly_map.astype(np.float32),
            "padim_320": padim_result.anomaly_map.astype(np.float32),
        }
        for k, scores in distances.items():
            maps[f"exact_knn_k{k}"] = make_anomaly_maps(
                scores.reshape(1, -1), encoder.patch_grid, encoder.padded_size, encoder.config.image_size
            )[0]
        for method in ("patchcore_open3", "patchcore_open5", "patchcore_open7", "patchcore_open9"):
            maps[method] = refine_heatmap(maps["patchcore_raw"], MORPHOLOGY[method])
        rows.append({
            "maps": maps,
            "features": patch_tokens.numpy().astype(np.float32),
            "raw_map": maps["patchcore_raw"],
        })
    return rows


def _record(category: str, image: Path, is_defect: bool, defect_type: str, custom_split: str | None = None, mask: Path | None = None) -> dict[str, Any]:
    return {
        "category": category, "image": image, "is_defect": is_defect,
        "defect_type": defect_type, "custom_split": custom_split, "mask": mask,
    }


def _category_metrics(score: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    return {**_rank(score, truth), **_confusion(score, truth, threshold)}


def _self_check() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    patch_labels = _patch_labels(mask, 2)
    assert patch_labels.shape == (4,) and patch_labels.sum() >= 1
    distances = _knn_triplet(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]]),
    )
    assert distances[1][0] < distances[3][0] < distances[5][0]
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[6:10, 6:10] = 1.0
    refined = refine_heatmap(heatmap, "heatmap_open3")
    assert refined.shape == heatmap.shape and np.isfinite(refined).all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/mvtec_ad_processed_320"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production_highres/mvtec_ad"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/experiments/goal_15_category_screw_heatmap_transfers_320"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder-epochs", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--self-check", action="store_true", help="run small shape and scoring checks, then exit")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        print("self-check passed")
        return 0
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite report directory: {args.report_dir}")
    if args.batch_size < 1 or args.decoder_epochs < 1:
        raise ValueError("batch size and decoder epochs must be positive")

    root = args.processed_root.resolve()
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    manifest_path = report / "classification_split.csv"
    manifest = _classification_manifest(root, manifest_path, args.seed)
    manifest_digest = _sha256(manifest_path)
    expected_manifest = "fac2c931624430c83e1a96f2122691cd5ad319c5f984b43c0bccf5ac7029d432"
    if manifest_digest != expected_manifest:
        raise ValueError(f"Recreated train/validation split differs from baseline: {manifest_digest}")

    good_train = {category: [] for category in CATEGORIES}
    good_validation = {category: [] for category in CATEGORIES}
    for row in manifest:
        target = good_train if row["split"] == "train" else good_validation
        target[row["category"]].append(root / row["image_path"])
    defects, defect_counts = _defect_records(root, args.seed)
    defect_split = {(row["category"], row["image"]): row["custom_split"] for row in defects}
    test_records: list[dict[str, Any]] = []
    for category in CATEGORIES:
        folder = root / category / "test"
        for path in _images(folder / "good"):
            test_records.append(_record(category, path, False, "good", "test"))
        for item in defects:
            if item["category"] == category:
                test_records.append(_record(category, item["image"], True, item["defect_type"], item["custom_split"], item["mask"]))
    test_records.sort(key=lambda item: (item["category"], str(item["image"])))
    if len(test_records) != 1725:
        raise ValueError(f"Expected all 1,725 official test images, found {len(test_records)}")
    paths_by_split = {
        "train": {str(path) for values in good_train.values() for path in values} | {str(item["image"]) for item in defects if item["custom_split"] == "train"},
        "validation": {str(path) for values in good_validation.values() for path in values} | {str(item["image"]) for item in defects if item["custom_split"] == "validation"},
        "test": {str(item["image"]) for item in test_records if not item["is_defect"] or item["custom_split"] == "test"},
    }
    if paths_by_split["train"] & paths_by_split["validation"] or paths_by_split["train"] & paths_by_split["test"] or paths_by_split["validation"] & paths_by_split["test"]:
        raise ValueError("An image path crosses the supervised train/validation/test split")
    split_rows = []
    for split, paths in paths_by_split.items():
        for value in sorted(paths):
            path = Path(value)
            split_rows.append({"image_path": str(path.relative_to(root)), "category": path.relative_to(root).parts[0], "split": split})
    _write_csv(report / "supervised_internal_split.csv", split_rows)

    config = DINOv2Config(image_size=(320, 320), feature_layers=(5, 8, 11))
    encoder = SharedDINOv2(config, artifact_root=args.artifact_root, device=args.device)
    thresholds_all: dict[str, dict[str, float]] = {}
    binary_counts: dict[str, dict[str, dict[str, int]]] = {}
    supervised_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    decoder_provenance: dict[str, Any] = {}

    for category_number, category in enumerate(CATEGORIES, start=1):
        print(f"[{category_number}/15] {category}: train normal memory/statistics", flush=True)
        patchcore = PatchCoreDetector().load(args.artifact_root / "variants/goal_15_heatmap_open5_320/detectors/patchcore" / category)
        padim = PaDiMDetector(config=patchcore.config)
        fused_train: list[torch.Tensor] = []
        exact_train: list[torch.Tensor] = []
        train_logit_x = {"default": [], "spatial": []}
        train_logit_y = {"default": [], "spatial": []}
        decoder_train: list[tuple[str, np.ndarray, np.ndarray]] = []
        selected_good = set(random.Random(args.seed + sum(map(ord, category))).sample(
            good_train[category], min(48, len(good_train[category]))
        ))
        for offset in range(0, len(good_train[category]), args.batch_size):
            paths = good_train[category][offset : offset + args.batch_size]
            batch = torch.stack([image_tensor(path, encoder.config) for path in paths])
            features = encoder.forward(batch)
            exact_train.append(features.patch_embeddings.detach().cpu())
            fused_train.append(fused_patch_features(features, patchcore.config.layers).detach().cpu())
            patch_results = patchcore.predict_batch(features)
            for index, (path, result) in enumerate(zip(paths, patch_results)):
                token = features.layer_patch_embeddings[11][index].detach().cpu().numpy()
                raw = result.anomaly_map.astype(np.float32)
                labels = np.zeros(encoder.patch_count, dtype=np.uint8)
                rng = np.random.default_rng(args.seed + offset + index)
                for mode in train_logit_x:
                    x, y = _sample_rows(_patch_inputs(token, raw, encoder.patch_grid[0], mode), labels, rng)
                    train_logit_x[mode].append(x)
                    train_logit_y[mode].append(y)
                if path in selected_good:
                    decoder_train.append((str(path), _decoder_input(token, raw, encoder.patch_grid[0]), np.zeros((1, *encoder.patch_grid), dtype=np.float32)))
            del features, batch
        train_descriptors = torch.cat(fused_train, dim=0)
        padim.means, padim.inverse_covariance, padim.projection = fit_padim_statistics(
            train_descriptors, patchcore.config.reduced_dim, patchcore.config.covariance_regularization, patchcore.config.seed
        )
        padim.category = category
        padim.patch_grid, padim.padded_size, padim.image_size = encoder.patch_grid, encoder.padded_size, encoder.config.image_size
        padim.save(report / "padim_320" / category)
        padim.means = padim.means.to(args.device)
        padim.inverse_covariance = padim.inverse_covariance.to(args.device)
        padim.projection = padim.projection.to(args.device)
        del fused_train, train_descriptors
        bank = ExactKNNDetector(k=1)
        bank.bank = torch.cat(exact_train, dim=0).reshape(-1, exact_train[0].shape[-1]).contiguous().float()
        bank.category = category
        bank.patch_grid, bank.padded_size, bank.image_size = encoder.patch_grid, encoder.padded_size, encoder.config.image_size
        del exact_train

        # Cache all official test tensors/maps for this category so the supervised
        # refiner can use an explicit custom split without another DINO pass.
        print(f"[{category_number}/15] {category}: validation and official test inference", flush=True)
        validation_records = [
            _record(category, path, False, "good", "validation") for path in good_validation[category]
        ] + [item for item in defects if item["category"] == category and item["custom_split"] == "validation"]
        validation_records.sort(key=lambda item: str(item["image"]))
        val_cache: list[dict[str, Any]] = []
        val_good_maps: dict[str, list[np.ndarray]] = {method: [] for method in SCORE_METHODS}
        for offset in range(0, len(validation_records), args.batch_size):
            batch_records = validation_records[offset : offset + args.batch_size]
            batch = torch.stack([image_tensor(item["image"], encoder.config) for item in batch_records])
            features = encoder.forward(batch)
            outputs = _score_batch(features, patchcore, padim, bank, encoder)
            for item, output in zip(batch_records, outputs):
                labels = _mask(item, 320)
                cache = {**output, "record": item, "mask": labels}
                val_cache.append(cache)
                if not item["is_defect"]:
                    for method, score in output["maps"].items():
                        val_good_maps[method].append(score)
            del features, batch
        thresholds = {method: _quantile(values) for method, values in val_good_maps.items()}

        category_test = [item for item in test_records if item["category"] == category]
        test_cache: list[dict[str, Any]] = []
        print(f"[{category_number}/15] {category}: {len(category_test)} test images", flush=True)
        for offset in range(0, len(category_test), args.batch_size):
            batch_records = category_test[offset : offset + args.batch_size]
            batch = torch.stack([image_tensor(item["image"], encoder.config) for item in batch_records])
            features = encoder.forward(batch)
            outputs = _score_batch(features, patchcore, padim, bank, encoder)
            for item, output in zip(batch_records, outputs):
                labels = _mask(item, 320)
                test_cache.append({**output, "record": item, "mask": labels})
            del features, batch

        # Fit the two patch classifiers from training GOOD plus the 70% defect
        # development partition. This is an internal split, not official-test scoring.
        defect_train = [item for item in test_cache if item["record"]["is_defect"] and item["record"]["custom_split"] == "train"]
        if not defect_train:
            raise ValueError(f"No supervised defect training masks for {category}")
        for item in defect_train:
            token = item["features"]
            raw = item["raw_map"]
            patch_labels = _patch_labels(item["mask"], encoder.patch_grid[0])
            rng = np.random.default_rng(args.seed + sum(map(ord, category + str(item["record"]["image"]))))
            for mode in train_logit_x:
                x, y = _sample_rows(_patch_inputs(token, raw, encoder.patch_grid[0], mode), patch_labels, rng)
                train_logit_x[mode].append(x)
                train_logit_y[mode].append(y)
        models = {}
        refiner_dir = report / "supervised_refiners" / category
        refiner_dir.mkdir(parents=True, exist_ok=True)
        for mode in ("default", "spatial"):
            x = np.concatenate(train_logit_x[mode]).astype(np.float32)
            y = np.concatenate(train_logit_y[mode]).astype(np.uint8)
            model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=args.seed)
            model.fit(x, y)
            models[mode] = model
            import joblib
            joblib.dump(model, refiner_dir / f"logistic_{mode}.joblib")

        # Select small, deterministic image groups for the convolutional decoder.
        rng = random.Random(args.seed + 10_000 + sum(map(ord, category)))
        defect_train_rows = [item for item in defect_train]
        decoder_defects = rng.sample(defect_train_rows, min(96, len(defect_train_rows)))
        decoder_train_rows = [row for row in decoder_train]
        decoder_x = [entry[1] for entry in decoder_train_rows]
        decoder_y = [entry[2] for entry in decoder_train_rows]
        for item in decoder_defects:
            decoder_x.append(_decoder_input(item["features"], item["raw_map"], encoder.patch_grid[0]))
            decoder_y.append(_patch_labels(item["mask"], encoder.patch_grid[0]).reshape(1, *encoder.patch_grid).astype(np.float32))
        val_good_for_decoder = [item for item in val_cache if not item["record"]["is_defect"]]
        val_defect_for_decoder = [item for item in val_cache if item["record"]["is_defect"]]
        val_rows_for_decoder = (
            rng.sample(val_good_for_decoder, min(32, len(val_good_for_decoder)))
            + rng.sample(val_defect_for_decoder, min(32, len(val_defect_for_decoder)))
        )
        val_x = [_decoder_input(item["features"], item["raw_map"], encoder.patch_grid[0]) for item in val_rows_for_decoder]
        val_y = [_patch_labels(item["mask"], encoder.patch_grid[0]).reshape(1, *encoder.patch_grid).astype(np.float32) for item in val_rows_for_decoder]
        decoder_device = torch.device(args.device if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
        decoder, decoder_best_epoch = _fit_decoder(decoder_x, decoder_y, val_x, val_y, decoder_device, args.decoder_epochs)
        torch.save({"state_dict": decoder.to("cpu").state_dict(), "channels": len(decoder_x[0]), "side": encoder.patch_grid[0], "best_epoch": decoder_best_epoch}, refiner_dir / "spatial_decoder.pt")
        decoder.to(decoder_device).eval()
        decoder_provenance[category] = {"best_epoch": decoder_best_epoch, "train_images": len(decoder_x), "validation_images": len(val_x)}

        def refiner_map(item: dict[str, Any], mode: str) -> np.ndarray:
            scores = models[mode].predict_proba(_patch_inputs(item["features"], item["raw_map"], encoder.patch_grid[0], mode))[:, 1]
            grid = scores.reshape(encoder.patch_grid)
            return cv2.resize(grid, (320, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        for mode in ("default", "spatial"):
            val_good_maps[f"supervised_{mode}"] = [refiner_map(item, mode) for item in val_cache if not item["record"]["is_defect"]]
            thresholds[f"supervised_{mode}"] = _quantile(val_good_maps[f"supervised_{mode}"])
        val_good_maps["spatial_decoder"] = [
            _predict_decoder(decoder, item["features"], item["raw_map"], encoder.patch_grid[0], 320, decoder_device)
            for item in val_cache if not item["record"]["is_defect"]
        ]
        thresholds["spatial_decoder"] = _quantile(val_good_maps["spatial_decoder"])
        thresholds_all[category] = thresholds

        # Official-test unsupervised binary cleanup comparisons use the raw
        # PatchCore q=.99 validation threshold for all mask-only operators.
        category_binary = {method: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "good_fp_images": 0, "good_images": 0, "good_fp_pixels": 0, "good_pixels": 0} for method in MASK_METHODS}
        supervised_scores = {"patchcore_raw": [], "supervised_default": [], "supervised_spatial": [], "spatial_decoder": []}
        supervised_labels: list[np.ndarray] = []
        custom_test = [item for item in test_cache if not item["record"]["is_defect"] or item["record"]["custom_split"] == "test"]
        custom_test_paths = {str(item["record"]["image"]) for item in custom_test}
        for item in test_cache:
            raw = item["raw_map"]
            truth = item["mask"]
            for method in MASK_METHODS:
                pred = _thresholded_mask(raw, thresholds["patchcore_raw"], method)
                counts = category_binary[method]
                truth_bool = truth.astype(bool)
                counts["tp"] += int(np.count_nonzero(pred & truth_bool))
                counts["fp"] += int(np.count_nonzero(pred & ~truth_bool))
                counts["fn"] += int(np.count_nonzero(~pred & truth_bool))
                counts["tn"] += int(np.count_nonzero(~pred & ~truth_bool))
                if not item["record"]["is_defect"]:
                    counts["good_images"] += 1
                    counts["good_fp_images"] += int(pred.any())
                    counts["good_fp_pixels"] += int(pred.sum())
                    counts["good_pixels"] += truth.size
            if str(item["record"]["image"]) in custom_test_paths:
                supervised_labels.append(truth.reshape(-1))
                supervised_scores["patchcore_raw"].append(raw.reshape(-1))
                supervised_scores["supervised_default"].append(refiner_map(item, "default").reshape(-1))
                supervised_scores["supervised_spatial"].append(refiner_map(item, "spatial").reshape(-1))
                supervised_scores["spatial_decoder"].append(_predict_decoder(decoder, item["features"], raw, encoder.patch_grid[0], 320, decoder_device).reshape(-1))
        binary_counts[category] = category_binary
        custom_truth = np.concatenate(supervised_labels)
        for method, chunks in supervised_scores.items():
            scores = np.concatenate(chunks).astype(np.float32)
            threshold = thresholds.get(method, thresholds["patchcore_raw"])
            report_metrics = _category_metrics(scores, custom_truth, threshold)
            supervised_rows.append({"category": category, "method": method, "images": len(custom_test), **report_metrics, "threshold": threshold})

        # Evaluate every continuous map while this category's arrays are in RAM.
        # This avoids materializing multi-gigabyte all-test score-map files.
        good_test = [item for item in test_cache if not item["record"]["is_defect"]]
        for method in SCORE_METHODS:
            scores = np.concatenate([item["maps"][method].reshape(-1) for item in test_cache])
            truth = np.concatenate([item["mask"].reshape(-1) for item in test_cache])
            threshold = thresholds[method]
            metric = _category_metrics(scores, truth, threshold)
            metric["good_fp_pixel_area"] = (
                float(sum(np.count_nonzero(item["maps"][method] >= threshold) for item in good_test))
                / (len(good_test) * 320 * 320)
                if good_test else 0.0
            )
            all_rows.append({"category": category, "method": method, "images": len(test_cache), "threshold": threshold, **metric})
            del scores, truth

        # Write local artifacts and free per-category caches before advancing.
        import joblib
        del test_cache, val_cache, models, decoder
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"[{category_number}/15] {category}: completed", flush=True)

    print("Summarizing per-category and overall macro metrics", flush=True)
    summary_methods: dict[str, Any] = {}
    for method in SCORE_METHODS:
        category_values = [row for row in all_rows if row["method"] == method]
        summary_methods[method] = {
            "pixel_ap_micro": None,
            "pixel_auroc_micro": None,
            "pixel_ap_macro_15": float(np.mean([row["pixel_ap"] for row in category_values])),
            "pixel_auroc_macro_15": float(np.mean([row["pixel_auroc"] for row in category_values])),
            "precision": float(np.sum([row["tp"] for row in category_values]) / max(1, np.sum([row["tp"] + row["fp"] for row in category_values]))),
            "recall": float(np.sum([row["tp"] for row in category_values]) / max(1, np.sum([row["tp"] + row["fn"] for row in category_values]))),
            "good_fp_pixel_area": float(np.mean([row["good_fp_pixel_area"] for row in category_values])),
            "map_coverage": len(test_records) / len(test_records),
        }
        print(f"{method}: macro AP={summary_methods[method]['pixel_ap_macro_15']:.4f}", flush=True)

    baseline_summary = json.loads(Path("reports/experiments/goal_15_category_heatmap_methods_320/comparison_summary.json").read_text(encoding="utf-8"))
    historical_baselines = {
        "patchcore_raw": "raw_patchcore",
        "patchcore_open5": "open5",
    }
    for method, old_name in historical_baselines.items():
        summary_methods[method]["pixel_ap_micro"] = baseline_summary["methods"][old_name]["pixel_ap_micro"]
        summary_methods[method]["pixel_auroc_micro"] = baseline_summary["methods"][old_name]["pixel_auroc_micro"]

    _write_csv(report / "per_category_score_maps.csv", all_rows)
    _write_csv(report / "supervised_internal_test.csv", supervised_rows)
    mask_rows = []
    for category, methods in binary_counts.items():
        for method, counts in methods.items():
            tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            mask_rows.append({
                "category": category, "method": method,
                "precision": p, "recall": r,
                "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                "good_fp_pixel_area": counts["good_fp_pixels"] / counts["good_pixels"] if counts["good_pixels"] else 0.0,
                "good_image_fp_rate": counts["good_fp_images"] / counts["good_images"] if counts["good_images"] else 0.0,
            })
    _write_csv(report / "binary_mask_postprocess.csv", mask_rows)
    binary_summary = {}
    for method in MASK_METHODS:
        counts = {key: sum(binary_counts[category][method][key] for category in CATEGORIES) for key in ("tp", "fp", "fn", "tn", "good_fp_pixels", "good_pixels", "good_fp_images", "good_images")}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        binary_summary[method] = {
            "precision": precision,
            "recall": recall,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "good_fp_pixel_area": counts["good_fp_pixels"] / counts["good_pixels"] if counts["good_pixels"] else 0.0,
            "good_image_fp_rate": counts["good_fp_images"] / counts["good_images"] if counts["good_images"] else 0.0,
        }
    summary = {
        "protocol": {
            "dataset": "MVTec AD processed 320x320", "categories": list(CATEGORIES),
            "official_test_images": len(test_records), "train_good": sum(map(len, good_train.values())),
            "validation_good": sum(map(len, good_validation.values())), "seed": args.seed,
            "normal_manifest_sha256": manifest_digest, "defect_split_counts_by_type": defect_counts,
            "exact_knn_bank": "rebuilt in memory from the same 2,540 GOOD train split with the frozen DINOv2 layer-11 patch embeddings; no bank files downloaded",
            "defect_split": "70/15/15 by category and defect type where group size permits; image-disjoint train/validation/test",
            "supervised_evaluation": "custom internal split carved from official test defects; do not call these official MVTec test results",
            "thresholds": "q=.99 per category/method on validation GOOD maps only",
            "prior_test_exposure": "Official MVTec test was previously inspected; all results remain exploratory, not an untouched-holdout estimate.",
        },
        "unsupervised_score_maps": summary_methods,
        "supervised_refiner": {
            "split_test": "official-good test images plus 15% held-out defect images per defect type",
            "methods": {method: {"macro_pixel_ap": float(np.mean([row["pixel_ap"] for row in supervised_rows if row["method"] == method])), "macro_pixel_auroc": float(np.mean([row["pixel_auroc"] for row in supervised_rows if row["method"] == method]))} for method in ("patchcore_raw", "supervised_default", "supervised_spatial", "spatial_decoder")},
            "decoder_training": decoder_provenance,
            "validation_thresholds": {category: {key: value for key, value in thresholds_all[category].items() if key.startswith("supervised_") or key == "spatial_decoder"} for category in CATEGORIES},
        },
        "pixel_metric_scope": "Per-category and macro metrics are recomputed for the added methods. Micro ranking metrics are omitted for new methods to avoid writing all maps to disk; the existing full-test micro baselines are copied from goal_15_category_heatmap_methods_320.",
        "binary_mask_postprocess": {
            "protocol": "thresholded raw PatchCore q=.99 outputs; these operators do not change continuous Pixel AP/AUROC",
            "overall": binary_summary,
        },
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (report / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds_all, handle, indent=2)
        handle.write("\n")

    names = list(summary_methods)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(names, [summary_methods[name]["pixel_ap_macro_15"] for name in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro Pixel AP")
    ax.set_title("MVTec AD 15 categories: heatmap method transfer (320px)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(report / "pixel_ap_macro.png", dpi=160)
    plt.close(fig)
    matrix = np.asarray([
        [next(row["pixel_ap"] for row in all_rows if row["category"] == category and row["method"] == method) for method in names]
        for category in CATEGORIES
    ])
    fig, ax = plt.subplots(figsize=(12, 7))
    heat = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    ax.set_title("Pixel AP by category and heatmap method")
    fig.colorbar(heat, ax=ax, label="Pixel AP")
    fig.tight_layout()
    fig.savefig(report / "pixel_ap_by_category.png", dpi=160)
    plt.close(fig)
    with (report / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Screw-only heatmap method transfer to MVTec AD 15 categories\n\n")
        handle.write("## Protocol and limitations\n\n")
        handle.write("Unsupervised methods use all 1,725 official MVTec test images at 320×320. Thresholds are q=.99 per category/method from known-GOOD validation maps. The supervised logistic refiners and spatial decoder are trained with GOOD train images and a deterministic 70/15/15 split of official test defect images by category/defect type; their held-out custom split is reported separately and is **not** an official MVTec test score. Earlier project work already inspected the official test, so every result is exploratory. No perceptual-duplicate/session audit or paired bootstrap was run.\n\n")
        handle.write("## Continuous heatmap results\n\n| Method | Pixel AP macro | Pixel AUROC macro | Pixel AP micro | Pixel AUROC micro |\n|---|---:|---:|---:|---:|\n")
        for name, values in summary_methods.items():
            micro_ap = f"{values['pixel_ap_micro']:.4f}" if values["pixel_ap_micro"] is not None else "—"
            micro_auc = f"{values['pixel_auroc_micro']:.4f}" if values["pixel_auroc_micro"] is not None else "—"
            handle.write(f"| {name} | {values['pixel_ap_macro_15']:.4f} | {values['pixel_auroc_macro_15']:.4f} | {micro_ap} | {micro_auc} |\n")
        handle.write("\nMicro Pixel AP/AUROC for PatchCore raw and open5 are copied from the previously completed full-test comparison. New variants report exact per-category and macro ranking metrics; their micro ranking scores are intentionally omitted to avoid retaining every test map.\n\n")
        handle.write("\n![Macro Pixel AP](pixel_ap_macro.png)\n\n")
        handle.write("![Pixel AP by category](pixel_ap_by_category.png)\n\n")
        handle.write("## Artifacts\n\n- `per_category_score_maps.csv`: Pixel AP/AUROC and thresholded localization metrics for every continuous method and category.\n- `supervised_internal_test.csv`: custom image-disjoint supervised-refiner test metrics.\n- `binary_mask_postprocess.csv`: binary precision, recall, Dice, IoU and false-positive measures by category.\n- `supervised_internal_split.csv`: exact train/validation/test image assignments for the refiner experiment.\n- `padim_320/`: fitted PaDiM 320px statistics for all categories.\n- `supervised_refiners/`: per-category logistic and spatial-decoder checkpoints.\n")
    print(f"Completed: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
