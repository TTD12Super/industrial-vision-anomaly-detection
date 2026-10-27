#!/usr/bin/env python3
"""Build and evaluate the egg Good/Damaged experiment without retraining YOLO."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen, urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egg_demo.anomaly import AnomalyScore
from egg_demo.detection import YoloEggDetector, crop_detection
from egg_demo.pipeline import EggInference
from egg_demo.visualization import draw_results
from mvtec_system.anomaly.feature_bank import load_or_build_feature_bank
from mvtec_system.anomaly.knn import nearest_neighbor_scores
from mvtec_system.anomaly.maps import make_anomaly_maps
from mvtec_system.anomaly.refinement import refine_binary_mask
from mvtec_system.anomaly.thresholds import (
    calibrate_anomaly_thresholds,
    image_scores_from_patch_scores,
)
from mvtec_system.classification.classifier import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)
from mvtec_system.core.dinov2 import DINOv2Config, SharedDINOv2, image_tensor, select_device


KAGGLE_LIST_URL = "https://www.kaggle.com/api/v1/datasets/list/amiski/damaged-eggshells"
KAGGLE_FILE_URL = "https://www.kaggle.com/api/v1/datasets/download/amiski/damaged-eggshells"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_yolo_label_line(line: str, width: int, height: int) -> tuple[int, tuple[int, int, int, int]]:
    """Convert one normalized YOLO box into clipped image coordinates."""

    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Expected five YOLO fields: {line!r}")
    class_id = int(fields[0])
    x_center, y_center, box_width, box_height = (float(value) for value in fields[1:])
    left = max(0, int(round((x_center - box_width / 2) * width)))
    top = max(0, int(round((y_center - box_height / 2) * height)))
    right = min(width, int(round((x_center + box_width / 2) * width)))
    bottom = min(height, int(round((y_center + box_height / 2) * height)))
    if right <= left or bottom <= top:
        raise ValueError(f"Degenerate YOLO box: {line!r}")
    return class_id, (left, top, right, bottom)


def _request_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "industrial-vision-anomaly-detection"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def list_kaggle_files(manifest_path: Path) -> list[dict]:
    """Cache Kaggle's public file index; no account token is needed."""

    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[dict] = []
    token = ""
    for _ in range(100):
        url = KAGGLE_LIST_URL + (f"?pageToken={quote(token)}" if token else "")
        page = _request_json(url)
        files.extend(page.get("datasetFiles", []))
        token = page.get("nextPageToken", "")
        if not token or not page.get("hasNextPageToken", False):
            break
    if not files:
        raise RuntimeError("Kaggle file index is empty")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(files, indent=2), encoding="utf-8")
    return files


def _download_one(item: tuple[str, Path]) -> str:
    name, output = item
    if output.is_file() and output.stat().st_size > 0:
        return "cached"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded_name = quote(name, safe="")
        urlretrieve(KAGGLE_FILE_URL + "/" + encoded_name + "?filename=" + encoded_name, output)
    except HTTPError as error:
        if output.exists():
            output.unlink()
        if error.code == 404:
            return "missing"
        raise
    return "downloaded"


def ensure_kaggle_fold(raw_root: Path) -> dict[str, list[Path]]:
    """Download only fold_0 train/valid images and labels used by this run."""

    files = list_kaggle_files(raw_root / "files.json")
    selected: list[tuple[str, Path]] = []
    for item in files:
        name = item["name"]
        matched = None
        for split in ("train", "valid"):
            for kind in ("images", "labels"):
                marker = f"/fold_0/{split}/{kind}/"
                if marker in name:
                    matched = (split, kind)
                    break
            if matched:
                break
        if matched is None:
            continue
        split, kind = matched
        output = raw_root / "fold_0" / split / kind / Path(name).name
        selected.append((name, output))
    if not selected:
        raise RuntimeError("Kaggle index did not contain fold_0 train/valid files")
    with ThreadPoolExecutor(max_workers=12) as executor:
        statuses = list(executor.map(_download_one, selected))
    paired = {}
    for split in ("train", "valid"):
        image_paths = sorted(
            path for path in (raw_root / "fold_0" / split / "images").glob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_map = {path.stem: path for path in (raw_root / "fold_0" / split / "labels").glob("*.txt")}
        image_paths = [path for path in image_paths if path.stem in label_map]
        paired[split] = (image_paths, [label_map[path.stem] for path in image_paths])
        if not image_paths:
            raise RuntimeError(f"Kaggle {split} has no complete image/label pairs")
    return {
        "train_images": paired["train"][0],
        "train_labels": paired["train"][1],
        "valid_images": paired["valid"][0],
        "valid_labels": paired["valid"][1],
        "downloaded": statuses.count("downloaded"),
        "cached": statuses.count("cached"),
        "missing": statuses.count("missing"),
    }


def _label_rows(label_path: Path, size: tuple[int, int]) -> list[tuple[int, tuple[int, int, int, int]]]:
    width, height = size
    rows = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(parse_yolo_label_line(line, width, height))
    return rows


def _save_crop(image: Image.Image, box: tuple[int, int, int, int], output: Path, margin: float = 0.05) -> None:
    left, top, right, bottom = box
    box_width, box_height = right - left, bottom - top
    left = max(0, int(round(left - box_width * margin)))
    top = max(0, int(round(top - box_height * margin)))
    right = min(image.width, int(round(right + box_width * margin)))
    bottom = min(image.height, int(round(bottom + box_height * margin)))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, top, right, bottom)).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS).save(output, quality=95)


def _append_record(records: list[dict[str, str]], output_root: Path, path: Path, label: str, split: str, origin: str, source: str) -> None:
    records.append({"image_path": path.relative_to(output_root).as_posix(), "label": label, "split": split, "origin": origin, "source": source})


def prepare_crops(
    kaggle_paths: dict[str, list[Path]],
    output_root: Path,
    yolo_weights: Path,
    yolo_confidence: float,
    yolo_device: str | None,
    seed: int = 42,
) -> dict:
    """Create Good train/validation and Good/Damaged test crops."""

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    records: list[dict[str, str]] = []

    detector = YoloEggDetector(yolo_weights, yolo_device)
    yolo_crops = 0
    yolo_misses = []
    source_yolo = Path("datasets/raw/egg_detection/data/images/train")
    for image_path in sorted(source_yolo.glob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(image_path) as image:
            detections = detector.detect(image_path, yolo_confidence)
            for index, detection in enumerate(detections):
                output = output_root / "good/train" / f"yolo_{image_path.stem}_{index:02d}.png"
                _save_crop(image, tuple(int(round(value)) for value in detection.xyxy), output, margin=0.05)
                _append_record(records, output_root, output, "good", "train", "yolo_crop", str(image_path))
                yolo_crops += 1
            if not detections:
                yolo_misses.append(str(image_path))

    train_images = sorted(kaggle_paths["train_images"])
    random.Random(seed).shuffle(train_images)
    train_normal_images = 0
    validation_normal_images = 0
    kaggle_train_normal_crops = 0
    for index, image_path in enumerate(train_images):
        split = "validation" if index % 5 == 0 else "train"
        with Image.open(image_path) as image:
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            rows = _label_rows(label_path, image.size)
            normal_index = 0
            for class_id, box in rows:
                if class_id != 1:
                    continue
                output = output_root / f"good/{split}" / f"kaggle_train_{image_path.stem}_{normal_index:02d}.png"
                _save_crop(image, box, output)
                _append_record(records, output_root, output, "good", split, "kaggle_normal", str(image_path))
                normal_index += 1
                kaggle_train_normal_crops += 1
            if normal_index:
                if split == "train":
                    train_normal_images += 1
                else:
                    validation_normal_images += 1

    test_good = 0
    test_damaged = 0
    professional_scenes: list[str] = []
    for image_path in sorted(kaggle_paths["valid_images"]):
        with Image.open(image_path) as image:
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            rows = _label_rows(label_path, image.size)
            if len(rows) > 1:
                professional_scenes.append(str(image_path))
            counters = {0: 0, 1: 0}
            for class_id, box in rows:
                label = "damaged" if class_id == 0 else "good" if class_id == 1 else None
                if label is None:
                    continue
                output = output_root / f"test/{label}" / f"kaggle_valid_{image_path.stem}_{counters[class_id]:02d}.png"
                _save_crop(image, box, output)
                _append_record(records, output_root, output, label, "test", "kaggle_valid", str(image_path))
                counters[class_id] += 1
                if label == "good":
                    test_good += 1
                else:
                    test_damaged += 1

    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "label", "split", "origin", "source"))
        writer.writeheader()
        writer.writerows(records)
    (output_root / "professional_scenes.txt").write_text("\n".join(professional_scenes[:20]) + "\n", encoding="utf-8")
    return {
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest.resolve()),
        "counts": {
            "/".join(key): value
            for key, value in Counter((row["split"], row["label"], row["origin"]) for row in records).items()
        },
        "yolo_train_crops": yolo_crops,
        "yolo_misses": yolo_misses,
        "kaggle_train_normal_images": train_normal_images,
        "kaggle_validation_normal_images": validation_normal_images,
        "kaggle_train_normal_crops": kaggle_train_normal_crops,
        "test_good_crops": test_good,
        "test_damaged_crops": test_damaged,
        "professional_scenes": len(professional_scenes),
    }


def _manifest_paths(root: Path, split: str, label: str) -> list[Path]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return [root / row["image_path"] for row in csv.DictReader(handle) if row["split"] == split and row["label"] == label]


def _score_paths(paths: list[Path], encoder: SharedDINOv2, bank: torch.Tensor, batch_size: int = 16) -> tuple[list[float], list[np.ndarray]]:
    scores: list[float] = []
    maps: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([image_tensor(path, encoder.config) for path in batch_paths])
        features = encoder.forward(images)
        patch_scores = nearest_neighbor_scores(
            features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1]), bank
        ).reshape(len(batch_paths), -1)
        scores.extend(float(value) for value in image_scores_from_patch_scores(patch_scores))
        maps.extend(make_anomaly_maps(patch_scores, encoder.patch_grid, encoder.padded_size, encoder.config.image_size))
    return scores, maps


def _save_heatmap(image: Image.Image, anomaly_map: np.ndarray, output: Path, title: str) -> None:
    image = image.convert("RGB").resize((anomaly_map.shape[1], anomaly_map.shape[0]), Image.Resampling.LANCZOS)
    pixels = np.asarray(image).astype(np.float32) / 255.0
    maximum = max(float(anomaly_map.max()), 1e-6)
    heat = plt.get_cmap("inferno")(np.clip(anomaly_map / maximum, 0, 1))[..., :3]
    overlay = np.clip(pixels * 0.55 + heat * 0.45, 0, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(overlay)
    axis.set_title(title)
    axis.axis("off")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _global_embeddings(paths: list[Path], encoder: SharedDINOv2, batch_size: int) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([image_tensor(path, encoder.config) for path in batch_paths])
        embeddings.append(encoder.forward(images).global_embedding.cpu())
    return torch.cat(embeddings) if embeddings else torch.empty((0, 768))


def _classification_metrics(labels: np.ndarray, egg_probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (egg_probabilities >= threshold).astype(np.int64)
    return {
        "images": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_egg": float(precision_score(labels, predictions, zero_division=0)),
        "recall_egg": float(recall_score(labels, predictions, zero_division=0)),
        "f1_egg": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "predicted_class_counts": {
            "unknown": int((predictions == 0).sum()),
            "egg": int((predictions == 1).sum()),
        },
    }


def _best_binary_threshold(labels: np.ndarray, egg_probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 1.0], egg_probabilities)))
    scores = [balanced_accuracy_score(labels, egg_probabilities >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def _train_binary_classifier(
    egg_train: list[Path],
    egg_validation: list[Path],
    egg_test: list[Path],
    encoder: SharedDINOv2,
    source_classifier_root: Path,
    output_root: Path,
    batch_size: int,
    seed: int,
    classification_threshold: float | None = None,
) -> dict:
    """Train only egg/unknown; the old 16-class classifier is not used for routing."""

    base_root = source_classifier_root.parent / "embeddings"
    base_train = torch.load(base_root / "train_embeddings.pt", map_location="cpu", weights_only=True)
    base_validation = torch.load(base_root / "validation_embeddings.pt", map_location="cpu", weights_only=True)
    train_unknown = base_train["embeddings"][torch.tensor([category != "egg" for category in base_train["categories"]])]
    validation_unknown = base_validation["embeddings"][torch.tensor([category != "egg" for category in base_validation["categories"]])]
    egg_train_embeddings = _global_embeddings(egg_train, encoder, batch_size)
    egg_validation_embeddings = _global_embeddings(egg_validation, encoder, batch_size)
    egg_test_embeddings = _global_embeddings(egg_test, encoder, batch_size)

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(train_unknown), size=min(len(train_unknown), len(egg_train_embeddings)), replace=False)
    train_embeddings = torch.cat((train_unknown[selected], egg_train_embeddings))
    train_labels = torch.cat((torch.zeros(len(selected), dtype=torch.long), torch.ones(len(egg_train_embeddings), dtype=torch.long)))
    class_mapping = {"unknown": 0, "egg": 1}
    train_artifact = {"embeddings": train_embeddings, "labels": train_labels}
    classifier_config = LinearClassifierConfig()
    classifier = fit_linear_classifier(train_artifact, class_mapping, classifier_config)
    output_root.mkdir(parents=True, exist_ok=True)
    save_linear_classifier(
        output_root,
        classifier,
        class_mapping,
        classifier_config,
        {"artifact_version": "v1", "source": "egg_crops_plus_mvtec_unknown"},
    )

    validation_order = rng.permutation(len(validation_unknown))
    midpoint = max(1, len(validation_order) // 2)
    calibration_unknown = validation_unknown[validation_order[:midpoint]]
    test_unknown = validation_unknown[validation_order[midpoint:]]
    calibration_embeddings = torch.cat((calibration_unknown, egg_validation_embeddings))
    calibration_labels = np.concatenate((np.zeros(len(calibration_unknown), dtype=np.int64), np.ones(len(egg_validation_embeddings), dtype=np.int64)))
    _, calibration_probabilities = predict_linear_classifier(classifier, calibration_embeddings)
    egg_index = class_mapping["egg"]
    threshold = (
        float(classification_threshold)
        if classification_threshold is not None
        else _best_binary_threshold(calibration_labels, calibration_probabilities[:, egg_index].numpy())
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("classification_threshold must be between 0 and 1")
    test_embeddings = torch.cat((test_unknown, egg_test_embeddings))
    test_labels = np.concatenate((np.zeros(len(test_unknown), dtype=np.int64), np.ones(len(egg_test_embeddings), dtype=np.int64)))
    _, test_probabilities = predict_linear_classifier(classifier, test_embeddings)
    report = _classification_metrics(test_labels, test_probabilities[:, egg_index].numpy(), threshold)
    report["output_classes"] = ["egg", "unknown"]
    report["train_images"] = {"egg": len(egg_train), "unknown": len(selected)}
    report["calibration_images"] = {"egg": len(egg_validation), "unknown": len(calibration_unknown)}
    report["test_images_by_source"] = {"egg_kaggle": len(egg_test), "mvtec_unknown": len(test_unknown)}
    report["egg_only_recall"] = float((test_probabilities[len(test_unknown):, egg_index] >= threshold).float().mean())
    report["classifier_path"] = str((output_root / "classifier.pt").resolve())
    report["threshold_strategy"] = "explicit_recall_first" if classification_threshold is not None else "validation_balanced_accuracy"
    (output_root / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _binary_report(good_scores: list[float], damaged_scores: list[float], threshold: float) -> dict:
    scores = good_scores + damaged_scores
    labels = [0] * len(good_scores) + [1] * len(damaged_scores)
    predictions = [int(score >= threshold) for score in scores]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    return {
        "good_images": len(good_scores),
        "damaged_images": len(damaged_scores),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_damaged": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix": matrix,
    }


def _professional_test(
    scene_paths: list[Path],
    detector: YoloEggDetector,
    encoder: SharedDINOv2,
    bank: torch.Tensor,
    image_threshold: float,
    pixel_threshold: float,
    output_root: Path,
    confidence: float,
) -> dict:
    rows = []
    for scene_path in scene_paths:
        with Image.open(scene_path) as source:
            image = source.convert("RGB")
            detections = detector.detect(scene_path, confidence)
            inferences = []
            for index, detection in enumerate(detections):
                crop = crop_detection(image, detection, margin=0.05)
                pixels = np.asarray(crop.resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(pixels).permute(2, 0, 1)
                mean = torch.tensor(encoder.config.mean, dtype=tensor.dtype).view(3, 1, 1)
                std = torch.tensor(encoder.config.std, dtype=tensor.dtype).view(3, 1, 1)
                features = encoder.forward(((tensor - mean) / std).unsqueeze(0))
                patch_scores = nearest_neighbor_scores(features.patch_embeddings.reshape(-1, features.patch_embeddings.shape[-1]), bank).reshape(1, -1)
                score = float(image_scores_from_patch_scores(patch_scores)[0])
                anomaly_map = make_anomaly_maps(patch_scores, encoder.patch_grid, encoder.padded_size, encoder.config.image_size)[0]
                label = "DEFECT" if score >= image_threshold else "GOOD"
                binary_mask = refine_binary_mask(anomaly_map >= pixel_threshold, 9, 3)
                anomaly = AnomalyScore(score, label, image_threshold, anomaly_map, binary_mask)
                inferences.append(EggInference(detection, anomaly))
                heatmap_path = output_root / "heatmaps" / f"{scene_path.stem}_egg{index:02d}.png"
                _save_heatmap(crop, anomaly_map, heatmap_path, f"{label} {score:.3f}")
                rows.append({"scene": str(scene_path), "detection_index": index, "confidence": detection.confidence, "class": detection.class_name, "score": score, "label": label, "heatmap": str(heatmap_path)})
            draw_results(scene_path, inferences, output_root / "annotated" / scene_path.name)
    return {"scene_count": len(scene_paths), "detection_count": len(rows), "predictions": rows}


def run(args: argparse.Namespace) -> dict:
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    artifact_root = Path(args.artifact_root)
    reports_root = Path(args.reports_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = ensure_kaggle_fold(raw_root)
    preparation = prepare_crops(paths, processed_root, Path(args.yolo_weights), args.yolo_confidence, args.yolo_device, args.seed)

    device = args.device or str(select_device())
    encoder = SharedDINOv2(DINOv2Config(image_size=(256, 256), feature_layers=()), artifact_root=args.encoder_root, device=device)
    train_paths = _manifest_paths(processed_root, "train", "good")
    validation_paths = _manifest_paths(processed_root, "validation", "good")
    bank_result = load_or_build_feature_bank(artifact_root / "feature_banks", "egg", train_paths, processed_root, encoder, bank_device=device, batch_size=args.batch_size, rebuild=args.rebuild_feature_bank)
    validation_scores, validation_maps = _score_paths(validation_paths, encoder, bank_result.features, args.batch_size)
    thresholds = calibrate_anomaly_thresholds(validation_scores, np.concatenate([value.reshape(-1) for value in validation_maps]), args.target_acceptance_rate)
    (artifact_root / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    good_test = _manifest_paths(processed_root, "test", "good")
    damaged_test = _manifest_paths(processed_root, "test", "damaged")
    good_scores, good_maps = _score_paths(good_test, encoder, bank_result.features, args.batch_size)
    damaged_scores, damaged_maps = _score_paths(damaged_test, encoder, bank_result.features, args.batch_size)
    binary = _binary_report(good_scores, damaged_scores, thresholds["image"]["threshold"])
    test_records = []
    for label, paths_to_save, scores, maps in (("good", good_test, good_scores, good_maps), ("damaged", damaged_test, damaged_scores, damaged_maps)):
        for path, score, anomaly_map in zip(paths_to_save, scores, maps):
            test_records.append({"path": str(path), "label": label, "score": score, "decision": "DEFECT" if score >= thresholds["image"]["threshold"] else "GOOD"})
        ranked = sorted(zip(paths_to_save, scores, maps), key=lambda item: item[1], reverse=True)[: args.heatmap_examples]
        for index, (path, score, anomaly_map) in enumerate(ranked):
            with Image.open(path) as image:
                _save_heatmap(image, anomaly_map, reports_root / "heatmaps" / f"{label}_{index:02d}_{path.stem}.png", f"{label} {score:.3f}")
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "test_predictions.json").write_text(json.dumps(test_records, indent=2), encoding="utf-8")

    classification = {
        "status": "disabled",
        "mode": "yolo_crop_is_egg",
        "output_classes": ["egg"],
        "reason": "YOLO detections are treated as egg crops; no egg/unknown gate is applied.",
    }
    scene_paths = [Path(line) for line in (processed_root / "professional_scenes.txt").read_text().splitlines() if line]
    professional = _professional_test(scene_paths[: args.professional_images], YoloEggDetector(args.yolo_weights, args.yolo_device), encoder, bank_result.features, thresholds["image"]["threshold"], thresholds["pixel"]["threshold"], reports_root / "professional", args.yolo_confidence)
    summary = {"dataset": "amiski/damaged-eggshells", "device": device, "preparation": preparation, "feature_bank": {"path": str(bank_result.path), "train_good_crops": len(train_paths), "reused": bank_result.reused}, "thresholds": thresholds, "anomaly_test": binary, "classification_test": classification, "professional_test": professional}
    (reports_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/external/egg_damaged_kaggle"))
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/processed/egg_kaggle_experiment"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/production/egg_kaggle_experiment"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/egg_kaggle_experiment"))
    parser.add_argument("--yolo-weights", type=Path, default=Path("artifacts/egg_demo/yolo/train-2/weights/best.pt"))
    parser.add_argument("--classifier-root", type=Path, default=Path("artifacts/production/mvtec_ad_egg/classifier"))
    parser.add_argument("--encoder-root", type=Path, default=Path("artifacts/production/mvtec_ad"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.90)
    parser.add_argument("--rebuild-feature-bank", action="store_true")
    parser.add_argument("--classification-threshold", type=float, default=0.10)
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap-examples", type=int, default=10)
    parser.add_argument("--professional-images", type=int, default=10)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
