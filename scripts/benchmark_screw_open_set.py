#!/usr/bin/env python3
"""Reconcile and benchmark the screw DINOv2 open-set recognizer.

This command deliberately keeps product recognition experimental.  It builds
one immutable calibration/report pair from the same gallery, known validation
objects, and explicit MVTec unknown objects.  The MVTec source is useful for
QA, but it is not a target-camera production holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inspection.config import ProductConfig, load_product_config
from inspection.dataset import load_dataset_records
from inspection.geometry import oriented_crop
from inspection.product_recognition import (
    CALIBRATION_ARTIFACT_TYPE,
    CALIBRATION_SCHEMA_VERSION,
    _crop_tensor,
    _encoder_config,
    _product_score_matrix,
    calibrate_gallery_threshold,
    load_gallery_artifact,
)
from mvtec_system.core.dinov2 import SharedDINOv2, select_device
from screw_yolo_obb import mvtec_bbox_to_corners


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _known_records(config: ProductConfig, split: str) -> list[dict[str, Any]]:
    images, objects = load_dataset_records(config)
    image_by_id = {row["image_id"]: row for row in images if row["split"] == split}
    records = []
    for row in objects:
        image = image_by_id.get(row["image_id"])
        if image is None:
            continue
        records.append(
            {
                "sample_id": row["object_id"],
                "product_key": config.product_key,
                "image_path": config.dataset_path(Path(image["relative_path"])),
                "corners": row["obb_xyxyxyxy"],
                "source_image_id": row["image_id"],
            }
        )
    if not records:
        raise ValueError(f"No known object records exist in {split}")
    return records


def _unknown_records(
    annotation_path: Path,
    image_root: Path,
    selected_types: set[int],
    image_id_min: int,
    image_id_max: int | None,
) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in payload.get("categories", [])}
    images = {int(row["id"]): row for row in payload.get("images", [])}
    records: list[dict[str, Any]] = []
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id < image_id_min or (image_id_max is not None and image_id > image_id_max):
            continue
        category_name = categories.get(int(annotation["category_id"]), "")
        if not category_name.startswith("type_"):
            continue
        try:
            type_number = int(category_name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if type_number not in selected_types:
            continue
        image = images[image_id]
        image_path = image_root / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append(
            {
                "sample_id": f"{category_name}:{image_id}:{annotation['id']}",
                "product_key": category_name,
                "image_path": image_path,
                "corners": mvtec_bbox_to_corners(annotation["bbox"]).tolist(),
                "source_image_id": image_id,
            }
        )
    if not records:
        raise ValueError("No unknown records matched the requested MVTec image/type range")
    return records


def _encode_records(records: list[dict[str, Any]], encoder: Any, batch_size: int) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        tensors = []
        for record in records[start : start + batch_size]:
            with Image.open(record["image_path"]) as source:
                crop = oriented_crop(np.asarray(source.convert("RGB")), record["corners"]).image
            tensors.append(_crop_tensor(crop, encoder.config))
        with torch.no_grad():
            output = encoder.forward(torch.stack(tensors)).global_embedding.float().cpu()
        if output.ndim != 2 or not torch.isfinite(output).all().item():
            raise ValueError("DINOv2 produced invalid open-set embeddings")
        encoded.append(F.normalize(output, dim=1))
    return torch.cat(encoded, dim=0)


def _scores(embeddings: torch.Tensor, gallery: dict[str, Any]) -> np.ndarray:
    _, product_scores = _product_score_matrix(
        embeddings, gallery["embeddings"], gallery["product_keys"]
    )
    return product_scores.max(dim=1).values.numpy()


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _set_metrics(scores: np.ndarray, threshold: float, known: bool) -> dict[str, Any]:
    accepted = scores >= threshold
    return {
        "count": int(scores.size),
        "accepted_as_known": int(accepted.sum()),
        "rejected_as_unknown": int((~accepted).sum()),
        "accepted_rate": _rate(accepted.tolist()),
        "rejection_rate": _rate((~accepted).tolist()),
        "false_known_rate": _rate(accepted.tolist()) if not known else None,
        "score_min": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
    }


def _provenance(config: ProductConfig, split: str) -> dict[str, Any]:
    image_manifest = config.dataset_path(config.image_manifest)
    object_manifest = config.dataset_path(config.object_manifest)
    return {
        "product_key": config.product_key,
        "split": split,
        "image_manifest": str(image_manifest),
        "image_manifest_sha256": _sha256(image_manifest),
        "object_manifest": str(object_manifest),
        "object_manifest_sha256": _sha256(object_manifest),
        "split_policy": config.split_policy,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config = load_product_config(args.config)
    gallery_source = config.project_path(args.gallery)
    gallery = load_gallery_artifact(gallery_source)
    output_dir = config.project_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = output_dir / "gallery.pt"
    shutil.copy2(gallery_source, gallery_path)
    gallery = load_gallery_artifact(gallery_path)

    annotation_path = config.project_path(args.unknown_annotations)
    image_root = config.project_path(args.unknown_image_root) if args.unknown_image_root else annotation_path.parent / "images"
    selected_types = {int(value) for value in args.unknown_types.split(",") if value.strip()}
    known_validation = _known_records(config, "validation")
    known_holdout = _known_records(config, "test")
    unknown_validation = _unknown_records(
        annotation_path, image_root, selected_types, args.unknown_calibration_min, args.unknown_calibration_max
    )
    unknown_holdout = _unknown_records(
        annotation_path, image_root, selected_types, args.unknown_holdout_min, args.unknown_holdout_max
    )

    encoder_config = _encoder_config(gallery["encoder"])
    encoder_root = args.encoder_artifact_root or config.model("classifier").get("encoder_artifact_root", "artifacts/dinov2_knn")
    encoder = SharedDINOv2(
        encoder_config,
        artifact_root=config.project_path(encoder_root),
        device=args.device or str(select_device()),
    )
    known_validation_embeddings = _encode_records(known_validation, encoder, args.batch_size)
    unknown_validation_embeddings = _encode_records(unknown_validation, encoder, args.batch_size)
    known_holdout_embeddings = _encode_records(known_holdout, encoder, args.batch_size)
    unknown_holdout_embeddings = _encode_records(unknown_holdout, encoder, args.batch_size)

    calibration = calibrate_gallery_threshold(
        gallery["embeddings"],
        known_validation_embeddings,
        unknown_validation_embeddings,
        args.target_known_recall,
        [record["product_key"] for record in known_validation],
        gallery["product_keys"],
    )
    calibration.update(
        {
            "artifact_type": CALIBRATION_ARTIFACT_TYPE,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "gallery_sha256": _sha256(gallery_path),
            "known_product_keys": sorted({record["product_key"] for record in known_validation}),
            "unknown_product_keys": sorted({record["product_key"] for record in unknown_validation}),
            "calibration_split": "known_validation + explicit_unknown_calibration",
            "unknown_examples_used": True,
            "known_source_manifests": [_provenance(config, "validation")],
            "unknown_source": str(annotation_path),
            "unknown_source_sha256": _sha256(annotation_path),
            "unknown_calibration_image_id_range": [args.unknown_calibration_min, args.unknown_calibration_max],
            "unknown_holdout_image_id_range": [args.unknown_holdout_min, args.unknown_holdout_max],
            "selected_unknown_types": sorted(selected_types),
        }
    )
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    threshold = float(calibration["known_threshold"])
    report = {
        "schema_version": "1.0",
        "product_key": config.product_key,
        "recognizer": "DINOv2 global embedding nearest GOOD gallery",
        "gallery_path": str(gallery_path),
        "gallery_sha256": _sha256(gallery_path),
        "gallery_count": len(gallery["product_keys"]),
        "calibration_path": str(calibration_path),
        "calibration": calibration,
        "known_validation": _set_metrics(_scores(known_validation_embeddings, gallery), threshold, True),
        "unknown_calibration": _set_metrics(_scores(unknown_validation_embeddings, gallery), threshold, False),
        "known_holdout": _set_metrics(_scores(known_holdout_embeddings, gallery), threshold, True),
        "unknown_holdout": _set_metrics(_scores(unknown_holdout_embeddings, gallery), threshold, False),
        "protocol": {
            "gallery_source": "screw train GOOD object crops only",
            "known_calibration": "screw validation GOOD + DEFECT object crops",
            "unknown_calibration": f"MVTec Screws unknown types {sorted(selected_types)}, image ids {args.unknown_calibration_min}-{args.unknown_calibration_max}",
            "known_holdout": "screw test GOOD + DEFECT object crops",
            "unknown_holdout": f"MVTec Screws unknown types {sorted(selected_types)}, image ids {args.unknown_holdout_min}-{args.unknown_holdout_max}",
            "threshold_selection": f"highest observed cosine threshold meeting known validation recall target {args.target_known_recall:.2f}",
            "test_selection": "known and unknown holdouts were not used for threshold selection",
        },
        "warning": "This is a reconciled development benchmark, not a target-camera production release. Acquire external known-good and unknown captures before enabling the recognizer in the main product config.",
    }
    report_path = output_dir / "open_set_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir.resolve()),
        "gallery": str(gallery_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "report": str(report_path.resolve()),
        "threshold": threshold,
        "known_validation": report["known_validation"],
        "unknown_calibration": report["unknown_calibration"],
        "known_holdout": report["known_holdout"],
        "unknown_holdout": report["unknown_holdout"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/products/screw.yaml")
    parser.add_argument("--gallery", default="artifacts/products/screw/screw-recognizer-rnd2/gallery.pt")
    parser.add_argument("--unknown-annotations", default="datasets/external/mvtec_screws/extracted/mvtec_screws_test.json")
    parser.add_argument("--unknown-image-root")
    parser.add_argument("--unknown-types", default="7,9,10,11,12")
    parser.add_argument("--unknown-calibration-min", type=int, default=1)
    parser.add_argument("--unknown-calibration-max", type=int, default=30)
    parser.add_argument("--unknown-holdout-min", type=int, default=31)
    parser.add_argument("--unknown-holdout-max", type=int, default=60)
    parser.add_argument("--target-known-recall", type=float, default=0.90)
    parser.add_argument("--output-dir", default="artifacts/products/screw/screw-recognizer-rnd3")
    parser.add_argument("--encoder-artifact-root")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    return parser


if __name__ == "__main__":
    print(json.dumps(benchmark(build_parser().parse_args()), indent=2))
