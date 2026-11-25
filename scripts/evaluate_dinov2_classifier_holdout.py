#!/usr/bin/env python3
"""Evaluate the saved DINOv2 GOOD/DEFECT model on a locked labelled holdout.

The threshold is supplied by the calibration/ledger; this command never tunes
it on the holdout.  The holdout CSV must contain image_path and label columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mvtec_system.core.dinov2 import DINOv2Config  # noqa: E402
from train_dinov2_lastblock_patch_head import (  # noqa: E402
    DINOv2LastBlockClassifier,
    GlobalPatchHead,
    ScrewDataset,
    infer,
    load_backbone,
)

GOOD = 0
DEFECT = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_holdout(path: Path, split: str | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Holdout CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    has_split = "split" in (reader.fieldnames or [])
    if has_split:
        if split is None:
            values = {row.get("split", "") for row in rows}
            if len(values) != 1 or "" in values:
                raise ValueError("CSV has ambiguous split values; pass --split explicitly")
        elif any(not row.get("split") for row in rows):
            raise ValueError("CSV contains blank split values; cannot verify the requested role")
    rows = [row for row in rows if split is None or not has_split or row["split"] == split]
    if not rows:
        raise ValueError("Holdout selection is empty")
    labels = {row["label"] for row in rows}
    if labels != {"GOOD", "DEFECT"}:
        raise ValueError(f"Holdout must contain exactly GOOD and DEFECT; got {sorted(labels)}")
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        row["image_path"] = str(image_path)
    return rows


def summarize(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[GOOD, DEFECT]).ravel()
    good_count = int(np.sum(labels == GOOD))
    defect_count = int(np.sum(labels == DEFECT))
    return {
        "metric_unit": "object_crop",
        "images": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_defect": float(precision_score(labels, predictions, zero_division=0)),
        "defect_recall": float(tp / defect_count),
        "defect_f1": float(f1_score(labels, predictions, zero_division=0)),
        "good_false_positive_rate": float(fp / good_count),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "metric_numerators": {
            "defect_true_positive": int(tp),
            "good_false_alarm_count": int(fp),
        },
        "metric_denominators": {
            "defect_count": defect_count,
            "good_count": good_count,
        },
        "confusion_matrix_good_defect": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def load_locked_selection(
    ledger_manifest: Path,
    manifest: Path,
    model_path: Path,
    split_role: str,
    threshold: float,
) -> dict[str, Any]:
    ledger_manifest = ledger_manifest.resolve()
    selected_path = ledger_manifest.parent / "selected_candidate.json"
    ledger = json.loads(ledger_manifest.read_text(encoding="utf-8"))
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    candidate = selection.get("selected_candidate")
    if selection.get("status") != "passed" or not isinstance(candidate, dict):
        raise ValueError("selection ledger has no passed validation candidate")
    if candidate.get("split_role") != "validation":
        raise ValueError("selected candidate must come from validation")
    selected_threshold = float(candidate["metrics"]["threshold"])
    if abs(float(threshold) - selected_threshold) > 1e-12:
        raise ValueError("--threshold does not match the selected validation threshold")
    model_path = model_path.resolve()
    artifacts = candidate.get("artifact_hashes") or candidate.get("artifact_sha256") or {}
    model_key = str(model_path)
    if model_key not in artifacts:
        raise ValueError("model path is not the selected candidate artifact")
    model_hash = sha256_file(model_path)
    if model_hash != artifacts[model_key]:
        raise ValueError("model hash does not match the selected candidate")
    if ledger.get("independent_holdout_used_for_selection") is not False:
        raise ValueError("ledger must declare that the holdout was not used for selection")
    locked_role = ledger["split_roles"].get(split_role)
    if not isinstance(locked_role, dict):
        raise ValueError(f"ledger has no locked split role: {split_role}")
    manifest = manifest.resolve()
    manifest_hash = sha256_file(manifest)
    if manifest_hash != locked_role["sha256"]:
        raise ValueError(f"manifest hash does not match locked {split_role} role")
    return {
        "selection_ledger": str(ledger_manifest),
        "selection_ledger_sha256": sha256_file(ledger_manifest),
        "selected_candidate_name": candidate["name"],
        "selected_candidate_threshold": selected_threshold,
        "evaluated_threshold": float(threshold),
        "threshold_match": True,
        "model_hash_match": True,
        "locked_manifest_hash_match": True,
        "holdout_used_for_selection": False,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    model_path = args.model.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {args.output}")
    provenance = load_locked_selection(
        args.selection_ledger,
        manifest,
        model_path,
        args.split_role,
        args.threshold,
    )
    rows = read_holdout(manifest, args.split)
    config = DINOv2Config(image_size=(256, 256), feature_layers=())
    device = torch.device(args.device)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if int(checkpoint["feature_dimension"]) <= 0:
        raise ValueError("Invalid classifier feature dimension")
    backbone = load_backbone(config, args.encoder_root.resolve(), device)
    head = GlobalPatchHead(
        int(checkpoint["feature_dimension"]), int(checkpoint["top_k"]), float(checkpoint["dropout"])
    ).to(device)
    model = DINOv2LastBlockClassifier(backbone, head, padding=(10, 10)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(ScrewDataset(rows, config), batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels, scores = infer(model, loader, device)
    metrics = summarize(labels, scores, args.threshold)
    output = {
        "schema_version": "1.0",
        "status": "evaluated",
        "split_role": args.split_role,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "threshold": args.threshold,
        **provenance,
        "device": str(device),
        "metrics": metrics,
        "warning": "Do not use this holdout to tune the threshold; this command only evaluates a preselected operating point.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/products/screw/screw-full-dev-004/classifier_lastblock_patch_head/model.pt"))
    parser.add_argument("--encoder-root", type=Path, default=Path("artifacts/products/screw/screw-full-dev-004/classifier_lastblock_patch_head"))
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--selection-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-role", default="independent_holdout", choices=("diagnostic", "independent_holdout"))
    parser.add_argument("--split")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0 or args.batch_size < 1:
        parser.error("threshold must be in [0, 1] and batch-size must be positive")
    print(json.dumps(evaluate(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
