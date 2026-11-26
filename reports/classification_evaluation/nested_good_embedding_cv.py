"""Leakage-audited nested CV for DINOv2 good-image product classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
SEED = 20261126


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_pixel_group(image_path: Path) -> str:
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_top1_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=ROOT / "artifacts/production_highres/mvtec_ad/embeddings",
    )
    parser.add_argument(
        "--processed-root", type=Path, default=ROOT / "datasets/mvtec_ad_processed_320"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/experiments/goal_15_category_classification_nested_cv",
    )
    args = parser.parse_args()
    train_path = args.embedding_dir / "train_embeddings.pt"
    metadata_path = args.embedding_dir / "embedding_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = torch.load(train_path, map_location="cpu", weights_only=False)

    if artifact.get("split") != "train":
        raise ValueError("This benchmark must use only the saved train embedding split.")
    x = artifact["embeddings"].float().numpy()
    y = artifact["labels"].numpy().astype(np.int64)
    image_paths = artifact["image_paths"]
    categories = artifact["categories"]
    class_mapping = metadata["class_mapping"]
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
    if len(x) != len(y) or len(y) != len(image_paths) or len(y) != len(categories):
        raise ValueError("Embedding artifact rows and labels do not align.")
    if len(class_names) != 15 or set(np.unique(y)) != set(range(15)):
        raise ValueError("Expected all 15 MVTec categories in the training artifact.")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Duplicate manifest paths found in the training artifact.")
    if image_paths != metadata["splits"]["train"]["source_image_paths"]:
        raise ValueError("Training embedding paths do not match their metadata manifest.")
    if any(
        Path(path).is_absolute() or Path(path).parts[-3:-1] != ("train", "good")
        for path in image_paths
    ):
        raise ValueError("Only relative MVTec train/good image paths are allowed.")
    if any(class_names[label] != category for label, category in zip(y, categories)):
        raise ValueError("Category names do not match the embedding labels.")

    groups = np.asarray(
        [decoded_pixel_group(args.processed_root / relative_path) for relative_path in image_paths]
    )
    unique_group_count = int(len(np.unique(groups)))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline_oof = np.full(len(y), -1, dtype=np.int64)
    selected_oof = np.full(len(y), -1, dtype=np.int64)
    outer_fold_id = np.full(len(y), -1, dtype=np.int64)
    fold_rows: list[dict[str, object]] = []
    selected_params: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups), start=1):
        x_train, y_train, g_train = x[train_idx], y[train_idx], groups[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        outer_fold_id[test_idx] = fold

        baseline = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs", random_state=42)
        baseline.fit(x_train, y_train)
        baseline_oof[test_idx] = baseline.predict(x_test)

        inner = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=SEED + fold
        )
        inner_splits = list(inner.split(x_train, y_train, g_train))
        candidates = [
            {
                "clf": [LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
            {
                "clf": [LinearSVC(dual="auto", random_state=42)],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
            {
                "clf": [SVC(kernel="rbf")],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.001, 0.01],
            },
            {
                "clf": [KNeighborsClassifier(metric="cosine")],
                "clf__n_neighbors": [1, 3, 5, 7],
                "clf__weights": ["uniform", "distance"],
            },
        ]
        search = GridSearchCV(
            Pipeline([("clf", LogisticRegression())]),
            candidates,
            scoring="balanced_accuracy",
            cv=inner_splits,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        selected_oof[test_idx] = search.predict(x_test)
        params = {key: str(value) for key, value in search.best_params_.items()}
        selected_params.append(
            {"outer_fold": fold, "inner_balanced_accuracy": float(search.best_score_), **params}
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_images": len(train_idx),
                "heldout_images": len(test_idx),
                "train_groups": len(np.unique(g_train)),
                "heldout_groups": len(np.unique(groups[test_idx])),
                "baseline_accuracy": accuracy_score(y_test, baseline_oof[test_idx]),
                "baseline_macro_accuracy": balanced_accuracy_score(y_test, baseline_oof[test_idx]),
                "selected_accuracy": accuracy_score(y_test, selected_oof[test_idx]),
                "selected_macro_accuracy": balanced_accuracy_score(y_test, selected_oof[test_idx]),
                "selected_model": str(search.best_estimator_.named_steps["clf"]),
                "selected_params": json.dumps(params, sort_keys=True),
            }
        )

    if np.any(baseline_oof < 0) or np.any(selected_oof < 0):
        raise RuntimeError("An image did not receive exactly one out-of-fold prediction.")

    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "protocol": "nested 5-fold StratifiedGroupKFold; 4-fold inner model/parameter selection",
        "seed": SEED,
        "scope": "MVTec AD train/good images only; official test and validation embeddings were not loaded",
        "target": "15-way product category; GOOD-only images",
        "n_images": int(len(y)),
        "n_categories": len(class_names),
        "unique_decoded_pixel_groups": unique_group_count,
        "exact_duplicate_images_grouped": int(len(y) - unique_group_count),
        "groups": "SHA-256 over decoded RGB pixels; exact duplicate images kept in one fold",
        "residual_group_limit": "No physical-object or acquisition-session IDs exist in the source manifest; only exact decoded-pixel duplicates are grouped.",
        "baseline": metrics(y, baseline_oof),
        "nested_model_selection": metrics(y, selected_oof),
        "folds": fold_rows,
        "selected_models_by_outer_fold": selected_params,
        "per_category": {
            "baseline": {
                name: float(accuracy_score(y[np.asarray(categories) == name], baseline_oof[np.asarray(categories) == name]))
                for name in class_names
            },
            "nested_model_selection": {
                name: float(accuracy_score(y[np.asarray(categories) == name], selected_oof[np.asarray(categories) == name]))
                for name in class_names
            },
        },
        "limitations": [
            "This does not evaluate DEFECT images or UNKNOWN rejection.",
            "MVTec train/good is an in-domain development partition, not a fresh production-camera holdout.",
            "The 15-category research program previously inspected the official MVTec test; this nested CV is a training-only estimate for GOOD images and is not an untouched external acceptance result.",
            "Group IDs are unavailable, so the split can prevent exact decoded-image duplicates crossing folds but cannot rule out shared physical objects or acquisition sessions.",
        ],
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "embedding_manifest_sha256": metadata["manifest_sha256"],
            "train_embedding_file_sha256": sha256_file(train_path),
            "metadata_sha256": sha256_file(metadata_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fold_rows)
    with (args.output / "per_category.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["category", "baseline_accuracy", "nested_model_selection_accuracy"])
        for name in class_names:
            writer.writerow(
                [name, summary["per_category"]["baseline"][name], summary["per_category"]["nested_model_selection"][name]]
            )
    with (args.output / "per_image_oof.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "image_path",
                "decoded_pixel_group_sha256",
                "outer_fold",
                "true_category",
                "baseline_prediction",
                "nested_selection_prediction",
            ]
        )
        for index, image_path in enumerate(image_paths):
            writer.writerow(
                [
                    image_path,
                    groups[index],
                    int(outer_fold_id[index]),
                    categories[index],
                    class_names[baseline_oof[index]],
                    class_names[selected_oof[index]],
                ]
            )

    x_pos = np.arange(len(class_names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x_pos - width / 2,
        [summary["per_category"]["baseline"][name] for name in class_names],
        width,
        label="Logistic regression C=0.1",
    )
    ax.bar(
        x_pos + width / 2,
        [summary["per_category"]["nested_model_selection"][name] for name in class_names],
        width,
        label="Nested model selection",
    )
    ax.set_xticks(x_pos, class_names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Out-of-fold accuracy")
    ax.set_title("DINOv2 global embeddings — MVTec train/good only")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "per_category_oof_accuracy.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"output": str(args.output), "baseline": summary["baseline"], "nested_model_selection": summary["nested_model_selection"], "selected_models_by_outer_fold": selected_params}, indent=2))


if __name__ == "__main__":
    main()
