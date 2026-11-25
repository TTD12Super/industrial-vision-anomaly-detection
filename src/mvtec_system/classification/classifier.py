"""Lightweight linear product classifier on frozen DINOv2 embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch


@dataclass(frozen=True)
class LinearClassifierConfig:
    """Reproducible training settings for the routing classifier."""

    C: float = 1.0
    max_iter: int = 2000
    solver: str = "lbfgs"
    random_state: int = 42


def _validate_training_data(
    train_artifact: dict[str, Any],
    class_mapping: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = train_artifact.get("embeddings")
    labels = train_artifact.get("labels")
    if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("Training artifact must contain tensor embeddings and labels.")
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embedding and label shapes do not match.")
    if embeddings.shape[0] == 0 or not torch.isfinite(embeddings).all().item():
        raise ValueError("Training embeddings must be non-empty and finite.")
    expected_labels = set(class_mapping.values())
    actual_labels = set(labels.tolist())
    if actual_labels != expected_labels:
        raise ValueError(f"Training labels do not match class mapping: {actual_labels}")
    if labels.dtype != torch.int64:
        labels = labels.long()
    return embeddings.float().cpu(), labels.cpu()


def fit_linear_classifier(
    train_artifact: dict[str, Any],
    class_mapping: dict[str, int],
    config: LinearClassifierConfig | None = None,
) -> dict[str, Any]:
    """Fit multinomial logistic regression without changing the DINOv2 encoder."""

    config = config or LinearClassifierConfig()
    if config.C <= 0 or config.max_iter < 1:
        raise ValueError("C must be positive and max_iter must be at least one.")
    embeddings, labels = _validate_training_data(train_artifact, class_mapping)
    model = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        random_state=config.random_state,
        solver=config.solver,
    )
    model.fit(embeddings.numpy(), labels.numpy())
    classes = torch.from_numpy(model.classes_.astype(np.int64))
    if classes.tolist() != sorted(class_mapping.values()):
        raise ValueError(f"Classifier classes do not match mapping: {classes.tolist()}")
    coefficients = model.coef_.astype(np.float32)
    intercept = model.intercept_.astype(np.float32)
    if len(classes) == 2 and coefficients.shape[0] == 1:
        # sklearn stores binary logistic regression as one decision function;
        # expand it to two logits so predict_linear_classifier has one column per class.
        coefficients = np.concatenate((np.zeros_like(coefficients), coefficients), axis=0)
        intercept = np.concatenate((np.zeros(1, dtype=np.float32), intercept), axis=0)
    return {
        "weights": torch.from_numpy(coefficients).float(),
        "bias": torch.from_numpy(intercept).float(),
        "classes": classes,
        "feature_dimension": embeddings.shape[1],
        "class_count": len(class_mapping),
        "n_iter": model.n_iter_.tolist(),
    }


def predict_linear_classifier(
    classifier: dict[str, Any],
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and softmax probabilities for cached or new embeddings."""

    weights = classifier.get("weights")
    bias = classifier.get("bias")
    if not isinstance(weights, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("Classifier artifact is missing tensor weights or bias.")
    if embeddings.ndim != 2 or embeddings.shape[1] != weights.shape[1]:
        raise ValueError("Embedding dimension does not match classifier weights.")
    logits = embeddings.float().cpu() @ weights.float().t() + bias.float()
    return logits, torch.softmax(logits, dim=1)


def save_linear_classifier(
    output_dir: str | Path,
    classifier: dict[str, Any],
    class_mapping: dict[str, int],
    config: LinearClassifierConfig,
    embedding_metadata: dict[str, Any],
) -> None:
    """Save weights and a human-readable training record beside the cache."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier, output_dir / "classifier.pt")
    mapping_path = output_dir / "class_mapping.json"
    if mapping_path.is_file():
        stored_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if stored_mapping.get("category_to_index") != class_mapping:
            raise ValueError("Existing class mapping is incompatible with classifier output.")
    else:
        mapping_path.write_text(
            json.dumps(
                {
                    "categories": list(class_mapping),
                    "category_to_index": class_mapping,
                    "index_to_category": {
                        str(index): category for category, index in class_mapping.items()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    training_record = {
        "artifact_version": "v1",
        "classifier_type": "multinomial_logistic_regression",
        "encoder_frozen": True,
        "class_mapping": class_mapping,
        "feature_dimension": classifier["feature_dimension"],
        "class_count": classifier["class_count"],
        "n_iter": classifier["n_iter"],
        "config": asdict(config),
        "embedding_artifact_version": embedding_metadata.get("artifact_version"),
        "embedding_manifest_sha256": embedding_metadata.get("manifest_sha256"),
        "embedding_encoder": embedding_metadata.get("encoder", {}),
    }
    (output_dir / "classifier_training_config.json").write_text(
        json.dumps(training_record, indent=2) + "\n",
        encoding="utf-8",
    )
