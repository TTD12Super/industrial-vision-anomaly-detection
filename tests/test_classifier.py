"""Tests for the frozen-embedding product classifier."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from mvtec_system import (
    LinearClassifierConfig,
    fit_linear_classifier,
    predict_linear_classifier,
    save_linear_classifier,
)


class LinearClassifierTests(unittest.TestCase):
    def make_training_artifact(self) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        torch.manual_seed(42)
        mapping = {"bottle": 0, "cable": 1, "capsule": 2}
        centers = torch.zeros(3, 6)
        centers[:, :3] = torch.eye(3) * 5.0
        embeddings = torch.cat(
            [centers[label].repeat(4, 1) + torch.randn(4, 6) * 0.05 for label in range(3)]
        )
        labels = torch.arange(3).repeat_interleave(4)
        return {"embeddings": embeddings, "labels": labels}, mapping

    def test_fit_and_predict_on_frozen_embeddings(self) -> None:
        train_artifact, mapping = self.make_training_artifact()
        classifier = fit_linear_classifier(
            train_artifact,
            mapping,
            LinearClassifierConfig(max_iter=200),
        )
        logits, probabilities = predict_linear_classifier(classifier, train_artifact["embeddings"])

        self.assertEqual(tuple(classifier["weights"].shape), (3, 6))
        self.assertEqual(tuple(classifier["bias"].shape), (3,))
        self.assertEqual(tuple(logits.shape), (12, 3))
        self.assertEqual(tuple(probabilities.shape), (12, 3))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(12), atol=1e-6))
        self.assertEqual(probabilities.argmax(dim=1).tolist(), train_artifact["labels"].tolist())

    def test_training_rejects_missing_category(self) -> None:
        train_artifact, mapping = self.make_training_artifact()
        train_artifact["labels"] = torch.zeros_like(train_artifact["labels"])
        with self.assertRaises(ValueError):
            fit_linear_classifier(train_artifact, mapping)

    def test_save_writes_weights_mapping_and_config(self) -> None:
        train_artifact, mapping = self.make_training_artifact()
        config = LinearClassifierConfig(max_iter=200)
        classifier = fit_linear_classifier(train_artifact, mapping, config)
        metadata = {
            "artifact_version": "v1",
            "manifest_sha256": "manifest-hash",
            "encoder": {"model_name": "dinov2_vitb14"},
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_linear_classifier(output, classifier, mapping, config, metadata)
            saved = torch.load(output / "classifier.pt", map_location="cpu", weights_only=True)
            saved_mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
            record = json.loads(
                (output / "classifier_training_config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(tuple(saved["weights"].shape), (3, 6))
        self.assertEqual(saved_mapping["category_to_index"], mapping)
        self.assertTrue(record["encoder_frozen"])
        self.assertEqual(record["classifier_type"], "multinomial_logistic_regression")


if __name__ == "__main__":
    unittest.main()
