"""Tests for validation classification metrics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from mvtec_system import (
    LinearClassifierConfig,
    evaluate_product_classifier,
    fit_linear_classifier,
    save_classification_report,
    select_linear_classifier,
)


class EvaluationTests(unittest.TestCase):
    def make_classifier(self):
        torch.manual_seed(42)
        mapping = {"bottle": 0, "cable": 1, "capsule": 2}
        centers = torch.zeros(3, 6)
        centers[:, :3] = torch.eye(3) * 5.0
        embeddings = torch.cat(
            [centers[label].repeat(4, 1) + torch.randn(4, 6) * 0.05 for label in range(3)]
        )
        labels = torch.arange(3).repeat_interleave(4)
        artifact = {"embeddings": embeddings, "labels": labels, "split": "validation"}
        classifier = fit_linear_classifier(
            {"embeddings": embeddings, "labels": labels},
            mapping,
            LinearClassifierConfig(max_iter=200),
        )
        return classifier, artifact, mapping

    def test_validation_report_contains_required_metrics(self) -> None:
        classifier, artifact, mapping = self.make_classifier()
        report = evaluate_product_classifier(classifier, artifact, mapping)

        self.assertEqual(report["split"], "validation")
        self.assertEqual(report["sample_count"], 12)
        self.assertEqual(report["metrics"]["accuracy"], 1.0)
        self.assertEqual(report["metrics"]["macro_f1"], 1.0)
        self.assertEqual(report["metrics"]["top_3_accuracy"], 1.0)
        self.assertEqual(report["confusion_matrix"], [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
        self.assertEqual(report["per_class"]["bottle"]["support"], 4)

    def test_report_is_saved_as_json(self) -> None:
        classifier, artifact, mapping = self.make_classifier()
        report = evaluate_product_classifier(classifier, artifact, mapping)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_metrics.json"
            save_classification_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["metrics"], report["metrics"])

    def test_model_selection_uses_validation_metrics(self) -> None:
        classifier, artifact, mapping = self.make_classifier()
        del classifier
        selected, config, report, candidates = select_linear_classifier(
            {"embeddings": artifact["embeddings"], "labels": artifact["labels"]},
            artifact,
            mapping,
            [LinearClassifierConfig(C=0.1, max_iter=200), LinearClassifierConfig(C=1.0, max_iter=200)],
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(config.C, 0.1)
        self.assertEqual(report["model_selection"]["selection_metric"], "macro_f1")
        self.assertEqual(tuple(selected["weights"].shape), (3, 6))


if __name__ == "__main__":
    unittest.main()
