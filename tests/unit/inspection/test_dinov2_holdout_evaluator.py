from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

from scripts.evaluate_dinov2_classifier_holdout import (
    load_locked_selection,
    read_holdout,
    sha256_file,
    summarize,
)


class DINOv2HoldoutEvaluatorTests(unittest.TestCase):
    def test_summary_exposes_exact_counts_without_tuning(self) -> None:
        result = summarize(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.8, 0.6, 0.2]),
            threshold=0.5,
        )
        self.assertEqual(result["metric_numerators"], {
            "defect_true_positive": 1,
            "good_false_alarm_count": 1,
        })
        self.assertEqual(result["metric_denominators"], {"defect_count": 2, "good_count": 2})

    def test_holdout_requires_both_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdout.csv"
            path.write_text("image_path,label\nmissing.png,GOOD\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly GOOD and DEFECT"):
                read_holdout(path, None)

    def test_selection_ledger_binds_threshold_model_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            manifest = root / "holdout.csv"
            model.write_bytes(b"model")
            manifest.write_text("image_path,label\nimage.png,GOOD\n", encoding="utf-8")
            ledger_dir = root / "ledger"
            ledger_dir.mkdir()
            (ledger_dir / "manifest.json").write_text(
                json.dumps({
                    "independent_holdout_used_for_selection": False,
                    "split_roles": {"diagnostic": {"sha256": sha256_file(manifest)}},
                }),
                encoding="utf-8",
            )
            (ledger_dir / "selected_candidate.json").write_text(
                json.dumps({
                    "status": "passed",
                    "selected_candidate": {
                        "name": "threshold_001",
                        "split_role": "validation",
                        "metrics": {"threshold": 0.4},
                        "artifact_hashes": {str(model.resolve()): sha256_file(model)},
                    },
                }),
                encoding="utf-8",
            )
            result = load_locked_selection(
                ledger_dir / "manifest.json", manifest, model, "diagnostic", 0.4
            )
            self.assertTrue(result["threshold_match"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_locked_selection(
                    ledger_dir / "manifest.json", manifest, model, "diagnostic", 0.5
                )


if __name__ == "__main__":
    unittest.main()
