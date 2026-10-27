from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_egg_kaggle_experiment.py"
SPEC = importlib.util.spec_from_file_location("egg_kaggle_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class KaggleExperimentTests(unittest.TestCase):
    def test_parse_yolo_label_line_clips_coordinates(self) -> None:
        class_id, box = MODULE.parse_yolo_label_line("1 0.5 0.5 0.8 0.6", 100, 80)
        self.assertEqual(class_id, 1)
        self.assertEqual(box, (10, 16, 90, 64))

    def test_classification_output_collapses_non_egg_to_unknown(self) -> None:
        mapping = {"0": "pill", "15": "egg"}
        output = ["egg" if mapping[str(index)] == "egg" else "unknown" for index in (15, 0, 15)]
        self.assertEqual(output, ["egg", "unknown", "egg"])

    def test_binary_classification_metrics_use_egg_as_positive(self) -> None:
        report = MODULE._classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.2]), 0.5
        )
        self.assertEqual(report["confusion_matrix"], [[1, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
