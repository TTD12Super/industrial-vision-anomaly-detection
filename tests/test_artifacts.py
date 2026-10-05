"""Tests for versioned artifact paths."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mvtec_system import make_artifact_layout


class ArtifactLayoutTests(unittest.TestCase):
    def test_layout_is_versioned_and_created_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = make_artifact_layout(Path(directory) / "artifacts", "v1")
            self.assertFalse(layout.version_root.exists())
            layout.create()

            self.assertTrue(layout.classifier.is_dir())
            self.assertTrue(layout.embeddings.is_dir())
            self.assertTrue(layout.feature_banks.is_dir())
            self.assertTrue(layout.thresholds.is_dir())
            self.assertTrue(layout.evaluation.is_dir())
            self.assertEqual(
                layout.path_for("thresholds", "image.json"),
                (layout.thresholds / "image.json").resolve(),
            )

    def test_unknown_or_unsafe_paths_are_rejected(self) -> None:
        layout = make_artifact_layout("artifacts", "v1")
        with self.assertRaises(ValueError):
            make_artifact_layout("artifacts", "../old")
        with self.assertRaises(ValueError):
            layout.path_for("other", "result.json")
        with self.assertRaises(ValueError):
            layout.path_for("thresholds", "../outside.json")


if __name__ == "__main__":
    unittest.main()
