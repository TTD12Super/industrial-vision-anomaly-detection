from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_screw_classification_manifest import build_manifest


class ScrewClassificationManifestTests(unittest.TestCase):
    def test_builds_split_audit_and_rejects_cross_split_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "test").mkdir(parents=True)
            train_image = root / "images" / "train" / "a.png"
            test_image = root / "images" / "test" / "b.png"
            train_image.write_bytes(b"train")
            test_image.write_bytes(b"test")
            images = [
                {"image_id": "a", "relative_path": "images/train/a.png", "split": "train", "capture_group_id": "g-a", "sha256": "hash-a"},
                {"image_id": "b", "relative_path": "images/test/b.png", "split": "test", "capture_group_id": "g-b", "sha256": "hash-b"},
            ]
            objects = [
                {"image_id": "a", "quality_label": "GOOD", "defect_types": []},
                {"image_id": "b", "quality_label": "DEFECT", "defect_types": ["scratch_head"]},
            ]
            (root / "manifests" / "images.jsonl").write_text("\n".join(json.dumps(row) for row in images) + "\n")
            (root / "manifests" / "objects.jsonl").write_text("\n".join(json.dumps(row) for row in objects) + "\n")
            audit = build_manifest(root, root / "out.csv", root / "audit.json")
            self.assertTrue(audit["leakage_free_exact_content"])
            self.assertEqual(audit["split_label_counts"]["train__GOOD"], 1)
            self.assertEqual(audit["split_label_counts"]["test__DEFECT"], 1)


if __name__ == "__main__":
    unittest.main()
