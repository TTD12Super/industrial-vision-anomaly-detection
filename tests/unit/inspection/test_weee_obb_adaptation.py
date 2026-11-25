from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.prepare_weee_obb_adaptation import content_hash_split_manifest


class WEEEOBBAdaptationTests(TestCase):
    def test_source_manifest_is_content_disjoint_and_auditable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.png"
            validation = root / "validation.png"
            test = root / "test.png"
            train.write_bytes(b"train")
            validation.write_bytes(b"validation")
            test.write_bytes(b"test")

            manifest = content_hash_split_manifest([train], [validation], [test])

            self.assertTrue(manifest["content_disjoint"])
            self.assertEqual(manifest["counts"], {"train": 1, "validation": 1, "test_locked": 1})
            self.assertEqual(len(manifest["entries"]["test_locked"]), 1)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)

    def test_source_manifest_rejects_exact_content_leakage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.png"
            validation = root / "validation.png"
            test = root / "test.png"
            train.write_bytes(b"same")
            validation.write_bytes(b"different")
            test.write_bytes(b"same")

            with self.assertRaises(ValueError):
                content_hash_split_manifest([train], [validation], [test])


if __name__ == "__main__":
    import unittest

    unittest.main()
