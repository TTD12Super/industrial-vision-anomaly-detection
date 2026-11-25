from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dinov2_runtime import prepare_runtime


class DinoV2RuntimePreparationTests(unittest.TestCase):
    def test_materializes_trusted_list_and_reuses_cached_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            repository = source / "torch_hub/facebookresearch_dinov2_test"
            repository.mkdir(parents=True)
            for relative in (
                "hubconf.py",
                "dinov2/models/vision_transformer.py",
                "dinov2/layers/attention.py",
            ):
                required = repository / relative
                required.parent.mkdir(parents=True, exist_ok=True)
                required.write_text("# materialized test file\n")
            checkpoint = source / "torch_hub/checkpoints/dinov2_vitb14_pretrain.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            destination = root / "runtime"

            first = prepare_runtime(source, destination)
            second = prepare_runtime(source, destination)

            self.assertEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
            self.assertEqual((destination / "torch_hub/trusted_list").read_text(), "facebookresearch\n")
            self.assertTrue((destination / "torch_hub/facebookresearch_dinov2_test").is_symlink())
            self.assertTrue((destination / "torch_hub/checkpoints").is_symlink())


if __name__ == "__main__":
    unittest.main()
