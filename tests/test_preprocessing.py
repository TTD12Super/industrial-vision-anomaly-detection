"""Preprocessing / padding tests.

Pure geometry tests always run. Image-file tests require Pillow/numpy/torch
and are skipped with a clear reason when they are missing.
"""

import tempfile
import unittest
from pathlib import Path

from model.inference.preprocessing import compute_padding, compute_patch_grid

try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401
    import torch  # noqa: F401

    HAS_IMAGE_DEPS = True
except ImportError:
    HAS_IMAGE_DEPS = False


class TestPaddingMath(unittest.TestCase):
    """Notebook rule: 256 input, patch 14 -> pad 10 -> 266, grid 19x19."""

    def test_baseline_padding(self):
        pad_right, pad_bottom, padded_h, padded_w = compute_padding(256, 256, 14, 14)
        self.assertEqual((pad_right, pad_bottom), (10, 10))
        self.assertEqual((padded_h, padded_w), (266, 266))

    def test_baseline_grid(self):
        self.assertEqual(compute_patch_grid(266, 266, 14, 14), (19, 19))
        grid = compute_patch_grid(266, 266, 14, 14)
        self.assertEqual(grid[0] * grid[1], 361)

    def test_divisible_size_needs_no_padding(self):
        self.assertEqual(compute_padding(224, 224, 14, 14), (0, 0, 224, 224))

    def test_non_square_padding(self):
        pad_right, pad_bottom, padded_h, padded_w = compute_padding(256, 100, 14, 14)
        self.assertEqual(padded_h, 266)
        self.assertEqual(padded_w, 112)  # 100 + ((-100) % 14) = 100 + 12
        self.assertEqual((pad_right, pad_bottom), (12, 10))

    def test_invalid_dimensions_raise(self):
        with self.assertRaises(ValueError):
            compute_padding(0, 256, 14, 14)
        with self.assertRaises(ValueError):
            compute_padding(256, 256, 0, 14)
        with self.assertRaises(ValueError):
            compute_patch_grid(265, 266, 14, 14)


@unittest.skipIf(not HAS_IMAGE_DEPS, "Pillow/numpy/torch not installed")
class TestImagePreprocessing(unittest.TestCase):
    def test_prepare_image_from_file(self):
        from PIL import Image

        from model.inference.preprocessing import image_tensor, load_rgb_image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "red.png"
            Image.new("RGB", (512, 512), (255, 0, 0)).save(path)
            rgb = load_rgb_image(path)
            self.assertEqual(rgb.size, (256, 256))
            self.assertEqual(rgb.mode, "RGB")
            tensor = image_tensor(path)
            self.assertEqual(tuple(tensor.shape), (3, 256, 256))
            self.assertTrue(bool((tensor.abs() < 10).all()))

    def test_missing_file_raises(self):
        from model.inference.preprocessing import load_rgb_image

        with self.assertRaises(FileNotFoundError):
            load_rgb_image("does/not/exist.png")

    def test_invalid_ndarray_raises(self):
        import numpy as np

        from model.inference.preprocessing import load_rgb_image

        with self.assertRaises(ValueError):
            load_rgb_image(np.zeros((10, 10), dtype="uint8"))

    def test_pad_for_patch_embedding(self):
        import torch

        from model.inference.preprocessing import pad_for_patch_embedding

        batch = torch.zeros(2, 3, 256, 256)
        padded = pad_for_patch_embedding(batch, 10, 10)
        self.assertEqual(tuple(padded.shape), (2, 3, 266, 266))
        with self.assertRaises(ValueError):
            pad_for_patch_embedding(torch.zeros(3, 256, 256), 10, 10)


if __name__ == "__main__":
    unittest.main()
