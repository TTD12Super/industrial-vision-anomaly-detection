import unittest

import numpy as np

from inspection.geometry import oriented_crop, project_crop_map


class GeometryTests(unittest.TestCase):
    def test_crop_mask_projects_back_to_original_coordinates(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = oriented_crop(image, [[20, 10], [80, 20], [70, 70], [10, 60]])
        mask = np.ones(crop.image.shape[:2], dtype=bool)

        projected = project_crop_map(mask, crop, image.shape[:2])

        self.assertEqual(projected.dtype, bool)
        self.assertTrue(projected[40, 45])
        self.assertFalse(projected[0, 0])

    def test_invalid_corners_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            oriented_crop(np.zeros((10, 10, 3), dtype=np.uint8), [[0, 0], [1, 1]])
