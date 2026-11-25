from __future__ import annotations

import unittest

from scripts.evaluate_weee_aabb_detectors import _xyxy_corners


class WEEEAABBComparisonTests(unittest.TestCase):
    def test_xyxy_is_converted_to_four_corners(self) -> None:
        self.assertEqual(_xyxy_corners([1, 2, 11, 12]), [[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]])

if __name__ == "__main__":
    unittest.main()
