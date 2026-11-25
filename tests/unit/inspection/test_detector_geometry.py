"""Detector geometry must be safe for downstream object crops."""

from __future__ import annotations

import unittest

import numpy as np

from inspection.backends import clip_detection_corners


class DetectorGeometryTests(unittest.TestCase):
    def test_clips_out_of_frame_obb_before_crop(self) -> None:
        clipped = clip_detection_corners(
            np.asarray([[-20.0, 10.0], [120.0, 10.0], [120.0, 80.0], [-20.0, 80.0]], dtype=np.float32),
            (100, 100),
        )
        assert clipped is not None
        self.assertGreaterEqual(float(clipped.min()), 0.0)
        self.assertLessEqual(float(clipped[:, 0].max()), 99.0)
        self.assertLessEqual(float(clipped[:, 1].max()), 99.0)

    def test_discards_degenerate_clipped_obb(self) -> None:
        clipped = clip_detection_corners(
            np.asarray([[-10.0, -10.0], [-1.0, -10.0], [-1.0, -1.0], [-10.0, -1.0]], dtype=np.float32),
            (100, 100),
        )
        self.assertIsNone(clipped)


if __name__ == "__main__":
    unittest.main()
