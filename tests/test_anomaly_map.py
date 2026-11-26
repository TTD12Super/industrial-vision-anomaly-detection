"""Anomaly-map shape tests (require torch; skipped otherwise)."""

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipIf(not HAS_TORCH, "torch not installed")
class TestAnomalyMaps(unittest.TestCase):
    def test_batch_output_shape(self):
        from model.inference.anomaly_map import create_anomaly_maps

        scores = torch.rand(3, 361)
        maps = create_anomaly_maps(scores, (19, 19), (266, 266))
        self.assertEqual(maps.shape, (3, 256, 256))

    def test_single_output_shape(self):
        import numpy as np

        from model.inference.anomaly_map import create_anomaly_map

        single = create_anomaly_map(torch.rand(361), (19, 19), (266, 266))
        self.assertIsInstance(single, np.ndarray)
        self.assertEqual(single.shape, (256, 256))

    def test_single_matches_batch_row(self):
        import numpy as np

        from model.inference.anomaly_map import create_anomaly_map, create_anomaly_maps

        scores = torch.rand(2, 361)
        batch = create_anomaly_maps(scores, (19, 19), (266, 266))
        single = create_anomaly_map(scores[0], (19, 19), (266, 266))
        self.assertTrue(np.allclose(batch[0], single))

    def test_wrong_patch_count_raises(self):
        from model.inference.anomaly_map import create_anomaly_maps

        with self.assertRaises(ValueError):
            create_anomaly_maps(torch.rand(2, 100), (19, 19), (266, 266))

    def test_non_finite_scores_raise(self):
        import math

        from model.inference.anomaly_map import create_anomaly_maps

        bad = torch.full((1, 361), math.inf)
        with self.assertRaises(ValueError):
            create_anomaly_maps(bad, (19, 19), (266, 266))


if __name__ == "__main__":
    unittest.main()
