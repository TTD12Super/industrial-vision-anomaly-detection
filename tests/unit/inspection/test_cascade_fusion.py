import tempfile
import unittest
from pathlib import Path

import numpy as np

from inspection.cascade_fusion import CascadeScoreFusion, heatmap_statistics


class CascadeFusionTests(unittest.TestCase):
    def test_heatmap_statistics_are_finite_and_ordered(self) -> None:
        stats = heatmap_statistics(np.asarray([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32))

        self.assertEqual(stats.shape, (6,))
        self.assertTrue(np.isfinite(stats).all())
        self.assertGreaterEqual(stats[0], stats[1])
        self.assertGreaterEqual(stats[1], stats[2])

    def test_fit_predict_and_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(20, 8)).astype(np.float32)
        labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
        embeddings[labels == 1, 0] += 2.0
        p_global = np.asarray([0.1] * 10 + [0.8] * 10, dtype=np.float32)
        anomaly_score = np.asarray([0.1] * 10 + [0.7] * 10, dtype=np.float32)
        heatmap_stats = np.vstack([heatmap_statistics(np.full((4, 4), value, dtype=np.float32)) for value in p_global])
        model = CascadeScoreFusion.fit(embeddings, labels, p_global, anomaly_score, heatmap_stats, 0.5, 0.5)

        predictions = model.predict(embeddings, p_global, anomaly_score, heatmap_stats)

        self.assertEqual(predictions.shape, labels.shape)
        self.assertTrue(np.isfinite(model.predict_score(embeddings, p_global, anomaly_score, heatmap_stats)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cascade.pkl"
            model.save(path)
            loaded = CascadeScoreFusion.load(path)
            np.testing.assert_array_equal(predictions, loaded.predict(embeddings, p_global, anomaly_score, heatmap_stats))
