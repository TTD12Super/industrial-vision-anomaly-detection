import unittest

import numpy as np

from inspection.relation_map import contextual_features, relation_scores, scores_to_map


class RelationMapTests(unittest.TestCase):
    def test_attention_convention_is_matrix_times_features(self) -> None:
        attention = np.asarray([[0.75, 0.25], [0.10, 0.90]], dtype=np.float32)
        features = np.asarray([[2.0, 4.0], [10.0, 20.0]], dtype=np.float32)

        result = contextual_features(attention, features)

        np.testing.assert_allclose(result, [[4.0, 8.0], [9.2, 18.4]])

    def test_mismatched_patch_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contextual_features(np.eye(2, dtype=np.float32), np.ones((3, 4), dtype=np.float32))

    def test_v2_scores_and_square_grid_map(self) -> None:
        attention = np.eye(4, dtype=np.float32)
        features = np.arange(8, dtype=np.float32).reshape(4, 2)
        prototype = features.copy()

        scores = relation_scores(attention, features, prototype)
        mapped = scores_to_map(scores, (8, 8))

        np.testing.assert_allclose(scores, np.zeros(4))
        self.assertEqual(mapped.shape, (8, 8))
