"""Exact-kNN correctness tests (require torch; skipped otherwise).

Uses the same tiny vectors as the notebook cell "Check kNN against a tiny
direct calculation", plus chunk-boundary equivalence on seeded random data.
"""

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipIf(not HAS_TORCH, "torch not installed")
class TestNearestNeighborScores(unittest.TestCase):
    def test_notebook_tiny_example(self):
        from model.inference.knn import nearest_neighbor_scores

        example_bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [3.0, 3.0]])
        example_queries = torch.tensor([[0.0, 0.0], [2.0, 2.0], [1.0, 0.0]])
        direct = torch.linalg.vector_norm(
            example_queries[:, None, :] - example_bank[None, :, :], dim=-1
        )
        for k in (1, 2):
            expected = direct.topk(k, largest=False).values.mean(dim=1)
            actual = nearest_neighbor_scores(
                example_queries, example_bank, k,
                query_chunk_size=1, bank_chunk_size=2,
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_chunking_matches_brute_force(self):
        from model.inference.knn import nearest_neighbor_scores

        generator = torch.Generator().manual_seed(0)
        bank = torch.randn(300, 8, generator=generator)
        queries = torch.randn(40, 8, generator=generator)
        expected = torch.cdist(queries, bank, p=2).topk(1, largest=False).values.mean(dim=1)
        for query_chunk, bank_chunk in ((512, 16384), (7, 64), (1, 1)):
            actual = nearest_neighbor_scores(
                queries, bank, 1,
                query_chunk_size=query_chunk, bank_chunk_size=bank_chunk,
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_k_validation(self):
        from model.inference.knn import nearest_neighbor_scores

        bank = torch.zeros(4, 2)
        queries = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            nearest_neighbor_scores(queries, bank, k=0)
        with self.assertRaises(ValueError):
            nearest_neighbor_scores(queries, bank, k=5)

    def test_dimension_mismatch_raises(self):
        from model.inference.knn import nearest_neighbor_scores

        with self.assertRaises(ValueError):
            nearest_neighbor_scores(torch.zeros(2, 3), torch.zeros(4, 2), k=1)


if __name__ == "__main__":
    unittest.main()
