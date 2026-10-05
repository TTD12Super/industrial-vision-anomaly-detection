"""Tests for exact chunked Euclidean kNN."""

from __future__ import annotations

import unittest

import torch

from mvtec_system import nearest_neighbor_scores


class ExactKNNTests(unittest.TestCase):
    def test_chunked_scores_match_direct_calculation_for_k1_and_k2(self) -> None:
        bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [3.0, 3.0]])
        queries = torch.tensor([[0.0, 0.0], [2.0, 2.0], [1.0, 0.0]])
        direct = torch.linalg.vector_norm(queries[:, None, :] - bank[None, :, :], dim=-1)

        for k in (1, 2):
            expected = direct.topk(k, largest=False).values.mean(dim=1)
            actual = nearest_neighbor_scores(
                queries,
                bank,
                k=k,
                query_chunk_size=1,
                bank_chunk_size=2,
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_bank_chunk_boundary_does_not_change_result(self) -> None:
        torch.manual_seed(42)
        bank = torch.randn(17, 5)
        queries = torch.randn(9, 5)
        expected = torch.cdist(queries, bank).topk(3, largest=False).values.mean(dim=1)
        actual = nearest_neighbor_scores(
            queries, bank, k=3, query_chunk_size=4, bank_chunk_size=5
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_invalid_inputs_are_reported(self) -> None:
        bank = torch.zeros(3, 2)
        queries = torch.zeros(1, 2)
        with self.assertRaises(ValueError):
            nearest_neighbor_scores(queries, bank, k=0)
        with self.assertRaises(ValueError):
            nearest_neighbor_scores(queries, bank, k=4)
        with self.assertRaises(ValueError):
            nearest_neighbor_scores(torch.zeros(1, 3), bank)
        with self.assertRaises(ValueError):
            nearest_neighbor_scores(torch.zeros(0, 2), bank)


if __name__ == "__main__":
    unittest.main()
