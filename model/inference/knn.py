"""Exact kNN scoring, mirroring ``notebooks/02_first_model.ipynb``.

Rules preserved from the notebook:
* Euclidean distance via ``torch.cdist(p=2)``
* default k=1 (mean distance to the k nearest normal patches)
* chunked over queries (``QUERY_CHUNK_SIZE``) and bank rows
  (``BANK_CHUNK_SIZE``) so the full distance matrix is never materialized
* no FAISS, no approximate nearest neighbors

``torch`` is imported lazily inside functions so dependency-free unit tests
can import this module without torch installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import BANK_CHUNK_SIZE, K_NEIGHBORS, QUERY_CHUNK_SIZE

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def distances_for_chunk(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    bank_chunk_size: int = BANK_CHUNK_SIZE,
) -> torch.Tensor:
    """Mean distance of each query to its k nearest bank rows (one query chunk)."""
    import torch as _torch

    with _torch.inference_mode():
        if k < 1 or k > len(bank):
            raise ValueError("k must be between 1 and the bank size.")
        if bank_chunk_size < 1:
            raise ValueError(f"bank_chunk_size must be positive, got {bank_chunk_size}.")
        best_distances = _torch.full((len(queries), k), float("inf"), device=bank.device)
        for start in range(0, len(bank), bank_chunk_size):
            bank_chunk = bank[start : start + bank_chunk_size]
            distances = _torch.cdist(queries, bank_chunk, p=2)
            candidate_distances = _torch.cat((best_distances, distances), dim=1)
            best_distances = candidate_distances.topk(k, dim=1, largest=False).values
        return best_distances.mean(dim=1)


def nearest_neighbor_scores(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int = K_NEIGHBORS,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    bank_chunk_size: int = BANK_CHUNK_SIZE,
) -> torch.Tensor:
    """Patch anomaly scores: distance of each query patch to the normal bank.

    Args:
        queries: ``(num_queries, embedding_dim)`` float tensor.
        bank: ``(num_bank_patches, embedding_dim)`` float tensor.
        k: number of nearest neighbors (mean of the k smallest distances).
        query_chunk_size: rows of *queries* processed at once.
        bank_chunk_size: rows of *bank* compared at once.

    Returns:
        CPU float tensor of shape ``(num_queries,)`` with higher values
        meaning more anomalous.
    """
    import torch as _torch

    with _torch.inference_mode():
        if k < 1 or k > len(bank):
            raise ValueError("k must be between 1 and the bank size.")
        if queries.ndim != 2 or queries.shape[1] != bank.shape[1]:
            raise ValueError("Query and bank embedding dimensions must match.")
        if query_chunk_size < 1:
            raise ValueError(
                f"query_chunk_size must be positive, got {query_chunk_size}."
            )
        queries = queries.to(device=bank.device, dtype=_torch.float32)
        score_chunks = []
        for start in range(0, len(queries), query_chunk_size):
            query_chunk = queries[start : start + query_chunk_size]
            scores = distances_for_chunk(query_chunk, bank, k, bank_chunk_size)
            score_chunks.append(scores.cpu())
        scores = _torch.cat(score_chunks)
        if not _torch.isfinite(scores).all().item():
            raise ValueError("kNN returned invalid distances.")
        return scores
