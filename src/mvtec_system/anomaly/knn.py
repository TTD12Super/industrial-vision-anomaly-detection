"""Exact chunked Euclidean kNN used by the MVTec anomaly baseline."""

from __future__ import annotations

import torch


@torch.inference_mode()
def distances_for_chunk(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    bank_chunk_size: int,
) -> torch.Tensor:
    """Keep the k smallest distances while scanning the bank in chunks."""

    if bank_chunk_size < 1:
        raise ValueError("bank_chunk_size must be positive.")
    best_distances = torch.full(
        (len(queries), k), float("inf"), device=bank.device, dtype=torch.float32
    )
    for start in range(0, len(bank), bank_chunk_size):
        bank_chunk = bank[start : start + bank_chunk_size]
        distances = torch.cdist(queries, bank_chunk, p=2)
        candidates = torch.cat((best_distances, distances), dim=1)
        best_distances = candidates.topk(k, dim=1, largest=False).values
    return best_distances.mean(dim=1)


@torch.inference_mode()
def nearest_neighbor_scores(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int = 1,
    query_chunk_size: int = 512,
    bank_chunk_size: int = 16384,
) -> torch.Tensor:
    """Return exact Euclidean kNN scores on CPU, preserving notebook behavior."""

    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError("bank must be a non-empty two-dimensional tensor.")
    if queries.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("Query and bank embedding dimensions must match.")
    if len(queries) == 0:
        raise ValueError("queries must not be empty.")
    if k < 1 or k > len(bank):
        raise ValueError("k must be between 1 and the bank size.")
    if query_chunk_size < 1 or bank_chunk_size < 1:
        raise ValueError("chunk sizes must be positive.")
    bank = bank.to(dtype=torch.float32)
    queries = queries.to(device=bank.device, dtype=torch.float32)
    score_chunks = []
    for start in range(0, len(queries), query_chunk_size):
        query_chunk = queries[start : start + query_chunk_size]
        scores = distances_for_chunk(query_chunk, bank, k, bank_chunk_size)
        score_chunks.append(scores.cpu())
    scores = torch.cat(score_chunks)
    if not torch.isfinite(scores).all().item():
        raise ValueError("kNN returned invalid distances.")
    return scores
