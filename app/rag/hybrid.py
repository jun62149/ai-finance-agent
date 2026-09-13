"""Hybrid retrieval fusion + reranking orchestration (Phase 3).

Pipeline (design doc §9): dense recall + BM25 recall -> reciprocal rank fusion
-> reranker -> top-k evidence. Each stage keeps its own score so the evaluation
can compare BM25-only / Dense-only / Hybrid / Hybrid+Reranker.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.config import Config


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    k: int = 60,
    top_k: int | None = None,
    score_keys: Sequence[str] = ("dense_score", "bm25_score"),
) -> list[dict[str, Any]]:
    """Fuse ranked lists by RRF: score = sum(1 / (k + rank)) across lists.

    Items are matched by `chunk_id`; per-retriever scores are preserved on the
    merged item (`dense_score`, `bm25_score`) plus the fused `rrf_score`.
    """
    merged: dict[str, dict[str, Any]] = {}
    for list_index, results in enumerate(result_lists):
        score_key = score_keys[list_index] if list_index < len(score_keys) else None
        for rank, item in enumerate(results, start=1):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            entry = merged.get(chunk_id)
            if entry is None:
                entry = dict(item)
                entry["rrf_score"] = 0.0
                merged[chunk_id] = entry
            entry["rrf_score"] += 1.0 / (k + rank)
            if score_key and item.get(score_key) is not None:
                entry[score_key] = item[score_key]

    fused = sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)
    return fused[:top_k] if top_k else fused
