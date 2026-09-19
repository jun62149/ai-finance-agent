"""Hybrid retrieval service: dense + BM25 -> RRF -> reranker (Phase 3).

This is the single entry point the RAG tool (P5) and the evaluation script use,
so every consumer sees identical semantics and scores.

Returned evidence items carry traceable metadata (stock_code / title /
publish_time / source / url) plus the score of every stage it passed through
(`dense_score`, `bm25_score`, `rrf_score`, `rerank_score`) — required for both
citation and offline evaluation.
"""

from __future__ import annotations

import time
from typing import Any

from app.config import Config
from app.log import get_logger
from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.indexing import semantic_search
from app.rag.reranker import Reranker
from app.rag.vector_store import ChunkVectorStore
from app.timeutil import market_today

_COMPONENTS: dict[str, Any] = {}
_LOGGER = get_logger(__name__)


def get_components(
    *,
    device: str | None = None,
    with_reranker: bool = True,
) -> dict[str, Any]:
    """Lazily build and cache the retrieval components (API/tool friendly)."""
    cache_key = f"{device}|{with_reranker}"
    if cache_key not in _COMPONENTS:
        _COMPONENTS[cache_key] = {
            "embedder": Embedder(device=device),
            "store": ChunkVectorStore(),
            "bm25": BM25Index.load(),
            "reranker": Reranker(device=device, enabled=with_reranker),
        }
    return _COMPONENTS[cache_key]


def close_components() -> None:
    for components in _COMPONENTS.values():
        store = components.get("store")
        if store is not None:
            store.close()
    _COMPONENTS.clear()


def hybrid_search(
    query: str,
    *,
    top_k: int | None = None,
    stock_code: str | None = None,
    doc_type: str | None = None,
    days: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    dense_top_k: int | None = None,
    bm25_top_k: int | None = None,
    fusion_top_k: int | None = None,
    rrf_k: int | None = None,
    use_rerank: bool = True,
    embedder: Any = None,
    store: Any = None,
    bm25_index: Any = None,
    reranker: Any = None,
) -> dict[str, Any]:
    """Run the full retrieval pipeline for one query.

    `days` is a convenience window (published within the last N days, relative to
    today); `date_from` / `date_to` (YYYY-MM-DD) override it. Dense applies the
    window through the Qdrant `publish_ts` payload field, BM25 post-filters.
    """
    timings: dict[str, float] = {}
    if days is not None and not (date_from or date_to):
        from datetime import timedelta

        # "the last N days" means market calendar days (UTC+8), so the window is
        # identical in a UTC container and on a UTC+8 host.
        date_from = (market_today() - timedelta(days=int(days))).isoformat()

    if embedder is None or store is None or bm25_index is None or reranker is None:
        components = get_components(with_reranker=use_rerank)
        embedder = embedder or components["embedder"]
        store = store or components["store"]
        bm25_index = bm25_index or components["bm25"]
        reranker = reranker or components["reranker"]

    dense_top_k = dense_top_k or Config.DENSE_TOP_K
    bm25_top_k = bm25_top_k or Config.BM25_TOP_K
    fusion_top_k = fusion_top_k or Config.FUSION_TOP_K
    final_k = top_k or Config.RERANK_TOP_K

    started = time.perf_counter()
    dense_hits = semantic_search(
        embedder, store, query, top_k=dense_top_k,
        stock_code=stock_code, doc_type=doc_type,
        date_from=date_from, date_to=date_to,
    )
    for hit in dense_hits:
        hit["dense_score"] = hit.pop("score", None)
    timings["dense_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    bm25_diagnostics: dict[str, Any] = {}
    if bm25_index is not None:
        if hasattr(bm25_index, "search_with_diagnostics"):
            lexical_hits, bm25_diagnostics = bm25_index.search_with_diagnostics(
                query, top_k=bm25_top_k, stock_code=stock_code, doc_type=doc_type,
                date_from=date_from, date_to=date_to,
            )
        else:  # any BM25 implementation that only exposes search()
            lexical_hits = bm25_index.search(
                query, top_k=bm25_top_k, stock_code=stock_code, doc_type=doc_type,
                date_from=date_from, date_to=date_to,
            )
    else:
        lexical_hits = []
    timings["bm25_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    fused = reciprocal_rank_fusion(
        [dense_hits, lexical_hits], k=rrf_k or Config.RRF_K, top_k=fusion_top_k
    )
    # BM25-only hits carry no chunk text; hydrate it from SQLite.
    missing_text = [item for item in fused if not item.get("content")]
    if missing_text:
        from app.db import fetch_chunk_contents

        contents = fetch_chunk_contents(
            [int(item["point_id"]) for item in missing_text if item.get("point_id")]
        )
        for item in missing_text:
            point_id = item.get("point_id")
            if point_id is not None:
                item["content"] = contents.get(int(point_id), "")

    # A hit whose text cannot be hydrated means the index points at a chunk that
    # no longer exists (stale index). Returning an empty body would let a broken
    # citation reach the LLM, so drop it and make the loss visible.
    empty_ids = {
        str(item.get("chunk_id")) for item in fused
        if not str(item.get("content") or "").strip()
    }
    if empty_ids:
        fused = [item for item in fused if str(item.get("chunk_id")) not in empty_ids]
        _LOGGER.warning(
            "dropped %d retrieved chunk(s) whose content could not be hydrated "
            "(stale index? run scripts/build_bm25.py): %s",
            len(empty_ids), ", ".join(sorted(empty_ids)[:3]),
        )
    timings["fusion_ms"] = (time.perf_counter() - started) * 1000

    results = fused
    rerank_error = ""
    if use_rerank:
        started = time.perf_counter()
        results, rerank_error = reranker.rerank(query, fused, top_k=final_k)
        timings["rerank_ms"] = (time.perf_counter() - started) * 1000
    else:
        results = fused[:final_k]

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return {
        "query": query,
        "stock_code": stock_code,
        "doc_type": doc_type,
        "days": days,
        "date_from": date_from,
        "date_to": date_to,
        "dense_count": len(dense_hits),
        "bm25_count": len(lexical_hits),
        # BM25 diagnostics (additive): window/truncation plus how many candidates
        # were scanned and how many survived the metadata filters. They make the
        # "why did this query miss?" question answerable from the trace.
        "bm25_window": bm25_diagnostics.get("window"),
        "bm25_truncated": bm25_diagnostics.get("truncated"),
        "bm25_scanned": bm25_diagnostics.get("scanned"),
        "bm25_after_filter": bm25_diagnostics.get("after_filter"),
        "bm25_corpus": bm25_diagnostics.get("corpus"),
        "fused_count": len(fused),
        "dropped_empty_content": len(empty_ids),
        "reranked": bool(use_rerank and not rerank_error),
        "rerank_error": rerank_error,
        "timings": timings,
        "results": results,
    }
