"""Hybrid retrieval CLI: dense + BM25 -> RRF -> reranker (Phase 3).

Examples:
    python scripts/search_hybrid.py "贵州茅台 半年度 营业收入"
    python scripts/search_hybrid.py "回购 进展" --stock-code 000725 --doc-type announcement
    python scripts/search_hybrid.py "重大资产重组" --no-rerank
    python scripts/search_hybrid.py "分红 派息" --mode bm25      # dense | bm25 | hybrid

The BM25 index must exist (scripts/build_bm25.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.indexing import semantic_search
from app.rag.reranker import Reranker
from app.rag.retrieval import hybrid_search
from app.rag.vector_store import ChunkVectorStore


def _preview(text: str, width: int = 200) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[:width] + "…"


def _print_results(results: list[dict], preview: int) -> None:
    for item in results:
        scores = " ".join(
            f"{name}={item[key]:.4f}"
            for name, key in (
                ("dense", "dense_score"),
                ("bm25", "bm25_score"),
                ("rrf", "rrf_score"),
                ("rerank", "rerank_score"),
            )
            if item.get(key) is not None
        )
        print("")
        print(f"[{item.get('rank')}] {item.get('stock_code')} {item.get('stock_name')} "
              f"| {item.get('doc_type')} | {item.get('publish_time')}")
        print(f"    scores: {scores}")
        print(f"    title : {item.get('title')}")
        print(f"    url   : {item.get('url')}")
        print(f"    text  : {_preview(str(item.get('content') or ''), preview)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval (dense + BM25 + RRF + rerank)")
    parser.add_argument("query")
    parser.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--bm25-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--stock-code", default=None)
    parser.add_argument("--doc-type", default=None, choices=["news", "announcement"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--preview", type=int, default=200)
    args = parser.parse_args()

    top_k = args.top_k or Config.RERANK_TOP_K
    use_rerank = not args.no_rerank
    t0 = time.time()

    if args.mode == "hybrid":
        outcome = hybrid_search(
            args.query,
            top_k=top_k,
            stock_code=args.stock_code,
            doc_type=args.doc_type,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            fusion_top_k=args.fusion_top_k,
            use_rerank=use_rerank,
            embedder=Embedder(device=args.device),
            store=ChunkVectorStore(),
            bm25_index=BM25Index.load(),
            reranker=Reranker(device=args.device, enabled=use_rerank),
        )
        results = outcome["results"]
        print(
            f"mode=hybrid query={args.query!r} dense={outcome['dense_count']} "
            f"bm25={outcome['bm25_count']} fused={outcome['fused_count']} "
            f"reranked={outcome['reranked']}"
            + (f" rerank_error={outcome['rerank_error']}" if outcome["rerank_error"] else "")
        )
        print(
            "timings: "
            + " ".join(f"{k}={v:.0f}ms" for k, v in outcome["timings"].items())
            + f" total={1000 * (time.time() - t0):.0f}ms"
        )
        _print_results(results, args.preview)
        return

    if args.mode == "dense":
        embedder = Embedder(device=args.device)
        with ChunkVectorStore() as store:
            hits = semantic_search(
                embedder, store, args.query, top_k=top_k,
                stock_code=args.stock_code, doc_type=args.doc_type,
            )
        for rank, hit in enumerate(hits, start=1):
            hit["dense_score"] = hit.pop("score", None)
            hit["rank"] = rank
        print(f"mode=dense query={args.query!r} hits={len(hits)}")
        _print_results(hits, args.preview)
        return

    index = BM25Index.load()
    hits = index.search(
        args.query, top_k=top_k, stock_code=args.stock_code, doc_type=args.doc_type
    )
    if hits:
        from app.db import fetch_chunk_contents

        contents = fetch_chunk_contents(
            [int(hit["point_id"]) for hit in hits if hit.get("point_id")]
        )
        for hit in hits:
            hit["content"] = contents.get(int(hit["point_id"] or 0), "")
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
    print(f"mode=bm25 query={args.query!r} hits={len(hits)}")
    _print_results(hits, args.preview)


if __name__ == "__main__":
    main()
