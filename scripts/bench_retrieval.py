"""Latency benchmark for the retrieval stack (Phase 3).

First-call timings include model loading, so this script warms up before
measuring, and can compare the Qdrant **local mode** against the **server** with
the same in-process model (that comparison is what justified migrating).

Usage:
    python scripts/bench_retrieval.py                         # server (QDRANT_URL) vs local path
    python scripts/bench_retrieval.py --queries 5 --repeat 5
    python scripts/bench_retrieval.py --skip-local            # only the configured store
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder
from app.rag.indexing import semantic_search
from app.rag.reranker import Reranker
from app.rag.retrieval import hybrid_search
from app.rag.vector_store import ChunkVectorStore

QUERIES = [
    "贵州茅台 半年度 营业收入 增长",
    "回购 股份 实施结果",
    "重大资产重组 进展",
    "分红 派息 方案",
    "研发投入 同比 变化",
]


def _stats(samples: list[float]) -> str:
    if not samples:
        return "n/a"
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return (
        f"median={statistics.median(ordered):.0f}ms "
        f"min={ordered[0]:.0f}ms p95={p95:.0f}ms"
    )


def _bench_dense(embedder, store, queries, repeat: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeat):
        for query in queries:
            started = time.perf_counter()
            semantic_search(embedder, store, query, top_k=Config.DENSE_TOP_K)
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def _bench_bm25(index, queries, repeat: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeat):
        for query in queries:
            started = time.perf_counter()
            index.search(query, top_k=Config.BM25_TOP_K)
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def _bench_hybrid(embedder, store, index, reranker, queries, repeat: int, rerank: bool) -> list[float]:
    samples: list[float] = []
    for _ in range(repeat):
        for query in queries:
            started = time.perf_counter()
            hybrid_search(
                query,
                embedder=embedder,
                store=store,
                bm25_index=index,
                reranker=reranker,
                use_rerank=rerank,
            )
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def _run_suite(label: str, store: ChunkVectorStore, embedder, index, reranker, args) -> None:
    queries = QUERIES[: args.queries]
    print("")
    print(f"== {label} (points={store.count()}) ==")

    started = time.perf_counter()
    semantic_search(embedder, store, queries[0], top_k=Config.DENSE_TOP_K)
    print(f"warm-up (includes model load): {1000 * (time.perf_counter() - started):.0f}ms")

    print(f"dense  : {_stats(_bench_dense(embedder, store, queries, args.repeat))}")
    print(f"bm25   : {_stats(_bench_bm25(index, queries, args.repeat))}")
    if not args.no_rerank:
        hybrid_samples = _bench_hybrid(
            embedder, store, index, reranker, queries, args.repeat, rerank=True
        )
        print(f"hybrid+rerank: {_stats(hybrid_samples)}")
    print(
        f"hybrid : {_stats(_bench_hybrid(embedder, store, index, reranker, queries, args.repeat, rerank=False))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval latency benchmark")
    parser.add_argument("--queries", type=int, default=3, help="distinct queries per round")
    parser.add_argument("--repeat", type=int, default=3, help="rounds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--skip-local", action="store_true",
                        help="do not benchmark the local path store")
    args = parser.parse_args()

    embedder = Embedder(device=args.device)
    index = BM25Index.load()
    reranker = Reranker(device=args.device)
    print(
        f"embedding_model={embedder.model_name} device={embedder.device} "
        f"bm25_docs={len(index)} reranker={reranker.model_name} "
        f"enabled={reranker.enabled}"
    )

    target = Config.QDRANT_URL or f"path:{Config.QDRANT_PATH}"
    print(f"configured store: {target}")

    with ChunkVectorStore() as store:
        _run_suite(f"configured store ({target})", store, embedder, index, reranker, args)

    if not args.skip_local and Config.QDRANT_URL:
        local_exists = Path(Config.QDRANT_PATH).exists()
        if local_exists:
            with ChunkVectorStore(path=str(Config.QDRANT_PATH)) as local_store:
                _run_suite(
                    f"local mode (path:{Config.QDRANT_PATH})",
                    local_store, embedder, index, reranker, args,
                )


if __name__ == "__main__":
    main()
