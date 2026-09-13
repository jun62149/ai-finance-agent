"""Run the retrieval comparison: BM25-only vs Dense-only vs Hybrid vs Hybrid+Reranker.

Uses the frozen eval set (`data/eval/retrieval_eval.json`) and the metrics in
`app/rag/eval_metrics.py`; writes a JSON result file and prints markdown tables
(overall + per query category) that can be pasted into the README.

Usage:
    python scripts/evaluate_retrieval.py                       # 4 configs, k=5/10
    python scripts/evaluate_retrieval.py --top-k 10 --device cuda
    python scripts/evaluate_retrieval.py --configs bm25,hybrid --limit 10
    python scripts/evaluate_retrieval.py --out data/eval/results_2026-09-10.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.db import fetch_chunk_contents
from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder
from app.rag.eval_metrics import aggregate, median_latency, score_run
from app.rag.indexing import semantic_search
from app.rag.reranker import Reranker
from app.rag.retrieval import hybrid_search
from app.rag.vector_store import ChunkVectorStore

ALL_CONFIGS = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def _doc_ids(hits: list[dict]) -> list[str]:
    return [str(hit.get("document_id") or "") for hit in hits]


def run_config(
    name: str,
    query_spec: dict,
    *,
    embedder,
    store,
    index,
    reranker,
    top_k: int,
    ignore_filters: bool = False,
) -> tuple[list[str], float]:
    """Return (ranked document ids, latency seconds) for one config/query."""
    query = query_spec["query"]
    stock_code = None if ignore_filters else query_spec.get("stock_code")
    doc_type = None if ignore_filters else query_spec.get("doc_type")
    started = time.perf_counter()

    if name == "bm25":
        hits = index.search(query, top_k=top_k, stock_code=stock_code, doc_type=doc_type)
        if hits:
            contents = fetch_chunk_contents(
                [int(hit["point_id"]) for hit in hits if hit.get("point_id")]
            )
            for hit in hits:
                hit["content"] = contents.get(int(hit["point_id"] or 0), "")
        doc_ids = _doc_ids(hits)

    elif name == "dense":
        hits = semantic_search(
            embedder, store, query, top_k=top_k,
            stock_code=stock_code, doc_type=doc_type,
        )
        doc_ids = _doc_ids(hits)

    elif name == "hybrid":
        outcome = hybrid_search(
            query, top_k=top_k, stock_code=stock_code, doc_type=doc_type,
            use_rerank=False, embedder=embedder, store=store,
            bm25_index=index, reranker=reranker,
        )
        doc_ids = _doc_ids(outcome["results"])

    elif name == "hybrid_rerank":
        outcome = hybrid_search(
            query, top_k=top_k, stock_code=stock_code, doc_type=doc_type,
            use_rerank=True, embedder=embedder, store=store,
            bm25_index=index, reranker=reranker,
        )
        doc_ids = _doc_ids(outcome["results"])

    else:
        raise ValueError(f"unknown config {name!r}")

    return doc_ids, time.perf_counter() - started


def _markdown_table(rows: dict[str, dict], metric_keys: list[str]) -> str:
    header = "| config | " + " | ".join(metric_keys) + " | median latency |"
    separator = "| --- | " + " | ".join(["---"] * len(metric_keys)) + " | --- |"
    lines = [header, separator]
    for name, values in rows.items():
        cells = [f"{values.get(key, 0.0):.4f}" for key in metric_keys]
        latency = values.get("latency_ms", 0.0)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {latency:.0f} ms |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation (4 configurations)")
    parser.add_argument("--eval", default="data/eval/retrieval_eval.json")
    parser.add_argument("--out", default="data/eval/latest_results.json")
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", default="5,10")
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries")
    parser.add_argument("--ignore-filters", action="store_true",
                        help="do not apply the spec's stock_code/doc_type filters "
                             "(harder: retrieval must discriminate across the whole corpus)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    configs = [name.strip() for name in args.configs.split(",") if name.strip()]
    k_values = tuple(int(value) for value in args.k_values.split(","))
    metric_keys = (
        [f"recall@{k}" for k in k_values]
        + [f"precision@{k}" for k in k_values]
        + ["mrr"]
        + [f"ndcg@{k}" for k in k_values]
    )

    eval_path = Path(args.eval)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    queries = payload["queries"]
    if args.limit:
        queries = queries[: args.limit]
    print(f"eval set: {eval_path} | queries={len(queries)} | configs={configs} | top_k={args.top_k}")

    embedder = Embedder(device=args.device)
    index = BM25Index.load()
    reranker = Reranker(device=args.device)
    store = ChunkVectorStore()
    try:
        # warm up the models so latency numbers describe retrieval, not loading
        semantic_search(embedder, store, "预热查询", top_k=1)
        if "hybrid_rerank" in configs:
            reranker.rerank("预热", [{"content": "预热"}])

        per_config: dict[str, list[dict]] = {name: [] for name in configs}
        for position, spec in enumerate(queries, start=1):
            relevant = spec["relevant_document_ids"]
            for name in configs:
                doc_ids, latency = run_config(
                    name, spec, embedder=embedder, store=store,
                    index=index, reranker=reranker, top_k=args.top_k,
                    ignore_filters=args.ignore_filters,
                )
                scores = score_run(doc_ids, relevant, k_values=k_values)
                scores.update(
                    {
                        "id": spec["id"],
                        "category": spec["category"],
                        "query": spec["query"],
                        "latency_ms": latency * 1000,
                        "returned": len(doc_ids),
                        "relevant_count": spec["relevant_count"],
                    }
                )
                per_config[name].append(scores)
            print(f"  [{position}/{len(queries)}] {spec['id']} done", flush=True)
    finally:
        store.close()

    overall: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    for name, rows in per_config.items():
        summary = aggregate(rows, metric_keys=metric_keys)
        summary["latency_ms"] = median_latency([row["latency_ms"] for row in rows])
        overall[name] = summary
        by_category[name] = aggregate(rows, metric_keys=metric_keys, group_key="category")

    print("")
    print("## overall")
    print(_markdown_table(overall, metric_keys))
    for name in configs:
        print("")
        print(f"## by category — {name}")
        print(_markdown_table(by_category[name], metric_keys))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "eval_set": str(eval_path),
                "top_k": args.top_k,
                "filters_applied": not args.ignore_filters,
                "k_values": list(k_values),
                "queries": len(queries),
                "overall": overall,
                "by_category": by_category,
                "per_query": per_config,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
