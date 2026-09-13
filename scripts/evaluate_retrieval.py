"""Run the retrieval comparison: BM25-only vs Dense-only vs Hybrid vs Hybrid+Reranker.

Uses the frozen eval set (`data/eval/retrieval_eval.json`) and the metrics in
`app/rag/eval_metrics.py`; writes a JSON result file and prints markdown tables
(overall + per query category).

**Budget semantics (review batch 1, Issue 2).** The original script had one
`--top-k` doing three different jobs, which made the four configurations
incomparable: BM25/Dense retrieved 10 chunks while Hybrid retrieved 20+20 and
fused to 15 before truncating to 10. Because one document is split into many
chunks and the metrics are computed **per document**, more chunks meant more
distinct documents and therefore higher recall — the hybrid was being handed a
larger budget. Budgets are now explicit and separated:

    --candidate-k   per-retriever recall depth (dense and BM25 each take this many)
    --fusion-k      depth kept after RRF (the candidate set the reranker sees)
    --final-k       chunks actually returned/scored — IDENTICAL for all configs

Usage:
    python scripts/evaluate_retrieval.py                      # 4 configs, budget 20
    python scripts/evaluate_retrieval.py --candidate-k 20 --fusion-k 20 --final-k 20
    python scripts/evaluate_retrieval.py --final-k 10          # sensitivity check
    python scripts/evaluate_retrieval.py --ignore-filters      # harder setting
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
DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 20


def resolve_budgets(
    candidate_k: int,
    fusion_k: int | None,
    final_k: int,
    k_values: tuple[int, ...],
) -> dict[str, int]:
    """Validate and normalise the retrieval/scoring budgets.

    Raises ValueError with an actionable message; `main` turns it into a CLI error
    (a silent fallback here is exactly how the old budgets drifted apart).
    """
    if candidate_k < 1:
        raise ValueError(f"candidate-k must be >= 1, got {candidate_k}")
    if final_k < 1:
        raise ValueError(f"final-k must be >= 1, got {final_k}")
    fusion = candidate_k if fusion_k is None else fusion_k
    if candidate_k < final_k:
        raise ValueError(
            f"candidate-k ({candidate_k}) must be >= final-k ({final_k}); the single-source "
            "configurations retrieve exactly final-k chunks"
        )
    if fusion < final_k:
        raise ValueError(
            f"fusion-k ({fusion}) must be >= final-k ({final_k}); otherwise not every "
            "configuration can fill the shared budget"
        )
    if not k_values:
        raise ValueError("k-values must not be empty")
    if max(k_values) > final_k:
        raise ValueError(
            f"max(k-values)={max(k_values)} must be <= final-k={final_k}; otherwise the "
            "metric looks beyond the shared budget"
        )
    return {"candidate_k": candidate_k, "fusion_k": fusion, "final_k": final_k}


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
    budgets: dict[str, int],
    ignore_filters: bool = False,
) -> tuple[list[str], int, float]:
    """Return (ranked document ids, distinct document count, latency seconds).

    Every configuration returns at most `budgets["final_k"]` chunks, so the four
    configurations are scored under the same budget.
    """
    query = query_spec["query"]
    stock_code = None if ignore_filters else query_spec.get("stock_code")
    doc_type = None if ignore_filters else query_spec.get("doc_type")
    final_k = budgets["final_k"]
    started = time.perf_counter()

    if name == "bm25":
        hits = index.search(query, top_k=final_k, stock_code=stock_code, doc_type=doc_type)
        if hits:
            contents = fetch_chunk_contents(
                [int(hit["point_id"]) for hit in hits if hit.get("point_id")]
            )
            for hit in hits:
                hit["content"] = contents.get(int(hit["point_id"] or 0), "")
        doc_ids = _doc_ids(hits)

    elif name == "dense":
        hits = semantic_search(
            embedder, store, query, top_k=final_k,
            stock_code=stock_code, doc_type=doc_type,
        )
        doc_ids = _doc_ids(hits)

    elif name in ("hybrid", "hybrid_rerank"):
        outcome = hybrid_search(
            query, top_k=final_k, stock_code=stock_code, doc_type=doc_type,
            dense_top_k=budgets["candidate_k"],
            bm25_top_k=budgets["candidate_k"],
            fusion_top_k=budgets["fusion_k"],
            use_rerank=(name == "hybrid_rerank"),
            embedder=embedder, store=store, bm25_index=index, reranker=reranker,
        )
        doc_ids = _doc_ids(outcome["results"])

    else:
        raise ValueError(f"unknown config {name!r}")

    latency = time.perf_counter() - started
    return doc_ids, len(set(doc_ids)), latency


def _markdown_table(rows: dict[str, dict], metric_keys: list[str]) -> str:
    header = ("| config | " + " | ".join(metric_keys)
              + " | unique docs | median latency |")
    separator = ("| --- | " + " | ".join(["---"] * len(metric_keys))
                 + " | --- | --- |")
    lines = [header, separator]
    for name, values in rows.items():
        cells = [f"{values.get(key, 0.0):.4f}" for key in metric_keys]
        unique_docs = values.get("unique_docs", 0.0)
        latency = values.get("latency_ms", 0.0)
        lines.append(
            f"| {name} | " + " | ".join(cells)
            + f" | {unique_docs:.1f} | {latency:.0f} ms |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation (4 configurations)")
    parser.add_argument("--eval", default="data/eval/retrieval_eval.json")
    parser.add_argument("--out", default="data/eval/latest_results.json")
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
                        help="per-retriever recall depth (dense and BM25 each)")
    parser.add_argument("--fusion-k", type=int, default=None,
                        help="chunks kept after RRF (default: = candidate-k)")
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K,
                        help="chunks returned/scored — identical for every config")
    parser.add_argument("--top-k", type=int, default=None,
                        help="deprecated alias for --final-k")
    parser.add_argument("--k-values", default="5,10")
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries")
    parser.add_argument("--ignore-filters", action="store_true",
                        help="do not apply the spec's stock_code/doc_type filters "
                             "(harder: retrieval must discriminate across the whole corpus)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    configs = [name.strip() for name in args.configs.split(",") if name.strip()]
    k_values = tuple(int(value) for value in args.k_values.split(","))
    final_k = args.top_k if args.top_k is not None else args.final_k
    if args.top_k is not None:
        print("[warn] --top-k is deprecated; use --final-k (budget semantics were split "
              "into candidate-k / fusion-k / final-k)", file=sys.stderr)

    try:
        budgets = resolve_budgets(args.candidate_k, args.fusion_k, final_k, k_values)
    except ValueError as exc:
        parser.error(str(exc))

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
    print(
        f"eval set: {eval_path} | queries={len(queries)} | configs={configs} | "
        f"candidate_k={budgets['candidate_k']} fusion_k={budgets['fusion_k']} "
        f"final_k={budgets['final_k']} k_values={list(k_values)} | "
        f"filters={'off' if args.ignore_filters else 'on'}"
    )

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
                doc_ids, unique_docs, latency = run_config(
                    name, spec, embedder=embedder, store=store,
                    index=index, reranker=reranker, budgets=budgets,
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
                        "unique_docs": unique_docs,
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
        summary["unique_docs"] = median_latency([float(row["unique_docs"]) for row in rows])
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
                "budgets": budgets,
                "k_values": list(k_values),
                "filters_applied": not args.ignore_filters,
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
