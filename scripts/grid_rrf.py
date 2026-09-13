"""RRF parameter experiment (review batch 1, Step 3) - one-off experiment script.

Question: **why these RRF parameters?** Not "which cell scores highest".
Plan: reports/2026-09-13_Step3方案_RRF参数实验.md
Output: docs/rrf_parameter_experiment.md (+ JSON under data/eval/)

Design notes
- The retrieval protocol is frozen: candidate-k=20, fusion-k=20, final-k=20,
  BM25_WINDOW=0 (config default). The eval set is not touched.
- Dense top-20 and BM25 top-20 are computed **once per query**; the 12 grid cells
  only re-fuse those candidates in Python, so the whole grid costs about as much
  as a single normal evaluation.
- The weighted fusion lives here (not in app/): production code is only touched
  if the decision rule says a non-default cell must win.
- A self-check asserts that weights=(1,1) reproduces `app.rag.hybrid.reciprocal_rank_fusion`
  exactly, so the local implementation cannot silently drift from production.

Usage:
    python scripts/grid_rrf.py --setting filtered
    python scripts/grid_rrf.py --setting unfiltered
    python scripts/grid_rrf.py --setting filtered --rerank-shortlist
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.db import fetch_chunk_contents
from app.rag.bm25 import BM25Index
from app.rag.embedding import Embedder
from app.rag.eval_metrics import score_run
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.indexing import semantic_search
from app.rag.reranker import Reranker
from app.rag.vector_store import ChunkVectorStore

K_GRID = (10, 30, 60, 100)
WEIGHT_GRID = ((1.0, 1.0), (1.0, 2.0), (1.0, 3.0))
DEFAULT_CELL = (60, 1.0, 1.0)
METRIC_KEYS = ("recall@5", "recall@10", "precision@5", "mrr", "ndcg@5", "ndcg@10")


def weighted_rrf(
    dense_hits: Sequence[dict],
    bm25_hits: Sequence[dict],
    *,
    k: int,
    weights: tuple[float, float],
    top_k: int,
) -> list[dict]:
    """RRF with per-retriever weights: score = sum(w_r / (k + rank_r))."""
    merged: dict[str, dict] = {}
    for results, weight in ((dense_hits, weights[0]), (bm25_hits, weights[1])):
        for rank, item in enumerate(results, start=1):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            entry = merged.setdefault(chunk_id, dict(item))
            entry["rrf_score"] = entry.get("rrf_score", 0.0) + weight / (k + rank)
    ordered = sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)
    return ordered[:top_k]


def doc_ids(hits: Sequence[dict]) -> list[str]:
    return [str(hit.get("document_id") or "") for hit in hits]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def collect_candidates(
    queries: list[dict],
    *,
    embedder: Embedder,
    store: ChunkVectorStore,
    index: BM25Index,
    ignore_filters: bool,
    budget: int,
) -> tuple[list[dict], float]:
    """One pass over the eval set: dense top-N + BM25 top-N per query."""
    rows: list[dict] = []
    retrieval_latencies: list[float] = []
    for spec in queries:
        stock_code = None if ignore_filters else spec.get("stock_code")
        doc_type = None if ignore_filters else spec.get("doc_type")
        started = time.perf_counter()
        dense = semantic_search(
            embedder, store, spec["query"], top_k=budget,
            stock_code=stock_code, doc_type=doc_type,
        )
        lexical = index.search(
            spec["query"], top_k=budget, stock_code=stock_code, doc_type=doc_type
        )
        retrieval_latencies.append((time.perf_counter() - started) * 1000)

        for hit in dense:
            hit["dense_score"] = hit.pop("score", None)

        # hydrate BM25-only hits so the reranker shortlist can score them
        missing = [hit for hit in lexical if not hit.get("content")]
        if missing:
            contents = fetch_chunk_contents(
                [int(hit["point_id"]) for hit in missing if hit.get("point_id")]
            )
            for hit in missing:
                hit["content"] = contents.get(int(hit["point_id"] or 0), "")

        rows.append({"spec": spec, "dense": dense, "bm25": lexical})
    return rows, statistics.median(retrieval_latencies)


def evaluate_cell(rows: list[dict], *, k: int, weights: tuple[float, float], budget: int) -> dict:
    per_query: list[dict] = []
    fusion_ms: list[float] = []
    unique_docs: list[int] = []
    for row in rows:
        started = time.perf_counter()
        fused = weighted_rrf(row["dense"], row["bm25"], k=k, weights=weights, top_k=budget)
        fusion_ms.append((time.perf_counter() - started) * 1000)
        ids = doc_ids(fused)
        scores = score_run(ids, row["spec"]["relevant_document_ids"], k_values=(5, 10))
        scores["id"] = row["spec"]["id"]
        scores["top5_docs"] = sorted(set(ids[:5]))
        per_query.append(scores)
        unique_docs.append(len(set(ids)))
    summary = {key: mean([row[key] for row in per_query if row.get(key) is not None]) for key in METRIC_KEYS}
    summary["unique_docs"] = mean([float(value) for value in unique_docs])
    summary["fusion_ms"] = statistics.median(fusion_ms)
    return {"summary": summary, "per_query": per_query}


def fold_summary(cell: dict, fold: int) -> float:
    values = [
        row["recall@5"] for position, row in enumerate(cell["per_query"])
        if position % 2 == fold and row.get("recall@5") is not None
    ]
    return mean(values)


def rerank_cell(
    rows: list[dict],
    *,
    k: int,
    weights: tuple[float, float],
    budget: int,
    reranker: Reranker,
) -> dict:
    per_query: list[dict] = []
    rerank_ms: list[float] = []
    for row in rows:
        fused = weighted_rrf(row["dense"], row["bm25"], k=k, weights=weights, top_k=budget)
        started = time.perf_counter()
        ranked, error = reranker.rerank(row["spec"]["query"], fused, top_k=budget)
        rerank_ms.append((time.perf_counter() - started) * 1000)
        if error:
            raise RuntimeError(f"reranker failed: {error}")
        ids = doc_ids(ranked)
        scores = score_run(ids, row["spec"]["relevant_document_ids"], k_values=(5, 10))
        scores["id"] = row["spec"]["id"]
        per_query.append(scores)
    summary = {key: mean([r[key] for r in per_query if r.get(key) is not None]) for key in METRIC_KEYS}
    summary["rerank_ms"] = statistics.median(rerank_ms)
    return {"summary": summary, "per_query": per_query}


def markdown_grid(cells: dict[tuple, dict], retrieval_ms: float) -> str:
    header = ("| k | w_dense | w_bm25 | R@5 | R@10 | P@5 | MRR | NDCG@5 | NDCG@10 | unique docs | median latency |")
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for (k, wd, wb), cell in cells.items():
        s = cell["summary"]
        row = (
            f"| {k} | {wd:g} | {wb:g} | {s['recall@5']:.4f} | {s['recall@10']:.4f} | "
            f"{s['precision@5']:.4f} | {s['mrr']:.4f} | {s['ndcg@5']:.4f} | {s['ndcg@10']:.4f} | "
            f"{s['unique_docs']:.1f} | {retrieval_ms + s['fusion_ms']:.0f} ms |"
        )
        if (k, wd, wb) == DEFAULT_CELL:
            row = row.replace("| 60 | 1 | 1 |", "| **60** | **1** | **1** |") + "  ← 当前默认"
        lines.append(row)
    return "\n".join(lines)


def markdown_folds(cells: dict[tuple, dict]) -> str:
    lines = ["| k | w_dense | w_bm25 | fold0 R@5 | fold1 R@5 | 整体 R@5 |", "| --- | --- | --- | --- | --- | --- |"]
    for (k, wd, wb), cell in cells.items():
        lines.append(
            f"| {k} | {wd:g} | {wb:g} | {fold_summary(cell, 0):.4f} | {fold_summary(cell, 1):.4f} | "
            f"{cell['summary']['recall@5']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RRF parameter grid experiment")
    parser.add_argument("--eval", default="data/eval/retrieval_eval.json")
    parser.add_argument("--out", default=None, help="JSON output path")
    parser.add_argument("--setting", default="filtered", choices=["filtered", "unfiltered"])
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerank-shortlist", action="store_true",
                        help="also run the reranker for the default cell and the best cells")
    parser.add_argument("--rerank-cells", default=None,
                        help="extra cells to rerank, formatted k:w_dense:w_bm25 (comma separated)")
    args = parser.parse_args()

    payload = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    queries = payload["queries"]
    ignore_filters = args.setting == "unfiltered"
    print(
        f"eval={args.eval} queries={len(queries)} setting={args.setting} budget={args.budget} "
        f"RRF_K grid={list(K_GRID)} weights={[f'{a:g}:{b:g}' for a, b in WEIGHT_GRID]} "
        f"BM25_WINDOW={Config.BM25_WINDOW}"
    )

    embedder = Embedder(device=args.device)
    store = ChunkVectorStore()
    index = BM25Index.load()
    reranker = Reranker(device=args.device)

    # --- self-check: weights=(1,1) must reproduce production RRF exactly --------
    rows, retrieval_ms = collect_candidates(
        queries, embedder=embedder, store=store, index=index,
        ignore_filters=ignore_filters, budget=args.budget,
    )
    probe = rows[0]
    local = weighted_rrf(probe["dense"], probe["bm25"], k=DEFAULT_CELL[0],
                         weights=(1.0, 1.0), top_k=args.budget)
    production = reciprocal_rank_fusion(
        [probe["dense"], probe["bm25"]], k=DEFAULT_CELL[0], top_k=args.budget
    )
    assert [h["chunk_id"] for h in local] == [h["chunk_id"] for h in production], (
        "weighted_rrf(1,1) must match production reciprocal_rank_fusion"
    )
    print(f"self-check passed: weighted_rrf(1,1) == reciprocal_rank_fusion (query {probe['spec']['id']})")
    print(f"retrieval median (dense+bm25, both branches): {retrieval_ms:.0f} ms")

    # --- the grid --------------------------------------------------------------
    cells: dict[tuple, dict] = {}
    for k in K_GRID:
        for weights in WEIGHT_GRID:
            cells[(k, weights[0], weights[1])] = evaluate_cell(
                rows, k=k, weights=weights, budget=args.budget
            )

    print("")
    print(f"## RRF grid — {args.setting}")
    print(markdown_grid(cells, retrieval_ms))
    print("")
    print(f"## 2-fold stability (query index parity) — {args.setting}")
    print(markdown_folds(cells))

    default_cell = cells[DEFAULT_CELL]
    ranked = sorted(cells.items(), key=lambda item: item[1]["summary"]["recall@5"], reverse=True)
    best_key, best_cell = ranked[0]
    print("")
    print(f"default (k=60, 1:1) R@5 = {default_cell['summary']['recall@5']:.4f}")
    print(f"best    (k={best_key[0]}, {best_key[1]:g}:{best_key[2]:g}) R@5 = "
          f"{best_cell['summary']['recall@5']:.4f}  gap={best_cell['summary']['recall@5'] - default_cell['summary']['recall@5']:+.4f}")
    print("top-5 cells:")
    for key, cell in ranked[:5]:
        print(f"  k={key[0]:<4} w={key[1]:g}:{key[2]:g}  R@5={cell['summary']['recall@5']:.4f} "
              f"R@10={cell['summary']['recall@10']:.4f} MRR={cell['summary']['mrr']:.4f}")

    # --- per-query differences vs the default cell -----------------------------
    default_top5 = {row["id"]: row["top5_docs"] for row in default_cell["per_query"]}
    print("")
    print("queries whose top-5 document set differs from the default cell:")
    for key, cell in ranked[:4]:
        if key == DEFAULT_CELL:
            continue
        changed = [
            row["id"] for row in cell["per_query"]
            if row["top5_docs"] != default_top5.get(row["id"])
        ]
        print(f"  k={key[0]:<4} w={key[1]:g}:{key[2]:g} changed={len(changed)}/{len(queries)} ids={changed[:6]}")

    rerank_results: dict[str, dict] = {}
    if args.rerank_cells:
        extra = []
        for token in args.rerank_cells.split(","):
            k_str, wd_str, wb_str = token.strip().split(":")
            extra.append((int(k_str), float(wd_str), float(wb_str)))
        print("")
        print("## reranker for explicitly requested cells")
        print("| k | w_dense | w_bm25 | R@5 | R@10 | MRR | NDCG@5 | rerank ms |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key in extra:
            result = rerank_cell(rows, k=key[0], weights=(key[1], key[2]),
                                 budget=args.budget, reranker=reranker)
            rerank_results[f"{key[0]}|{key[1]:g}|{key[2]:g}"] = result["summary"]
            s2 = result["summary"]
            print(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {s2['recall@5']:.4f} | {s2['recall@10']:.4f} | "
                  f"{s2['mrr']:.4f} | {s2['ndcg@5']:.4f} | {s2['rerank_ms']:.0f} |")

    if args.rerank_shortlist:
        shortlist = [DEFAULT_CELL] + [key for key, _ in ranked[:3] if key != DEFAULT_CELL]
        print("")
        print("## reranker shortlist (hybrid+rerank)")
        print("| k | w_dense | w_bm25 | R@5 | R@10 | MRR | NDCG@5 | rerank ms |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key in shortlist:
            result = rerank_cell(rows, k=key[0], weights=(key[1], key[2]),
                                 budget=args.budget, reranker=reranker)
            rerank_results[f"{key[0]}|{key[1]:g}|{key[2]:g}"] = result["summary"]
            s = result["summary"]
            print(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {s['recall@5']:.4f} | {s['recall@10']:.4f} | "
                  f"{s['mrr']:.4f} | {s['ndcg@5']:.4f} | {s['rerank_ms']:.0f} |")

    out_path = Path(args.out or f"data/eval/rrf_grid_{args.setting}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "setting": args.setting,
                "eval_set": args.eval,
                "budget": args.budget,
                "bm25_window": Config.BM25_WINDOW,
                "k_grid": list(K_GRID),
                "weight_grid": [list(w) for w in WEIGHT_GRID],
                "default_cell": {"k": DEFAULT_CELL[0], "w_dense": DEFAULT_CELL[1], "w_bm25": DEFAULT_CELL[2]},
                "retrieval_ms": retrieval_ms,
                "cells": {
                    f"{k}|{wd:g}|{wb:g}": cell["summary"] for (k, wd, wb), cell in cells.items()
                },
                "folds": {
                    f"{k}|{wd:g}|{wb:g}": [fold_summary(cell, 0), fold_summary(cell, 1)]
                    for (k, wd, wb), cell in cells.items()
                },
                "per_query": {
                    f"{k}|{wd:g}|{wb:g}": cell["per_query"] for (k, wd, wb), cell in cells.items()
                },
                "rerank_shortlist": rerank_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("")
    print(f"wrote {out_path}")
    store.close()


if __name__ == "__main__":
    main()
