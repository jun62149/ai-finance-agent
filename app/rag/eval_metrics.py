"""Retrieval evaluation metrics (Phase 9).

Relevance is labelled at the **document** level (see scripts/build_eval_set.py),
while retrieval returns **chunks**. A retrieved chunk counts as relevant when its
document is one of the labelled documents, which is the standard way to score a
chunked corpus without pretending every chunk of a document is equally relevant.

Implemented: Recall@K, Precision@K, MRR, NDCG@K (binary gains), plus a small
aggregator that reports per-category means and medians.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

DEFAULT_K_VALUES = (5, 10)


def score_run(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Score one ranked chunk list against the labelled documents.

    `ranked_document_ids` is the document id of each retrieved chunk, in rank
    order. The ranking is **collapsed to distinct documents on first
    occurrence** before scoring: retrieval returns many chunks per document, and
    without collapsing, one relevant document could fill the whole top-k (and
    push NDCG above 1). Collapsing makes every metric answer the same, readable
    question: how well are distinct documents ranked?
    """
    relevant = set(relevant_document_ids)
    scores: dict[str, float] = {}
    if not relevant:
        return scores

    ranked: list[str] = []
    seen: set[str] = set()
    for document_id in ranked_document_ids:
        key = str(document_id)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(key)

    for k in k_values:
        top_k = ranked[:k]
        hits = [doc for doc in top_k if doc in relevant]
        scores[f"recall@{k}"] = len(hits) / len(relevant)
        scores[f"precision@{k}"] = len(hits) / k
        scores[f"ndcg@{k}"] = _ndcg(top_k, relevant, k)

    first_hit = next((rank for rank, doc in enumerate(ranked, start=1) if doc in relevant), None)
    scores["mrr"] = 1.0 / first_hit if first_hit else 0.0
    return scores


def _ndcg(ranked_documents: Sequence[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if doc in relevant else 0.0 for doc in ranked_documents[:k]]
    dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def aggregate(
    per_query: Sequence[dict[str, Any]],
    *,
    metric_keys: Sequence[str] | None = None,
    group_key: str | None = None,
) -> dict[str, Any]:
    """Mean of each metric across queries, optionally grouped (e.g. by category).

    Each entry of `per_query` must contain the metric keys at the top level and,
    when grouping, the group value under `group_key`.
    """
    if metric_keys is None:
        metric_keys = sorted(
            {key for entry in per_query for key in entry if "@" in key or key == "mrr"}
        )

    def mean_for(entries: Sequence[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in metric_keys:
            values = [float(entry[key]) for entry in entries if entry.get(key) is not None]
            out[key] = statistics.fmean(values) if values else 0.0
        latencies = [float(entry["latency_ms"]) for entry in entries if entry.get("latency_ms") is not None]
        if latencies:
            out["latency_ms"] = statistics.median(latencies)
        out["queries"] = len(entries)
        return out

    if not group_key:
        return mean_for(per_query)

    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in per_query:
        groups.setdefault(str(entry.get(group_key, "unknown")), []).append(entry)
    return {name: mean_for(entries) for name, entries in sorted(groups.items())}


def median_latency(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0
