"""RAG tool: `search_financial_knowledge` (Phase 5).

Agent-facing wrapper around `app.rag.retrieval.hybrid_search`. Rules it enforces:

- every returned item carries traceable metadata (chunk/document id, stock,
  doc_type, title, publish_time, source, url) — no metadata, no evidence;
- every stage score is preserved (dense / bm25 / rrf / rerank) so the Agent can
  explain *why* a piece of evidence was surfaced and the offline evaluation
  stays reproducible;
- no results means an empty, explicit answer — never fabricated content;
- arguments are validated at the boundary (bad arguments raise
  `ToolArgumentError`, which the tool runner turns into a structured error).
"""

from __future__ import annotations

import re
from typing import Any

from app.config import Config
from app.rag.retrieval import get_components, hybrid_search
from app.tools.errors import EvidenceMetadataMissing, ToolArgumentError

SEARCH_TOOL_NAME = "search_financial_knowledge"
STOCK_CODE_RE = re.compile(r"^\d{6}$")
DOC_TYPES = ("news", "announcement")
MAX_TOP_K = 20
MAX_DAYS = 3650

#: Metadata that must be present on every piece of returned evidence.
REQUIRED_EVIDENCE_FIELDS = (
    "chunk_id",
    "document_id",
    "stock_code",
    "doc_type",
    "title",
    "source",
    "publish_time",
    "url",
)

_SCORE_FIELDS = ("dense_score", "bm25_score", "rrf_score", "rerank_score")

SEARCH_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "自然语言检索问题，例如「京东方A 回购进展」或「2026年半年报 营业收入」",
        },
        "stock_code": {
            "type": "string",
            "description": "可选，6 位股票代码过滤（如 000725）",
        },
        "doc_type": {
            "type": "string",
            "enum": list(DOC_TYPES),
            "description": "可选，限定新闻或公告",
        },
        "days": {
            "type": "integer",
            "description": f"可选，只检索最近 N 天发布的文档（1~{MAX_DAYS}）",
        },
        "top_k": {
            "type": "integer",
            "description": f"返回条数（1~{MAX_TOP_K}，默认 5）",
        },
    },
    "required": ["query"],
}


def _validate(
    query: str,
    stock_code: str | None,
    doc_type: str | None,
    days: int | None,
    top_k: int,
) -> tuple[str, str | None, str | None, int | None, int]:
    if not isinstance(query, str) or len(query.strip()) < 2:
        raise ToolArgumentError("query must be a non-empty string (>= 2 characters)")
    query = query.strip()
    if len(query) > 300:
        raise ToolArgumentError("query is too long (max 300 characters)")

    if stock_code is not None:
        stock_code = str(stock_code).strip()
        if not STOCK_CODE_RE.match(stock_code):
            raise ToolArgumentError(
                f"stock_code must be 6 digits, got {stock_code!r}"
            )
    if doc_type is not None:
        doc_type = str(doc_type).strip()
        if doc_type not in DOC_TYPES:
            raise ToolArgumentError(
                f"doc_type must be one of {list(DOC_TYPES)}, got {doc_type!r}"
            )
    if days is not None:
        try:
            days = int(days)
        except (TypeError, ValueError) as exc:
            raise ToolArgumentError(f"days must be an integer, got {days!r}") from exc
        if not 1 <= days <= MAX_DAYS:
            raise ToolArgumentError(f"days must be between 1 and {MAX_DAYS}, got {days}")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(f"top_k must be an integer, got {top_k!r}") from exc
    if not 1 <= top_k <= MAX_TOP_K:
        raise ToolArgumentError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
    return query, stock_code, doc_type, days, top_k


def _citation(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "").strip()
    date = str(item.get("publish_time") or "")[:10]
    source = str(item.get("source") or "")
    return f"{title}（{source}，{date}）"


def _to_evidence(item: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "stock_code": item.get("stock_code"),
        "stock_name": item.get("stock_name"),
        "doc_type": item.get("doc_type"),
        "title": item.get("title"),
        "publish_time": item.get("publish_time"),
        "source": item.get("source"),
        "url": item.get("url"),
        "chunk_index": item.get("chunk_index"),
        "content": item.get("content") or "",
        "rank": item.get("rank"),
        "citation": _citation(item),
    }
    for field in _SCORE_FIELDS:
        if item.get(field) is not None:
            evidence[field] = round(float(item[field]), 6)
    return evidence


def search_financial_knowledge(
    query: str,
    stock_code: str | None = None,
    doc_type: str | None = None,
    days: int | None = None,
    top_k: int = 5,
    *,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve financial news/announcement evidence for a query.

    Returns a dict with `results` (each item carrying source metadata and the
    scores of every retrieval stage) plus run diagnostics. An empty `results`
    list means "no evidence found" and must be reported as such by the Agent.
    """
    query, stock_code, doc_type, days, top_k = _validate(
        query, stock_code, doc_type, days, top_k
    )

    if components is None:
        cached = get_components(with_reranker=True)
        components = {
            "embedder": cached["embedder"],
            "store": cached["store"],
            "bm25_index": cached["bm25"],
            "reranker": cached["reranker"],
        }

    outcome = hybrid_search(
        query,
        top_k=top_k,
        stock_code=stock_code,
        doc_type=doc_type,
        days=days,
        **components,
    )

    results = [_to_evidence(item) for item in outcome["results"]]
    for item in results:
        missing = [field for field in REQUIRED_EVIDENCE_FIELDS if not item.get(field)]
        if missing:
            raise EvidenceMetadataMissing(
                f"evidence {item.get('chunk_id')!r} is missing metadata: {missing}"
            )

    return {
        "query": query,
        "filters": {
            "stock_code": stock_code,
            "doc_type": doc_type,
            "days": days,
        },
        "count": len(results),
        "has_evidence": bool(results),
        "results": results,
        "retrieval": {
            "dense": outcome["dense_count"],
            "bm25": outcome["bm25_count"],
            # additive diagnostics: how BM25 was bounded and how much it scanned
            "bm25_window": outcome.get("bm25_window"),
            "bm25_truncated": outcome.get("bm25_truncated"),
            "bm25_scanned": outcome.get("bm25_scanned"),
            "bm25_after_filter": outcome.get("bm25_after_filter"),
            "fused": outcome["fused_count"],
            "reranked": outcome["reranked"],
            "rerank_error": outcome["rerank_error"],
            "timings_ms": {key: round(value, 1) for key, value in outcome["timings"].items()},
        },
        "embedding_version": Config.EMBEDDING_VERSION,
    }
