"""RAG API (design doc §20): direct access to the knowledge-base retrieval tool.

POST /api/rag/search exposes exactly the `search_financial_knowledge` tool used
by the Agent, so the HTTP contract and the tool contract can never drift. Errors
use the same structured `{code, message}` shape as the Quant API.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.tools import rag_tools


router = APIRouter(prefix="/api")


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="自然语言检索问题")
    stock_code: str | None = Field(default=None, description="6 位股票代码过滤")
    doc_type: Literal["news", "announcement"] | None = Field(default=None)
    days: int | None = Field(default=None, ge=1, le=rag_tools.MAX_DAYS,
                             description="只检索最近 N 天发布的文档")
    top_k: int = Field(default=5, ge=1, le=rag_tools.MAX_TOP_K)


class RagEvidence(BaseModel):
    chunk_id: str
    document_id: str
    stock_code: str
    stock_name: str | None = None
    doc_type: str
    title: str
    publish_time: str
    source: str
    url: str
    content: str = ""
    rank: int | None = None
    citation: str = ""
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RagSearchResponse(BaseModel):
    query: str
    filters: dict[str, Any]
    count: int
    has_evidence: bool
    results: list[RagEvidence]
    retrieval: dict[str, Any]
    embedding_version: str


@router.post("/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    """Hybrid retrieval (dense + BM25 + RRF + reranker) over news/announcements."""
    outcome = rag_tools.search_financial_knowledge(
        query=request.query,
        stock_code=request.stock_code,
        doc_type=request.doc_type,
        days=request.days,
        top_k=request.top_k,
    )
    return RagSearchResponse(**outcome)
