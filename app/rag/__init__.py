"""RAG pipeline (Phase 1: ingestion / cleaning / chunking).

Phase roadmap (single-agent, decisions recorded 2026-09-08):
  P1  news/announcement ingestion -> documents/chunks tables (this module)
  P2  BGE-M3 embeddings stored in Qdrant LOCAL MODE (no Docker needed;
      switching to the Qdrant server later only changes the client URL)
  P3-P4  BM25 + dense + RRF + reranker
  P5-P6  RAG/Quant tools
  P7-P8  LangGraph agent + minimal UI
"""

from app.rag import chunker, cleaner, ingestion, sources  # noqa: F401

__all__ = ["chunker", "cleaner", "ingestion", "sources"]
