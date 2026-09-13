"""Unit tests for the Phase 2 vector store + indexing pipeline.

A deterministic fake embedder keeps these tests offline and torch-free while
still exercising the real Qdrant local-mode client, the SQLite chunk table and
the resume/idempotency rules.
"""

from __future__ import annotations

import pytest

from app.db import embedding_stats, insert_chunks, insert_documents
from app.rag.indexing import build_embeddings, semantic_search
from app.rag.vector_store import ChunkVectorStore

VERSION = "test-v1"
DIM = 4

DOCS = [
    {"code": "600519", "title": "半年度报告", "text": "本期营业收入同比增长。", "date": "2026-08-15"},
    {"code": "000725", "title": "回购进展公告", "text": "公司回购股份进展顺利。", "date": "2026-08-20"},
    {"code": "000725", "title": "股东减持公告", "text": "股东计划减持部分股份。", "date": "2026-08-21"},
]


class FakeEmbedder:
    """Keyword-driven embeddings: revenue / buyback / other -> unit vectors."""

    version = VERSION
    dim = DIM

    def encode(self, texts):
        out = []
        for text in texts:
            if "营业" in text or "营收" in text:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "回购" in text:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0, 0.0])
        return out

    def encode_one(self, text):
        return self.encode([text])[0]


@pytest.fixture()
def chunk_db(db_env):
    """Insert three synthetic documents with one chunk each."""
    for i, doc in enumerate(DOCS, start=1):
        document_id = f"doc-{i}"
        insert_documents(
            [
                {
                    "document_id": document_id,
                    "stock_code": doc["code"],
                    "stock_name": "测试",
                    "doc_type": "announcement",
                    "title": doc["title"],
                    "source": "cninfo",
                    "publish_time": doc["date"],
                    "content": doc["text"],
                    "url": f"http://example.com/{i}",
                    "content_hash": f"hash-{i}",
                    "created_at": "2026-09-10 00:00:00",
                }
            ]
        )
        insert_chunks(
            [
                {
                    "chunk_id": f"{document_id}:c000",
                    "document_id": document_id,
                    "stock_code": doc["code"],
                    "stock_name": "测试",
                    "doc_type": "announcement",
                    "title": doc["title"],
                    "source": "cninfo",
                    "publish_time": doc["date"],
                    "url": f"http://example.com/{i}",
                    "content": doc["text"],
                    "chunk_index": 0,
                    "chunk_hash": f"chunk-hash-{i}",
                    "embedding_version": None,
                    "created_at": "2026-09-10 00:00:00",
                }
            ]
        )
    return db_env


@pytest.fixture()
def store(tmp_path):
    s = ChunkVectorStore(
        path=str(tmp_path / "qdrant"), collection="test_chunks", dim=DIM
    )
    yield s
    s.close()


def test_build_embeddings_indexes_and_marks_version(chunk_db, store) -> None:
    summary = build_embeddings(FakeEmbedder(), store, version=VERSION)

    assert summary["processed"] == 3
    assert summary["indexed_points"] == 3
    stats = embedding_stats(VERSION)
    assert stats["announcement"] == {"total": 3, "embedded": 3, "pending": 0}


def test_build_embeddings_is_idempotent(chunk_db, store) -> None:
    build_embeddings(FakeEmbedder(), store, version=VERSION)
    second = build_embeddings(FakeEmbedder(), store, version=VERSION)

    assert second["processed"] == 0  # nothing pending
    assert store.count() == 3  # no duplicate points


def test_version_bump_reembeds(chunk_db, store) -> None:
    build_embeddings(FakeEmbedder(), store, version=VERSION)
    bumped = build_embeddings(FakeEmbedder(), store, version="test-v2")

    assert bumped["processed"] == 3  # chunks are pending for the new version


def test_limit_and_resume(chunk_db, store) -> None:
    first = build_embeddings(FakeEmbedder(), store, version=VERSION, limit=2)
    assert first["processed"] == 2
    assert embedding_stats(VERSION)["announcement"]["pending"] == 1

    second = build_embeddings(FakeEmbedder(), store, version=VERSION)
    assert second["processed"] == 1  # the remaining chunk
    assert embedding_stats(VERSION)["announcement"]["pending"] == 0


def test_semantic_search_hydrates_content_and_metadata(chunk_db, store) -> None:
    build_embeddings(FakeEmbedder(), store, version=VERSION)

    hits = semantic_search(FakeEmbedder(), store, "营收 增长", top_k=2)

    assert hits
    top = hits[0]
    assert top["title"] == "半年度报告"
    assert top["stock_code"] == "600519"
    assert top["content"] == DOCS[0]["text"]  # hydrated from SQLite
    assert top["publish_time"] == "2026-08-15"
    assert top["source"] == "cninfo"


def test_semantic_search_filters(chunk_db, store) -> None:
    build_embeddings(FakeEmbedder(), store, version=VERSION)

    hits = semantic_search(FakeEmbedder(), store, "回购 进展", top_k=5, stock_code="000725")

    assert hits
    assert {h["stock_code"] for h in hits} == {"000725"}
    assert hits[0]["title"] == "回购进展公告"


def test_dimension_mismatch_raises(chunk_db, store) -> None:
    class WrongDim(FakeEmbedder):
        dim = 8

        def encode(self, texts):
            return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(ValueError, match="dimension mismatch"):
        build_embeddings(WrongDim(), store, version=VERSION)
