"""Tests for the BM25 index manifest and the consistency guard (Step 4).

The failure this prevents: the BM25 index and the chunk corpus drift apart and
retrieval keeps answering from a stale index - returning chunks that no longer
exist, or hydrating someone else's text - with no error at all.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.config import Config
from app.db import insert_chunks
from app.errors import IndexOutOfSync
from app.rag.bm25 import MANIFEST_FILE, BM25Index, chunk_id_hash

ROWS = [
    {
        "id": i,
        "chunk_id": f"c{i}",
        "stock_code": "000725",
        "doc_type": "announcement",
        "title": f"t{i}",
        "publish_time": "2026-08-01",
        "content": f"回购 进展 第{i}条",
    }
    for i in (1, 2, 3)
]


def _insert_db_rows(db_env, rows=ROWS, version: str | None = "bge-m3-v1") -> None:
    insert_chunks([
        {
            "chunk_id": row["chunk_id"],
            "document_id": f"d{row['id']}",
            "stock_code": row["stock_code"],
            "stock_name": "京东方A",
            "doc_type": row["doc_type"],
            "title": row["title"],
            "source": "cninfo",
            "publish_time": row["publish_time"],
            "url": "http://x",
            "content": row["content"],
            "chunk_index": 0,
            "chunk_hash": f"h{row['id']}",
            "embedding_version": version,
            "created_at": "2026-09-13 00:00:00",
        }
        for row in rows
    ])


def test_manifest_is_written_with_the_expected_fields(db_env, tmp_path) -> None:
    _insert_db_rows(db_env)
    index = BM25Index.build(ROWS, embedding_version="bge-m3-v1")
    target = index.save(tmp_path)

    manifest = json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["embedding_version"] == "bge-m3-v1"
    assert manifest["all_chunks"] is False
    assert manifest["chunk_count"] == 3
    assert manifest["chunk_id_hash"] == chunk_id_hash([row["chunk_id"] for row in ROWS])
    assert manifest["collection"] == Config.QDRANT_COLLECTION
    assert manifest["tokenizer"] == "jieba-cut_for_search"
    assert manifest["built_at"]


def test_load_verifies_a_matching_index(db_env, tmp_path) -> None:
    _insert_db_rows(db_env)
    BM25Index.build(ROWS, embedding_version="bge-m3-v1").save(tmp_path)

    loaded = BM25Index.load(tmp_path)   # must not raise

    assert len(loaded) == 3
    assert loaded.search("回购", top_k=1)[0]["chunk_id"] in {"c1", "c2", "c3"}


def test_load_raises_when_the_corpus_has_grown(db_env, tmp_path) -> None:
    """The guard: the index was built from 3 chunks, the corpus now has 4."""
    _insert_db_rows(db_env)
    BM25Index.build(ROWS, embedding_version="bge-m3-v1").save(tmp_path)
    _insert_db_rows(db_env, rows=[{
        "id": 4, "chunk_id": "c4", "stock_code": "000725",
        "doc_type": "announcement", "title": "t4",
        "publish_time": "2026-08-02", "content": "新增 chunk",
    }], version="bge-m3-v1")

    with pytest.raises(IndexOutOfSync) as excinfo:
        BM25Index.load(tmp_path)

    message = str(excinfo.value)
    assert "out of sync" in message and "build_bm25" in message
    assert excinfo.value.code == "IndexOutOfSync"


def test_allow_stale_downgrades_to_a_warning(db_env, tmp_path, monkeypatch, caplog) -> None:
    _insert_db_rows(db_env)
    BM25Index.build(ROWS, embedding_version="bge-m3-v1").save(tmp_path)
    _insert_db_rows(db_env, rows=[{
        "id": 4, "chunk_id": "c4", "stock_code": "000725",
        "doc_type": "announcement", "title": "t4",
        "publish_time": "2026-08-02", "content": "新增 chunk",
    }], version="bge-m3-v1")
    monkeypatch.setattr(Config, "BM25_ALLOW_STALE", True)

    with caplog.at_level(logging.WARNING, logger="app.rag.bm25"):
        loaded = BM25Index.load(tmp_path)

    assert len(loaded) == 3   # keeps working, but loudly
    assert any("out of sync" in r.getMessage() for r in caplog.records)


def test_missing_manifest_warns_but_still_loads(db_env, tmp_path, caplog) -> None:
    """Backward compatibility: an index built before Step 4 stays usable."""
    _insert_db_rows(db_env)
    BM25Index.build(ROWS, embedding_version="bge-m3-v1").save(tmp_path)
    (tmp_path / MANIFEST_FILE).unlink()

    with caplog.at_level(logging.WARNING, logger="app.rag.bm25"):
        loaded = BM25Index.load(tmp_path)

    assert len(loaded) == 3
    assert any("manifest is missing" in r.getMessage() for r in caplog.records)


def test_tampered_manifest_hash_is_detected(db_env, tmp_path) -> None:
    _insert_db_rows(db_env)
    index = BM25Index.build(ROWS, embedding_version="bge-m3-v1")
    target = index.save(tmp_path)
    manifest = json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["chunk_id_hash"] = "deadbeefdeadbeef"
    (target / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexOutOfSync):
        BM25Index.load(tmp_path)


def test_chunk_id_hash_is_order_independent() -> None:
    assert chunk_id_hash(["b", "a", "c"]) == chunk_id_hash(["c", "b", "a"])
    assert chunk_id_hash(["a"]) != chunk_id_hash(["b"])


def test_verify_reports_the_numbers(db_env, tmp_path) -> None:
    _insert_db_rows(db_env)
    index = BM25Index.build(ROWS, embedding_version="bge-m3-v1")
    manifest = index.manifest()

    report = index.verify(manifest)

    assert report["ok"] is True
    assert report["expected_count"] == report["current_count"] == 3
    assert report["expected_hash"] == report["current_hash"]


def test_hybrid_search_drops_hits_whose_text_cannot_be_hydrated(db_env, caplog) -> None:
    """A stale BM25 hit must not reach the LLM as an empty-body citation."""
    from app.rag.retrieval import hybrid_search

    # DB deliberately empty: the BM25 hit's point_id cannot be hydrated
    index = BM25Index.build(ROWS)
    captured: dict = {}

    class FakeEmbedder:
        dim = 4
        version = "test"

        def encode(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def encode_one(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    class FakeStore:
        dim = 4

        def search(self, *args, **kwargs):
            return []

    class FakeReranker:
        def rerank(self, query, candidates, top_k=None):
            return list(candidates)[:top_k], ""

    with caplog.at_level(logging.WARNING, logger="app.rag.retrieval"):
        captured = hybrid_search(
            "回购", top_k=5, embedder=FakeEmbedder(), store=FakeStore(),
            bm25_index=index, reranker=FakeReranker(),
        )

    assert captured["bm25_count"] == 3          # BM25 did retrieve them
    assert captured["dropped_empty_content"] == 3
    assert captured["results"] == []            # ...but they are not passed on
    assert any("could not be hydrated" in r.getMessage() for r in caplog.records)


class _Embedder:
    dim = 4
    version = "test"

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_one(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class _Store:
    """Two dense hits whose point ids match the inserted chunk rows."""

    dim = 4

    def search(self, vector, top_k=5, **kwargs):
        return [
            {"chunk_id": "c2", "document_id": "d2", "point_id": 2, "score": 0.82,
             "stock_code": "000725", "doc_type": "announcement"},
            {"chunk_id": "c1", "document_id": "d1", "point_id": 1, "score": 0.71,
             "stock_code": "000725", "doc_type": "announcement"},
        ][:top_k]


class _Reranker:
    def rerank(self, query, candidates, top_k=None):
        return list(candidates)[:top_k], ""


def test_empty_content_filter_is_a_noop_for_healthy_chunks(db_env) -> None:
    """With hydratable chunks, results must equal the unfiltered pipeline exactly."""
    from app.rag.hybrid import reciprocal_rank_fusion
    from app.rag.indexing import semantic_search
    from app.rag.retrieval import hybrid_search

    _insert_db_rows(db_env)                      # chunk rows for point ids 1..3
    index = BM25Index.build(ROWS)                # BM25 metadata carries those ids

    # reference result, computed from the same primitives without the new filter
    dense = semantic_search(_Embedder(), _Store(), "回购", top_k=20)
    for hit in dense:
        hit["dense_score"] = hit.pop("score", None)
    lexical = index.search("回购", top_k=20)
    expected = reciprocal_rank_fusion([dense, lexical], k=60, top_k=20)

    outcome = hybrid_search(
        "回购", top_k=20, use_rerank=False,
        embedder=_Embedder(), store=_Store(), bm25_index=index, reranker=_Reranker(),
    )

    assert outcome["dropped_empty_content"] == 0                      # nothing dropped
    assert [r["chunk_id"] for r in outcome["results"]] == [r["chunk_id"] for r in expected]
    for got, exp in zip(outcome["results"], expected):
        assert got["rrf_score"] == pytest.approx(exp["rrf_score"])    # fusion score unchanged
        for key in ("dense_score", "bm25_score"):                     # per-stage scores unchanged
            if exp.get(key) is None:
                assert got.get(key) is None
            else:
                assert got[key] == pytest.approx(exp[key])
        assert got["document_id"] == exp["document_id"]               # metadata unchanged
        assert got["stock_code"] == exp["stock_code"]
        assert got["content"]                                         # body present
