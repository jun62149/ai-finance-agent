"""Regression tests for the BM25 candidate window (review batch 1, Issue 1).

The bug: BM25 scored only a `max(top_k*10, 200)` window *before* applying the
metadata filters, so a filter could silently hide documents that ranked outside
that window. These tests pin the behaviour permanently.
"""

from __future__ import annotations

import logging

import pytest

from app.config import Config
from app.rag.bm25 import BM25Index

CORPUS_STRONG = 240   # > the old 200-candidate window
CORPUS_TARGET = 10


def _rows() -> list[dict]:
    """Strong lexical matches for stock A (rank high), weak ones for stock B."""
    rows = [
        {
            "id": i,
            "chunk_id": f"a{i}",
            "stock_code": "000001",
            "doc_type": "news",
            "title": f"A{i}",
            "publish_time": "2026-08-01",
            "content": "回购 回购 回购 回购 回购 回购 回购 回购 回购 回购",
        }
        for i in range(CORPUS_STRONG)
    ]
    rows += [
        {
            "id": 1000 + i,
            "chunk_id": f"b{i}",
            "stock_code": "000002",
            "doc_type": "announcement",
            "title": f"B{i}",
            "publish_time": "2026-08-02",
            "content": "回购 " + "其他内容 " * 40,   # one occurrence, long document
        }
        for i in range(CORPUS_TARGET)
    ]
    return rows


@pytest.fixture()
def index() -> BM25Index:
    return BM25Index.build(_rows())


def test_window_mode_misses_the_target_full_mode_finds_it(index, monkeypatch) -> None:
    """The regression this whole fix exists for.

    With a 200-candidate window the target stock's chunks rank outside it, so the
    metadata filter returns nothing; scoring the whole corpus finds them.
    """
    monkeypatch.setattr(Config, "BM25_WINDOW", 200)
    windowed, diag_windowed = index.search_with_diagnostics(
        "回购", top_k=10, stock_code="000002"
    )

    monkeypatch.setattr(Config, "BM25_WINDOW", 0)
    full, diag_full = index.search_with_diagnostics(
        "回购", top_k=10, stock_code="000002"
    )

    assert windowed == []                      # silently missed before the fix
    assert len(full) == CORPUS_TARGET          # all targets recovered
    assert diag_windowed["truncated"] is True
    assert diag_windowed["scanned"] == 200
    assert diag_full["truncated"] is False
    assert diag_full["unlimited"] is True
    assert diag_full["window"] == diag_full["corpus"] == CORPUS_STRONG + CORPUS_TARGET
    assert diag_full["after_filter"] == CORPUS_TARGET


def test_default_window_is_unlimited(index, monkeypatch) -> None:
    monkeypatch.setattr(Config, "BM25_WINDOW", 0)

    hits, diag = index.search_with_diagnostics("回购", top_k=5, stock_code="000002")

    assert len(hits) == 5
    assert diag["unlimited"] is True and diag["truncated"] is False


def test_unfiltered_results_are_identical_with_and_without_window(index, monkeypatch) -> None:
    """Guard: without filters the window never changed the answer, so the fix must
    not change unfiltered numbers either (used as the Step 2 acceptance check)."""
    monkeypatch.setattr(Config, "BM25_WINDOW", 20)
    limited = index.search("回购", top_k=10)

    monkeypatch.setattr(Config, "BM25_WINDOW", 0)
    unlimited = index.search("回购", top_k=10)

    assert [hit["chunk_id"] for hit in limited] == [hit["chunk_id"] for hit in unlimited]


def test_truncation_is_logged_as_a_warning(index, monkeypatch, caplog) -> None:
    monkeypatch.setattr(Config, "BM25_WINDOW", 50)

    with caplog.at_level(logging.INFO, logger="app.rag.bm25"):
        index.search_with_diagnostics("回购", top_k=5)

    messages = [record.getMessage() for record in caplog.records]
    assert any("window=50 (TRUNCATED)" in message for message in messages)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and "truncation is active" in warnings[0].getMessage()


def test_full_scan_is_logged_with_its_counters(index, monkeypatch, caplog) -> None:
    monkeypatch.setattr(Config, "BM25_WINDOW", 0)

    with caplog.at_level(logging.INFO, logger="app.rag.bm25"):
        index.search_with_diagnostics("回购", top_k=3, stock_code="000002")

    messages = [r.getMessage() for r in caplog.records]
    assert any("window=ALL" in message for message in messages)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_search_still_returns_a_plain_list(index, monkeypatch) -> None:
    """Backward compatibility: the public `search()` contract is unchanged."""
    monkeypatch.setattr(Config, "BM25_WINDOW", 0)

    hits = index.search("回购", top_k=3)

    assert isinstance(hits, list) and len(hits) == 3
    assert all("bm25_score" in hit for hit in hits)


def test_diagnostics_reach_the_hybrid_retrieval_result(db_env, monkeypatch, tmp_path) -> None:
    """The counters must be visible in the retrieval result (for traces/eval)."""
    from app.rag.retrieval import hybrid_search

    monkeypatch.setattr(Config, "BM25_WINDOW", 0)
    bm25_index = BM25Index.build(_rows())

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

    outcome = hybrid_search(
        "回购", top_k=5, stock_code="000002",
        embedder=FakeEmbedder(), store=FakeStore(),
        bm25_index=bm25_index, reranker=FakeReranker(),
    )

    assert outcome["bm25_window"] == outcome["bm25_corpus"]
    assert outcome["bm25_truncated"] is False
    assert outcome["bm25_scanned"] > 0
    assert outcome["bm25_after_filter"] == CORPUS_TARGET
