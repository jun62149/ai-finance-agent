"""Unit tests for Phase 3: BM25 index, RRF fusion and hybrid orchestration."""

from __future__ import annotations

import pytest

from app.rag.bm25 import BM25Index, tokenize
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.retrieval import hybrid_search

ROWS = [
    {
        "id": 1,
        "chunk_id": "c1",
        "document_id": "d1",
        "stock_code": "000725",
        "stock_name": "京东方A",
        "doc_type": "announcement",
        "title": "回购股份进展公告",
        "source": "cninfo",
        "publish_time": "2026-08-20",
        "url": "http://x/1",
        "chunk_index": 0,
        "content": "公司通过集中竞价方式回购股份，回购进展顺利，累计回购金额较大。",
    },
    {
        "id": 2,
        "chunk_id": "c2",
        "document_id": "d2",
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "doc_type": "announcement",
        "title": "2026年半年度报告",
        "source": "cninfo",
        "publish_time": "2026-08-15",
        "url": "http://x/2",
        "chunk_index": 0,
        "content": "报告期内公司实现营业收入同比增长，净利润保持稳定，毛利率略有下降。",
    },
    {
        "id": 3,
        "chunk_id": "c3",
        "document_id": "d3",
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "doc_type": "news",
        "title": "机构调研纪要",
        "source": "eastmoney",
        "publish_time": "2026-08-16",
        "url": "http://x/3",
        "chunk_index": 0,
        "content": "调研中机构关注分红派息政策与渠道库存情况。",
    },
]


# ---------------------------------------------------------------- tokenizer --
def test_tokenize_segments_chinese_and_keeps_codes() -> None:
    tokens = tokenize("600519 贵州茅台发布2026年半年度报告，营业收入同比增长1.3%")

    assert "600519" in tokens
    assert "贵州茅台" in tokens or "茅台" in tokens
    assert all(token.strip() == token and token for token in tokens)
    assert not any(token in {"，", "%", "。"} for token in tokens)
    assert all(token == token.lower() for token in tokens)


# --------------------------------------------------------------------- BM25 --
@pytest.fixture()
def bm25_index() -> BM25Index:
    return BM25Index.build(ROWS)


def test_bm25_ranks_exact_term_match_first(bm25_index: BM25Index) -> None:
    hits = bm25_index.search("回购 进展", top_k=3)

    assert hits
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["bm25_score"] > 0


def test_bm25_supports_metadata_filters(bm25_index: BM25Index) -> None:
    only_news = bm25_index.search("分红 派息", top_k=3, doc_type="news")
    only_600519 = bm25_index.search("营业收入", top_k=3, stock_code="600519")

    assert {hit["doc_type"] for hit in only_news} == {"news"}
    assert {hit["stock_code"] for hit in only_600519} == {"600519"}


def test_bm25_returns_empty_for_unknown_tokens(bm25_index: BM25Index) -> None:
    assert bm25_index.search("完全不存在的词汇组合", top_k=3) == []
    assert bm25_index.search("   ", top_k=3) == []


def test_bm25_save_and_load_roundtrip(bm25_index: BM25Index, tmp_path) -> None:
    bm25_index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)

    assert len(loaded) == len(bm25_index)
    assert loaded.search("回购", top_k=1)[0]["chunk_id"] == "c1"
    assert BM25Index.exists(tmp_path)


def test_bm25_load_missing_index_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_bm25"):
        BM25Index.load(tmp_path / "nope")


# ---------------------------------------------------------------------- RRF --
def _hit(chunk_id: str, **scores) -> dict:
    return {"chunk_id": chunk_id, **scores}


def test_rrf_rewards_agreement_between_retrievers() -> None:
    dense = [_hit("a", dense_score=0.9), _hit("b", dense_score=0.8), _hit("c", dense_score=0.7)]
    bm25 = [_hit("b", bm25_score=12.0), _hit("a", bm25_score=9.0), _hit("d", bm25_score=5.0)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60)

    # a: 1/61 + 1/62 ; b: 1/62 + 1/61 — tie broken by insertion, both above c/d
    assert {item["chunk_id"] for item in fused[:2]} == {"a", "b"}
    top = fused[0]
    assert top["dense_score"] is not None and top["bm25_score"] is not None
    assert top["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_keeps_single_retriever_hits() -> None:
    fused = reciprocal_rank_fusion(
        [[_hit("a", dense_score=0.9)], [_hit("z", bm25_score=1.0)]], k=60
    )

    ids = {item["chunk_id"] for item in fused}
    assert ids == {"a", "z"}
    lexical = next(item for item in fused if item["chunk_id"] == "z")
    assert lexical["bm25_score"] == 1.0
    assert "dense_score" not in lexical or lexical["dense_score"] is None


def test_rrf_top_k_truncates() -> None:
    dense = [_hit(str(i), dense_score=1.0) for i in range(5)]
    assert len(reciprocal_rank_fusion([dense], k=60, top_k=2)) == 2


# --------------------------------------------------------------- orchestration --
class FakeEmbedder:
    version = "test-v1"
    dim = 4

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_one(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FakeStore:
    dim = 4

    def search(self, vector, top_k=5, stock_code=None, doc_type=None):
        return [
            {"chunk_id": "c2", "point_id": 2, "score": 0.82, "stock_code": "600519", "doc_type": "announcement"},
            {"chunk_id": "c1", "point_id": 1, "score": 0.71, "stock_code": "000725", "doc_type": "announcement"},
        ]


class FakeBM25:
    def search(self, query, top_k=20, stock_code=None, doc_type=None):
        return [
            {"chunk_id": "c1", "point_id": 1, "bm25_score": 11.5, "stock_code": "000725", "doc_type": "announcement"},
            {"chunk_id": "c3", "point_id": 3, "bm25_score": 4.2, "stock_code": "600519", "doc_type": "news"},
        ]


class FakeReranker:
    """Deterministic scores: c3 > c1 > c2 (overrides the fusion order)."""

    SCORES = {"c3": 1.0, "c1": 0.5, "c2": 0.4}

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def rerank(self, query, candidates, top_k=None):
        if self.fail:
            return list(candidates)[:top_k], "RuntimeError: model missing"
        ranked = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_score"] = self.SCORES.get(item["chunk_id"], 0.0)
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return (ranked[:top_k] if top_k else ranked), ""


def test_hybrid_search_fuses_and_reranks(db_env) -> None:
    outcome = hybrid_search(
        "回购 进展",
        top_k=2,
        embedder=FakeEmbedder(),
        store=FakeStore(),
        bm25_index=FakeBM25(),
        reranker=FakeReranker(),
    )

    assert outcome["dense_count"] == 2
    assert outcome["bm25_count"] == 2
    assert outcome["fused_count"] == 3  # c1, c2, c3 — c1 shared by both
    assert outcome["reranked"] is True
    assert outcome["results"][0]["chunk_id"] == "c3"  # reranker score decides
    assert outcome["results"][0]["rerank_score"] == 1.0
    shared = next(item for item in outcome["results"] if item["chunk_id"] == "c1")
    assert shared["dense_score"] is not None and shared["bm25_score"] is not None
    assert "dense_ms" in outcome["timings"]


def test_hybrid_search_without_rerank_keeps_fusion_order(db_env) -> None:
    outcome = hybrid_search(
        "回购 进展",
        top_k=3,
        use_rerank=False,
        embedder=FakeEmbedder(),
        store=FakeStore(),
        bm25_index=FakeBM25(),
        reranker=FakeReranker(),
    )

    assert outcome["reranked"] is False
    assert [item["chunk_id"] for item in outcome["results"]] == ["c1", "c2", "c3"]


def test_hybrid_search_degrades_when_reranker_fails(db_env) -> None:
    outcome = hybrid_search(
        "回购 进展",
        top_k=2,
        embedder=FakeEmbedder(),
        store=FakeStore(),
        bm25_index=FakeBM25(),
        reranker=FakeReranker(fail=True),
    )

    assert outcome["reranked"] is False
    assert "model missing" in outcome["rerank_error"]
    assert len(outcome["results"]) == 2  # still returns fused results
