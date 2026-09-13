"""Unit tests for the retrieval-budget semantics (review batch 1, Issue 2).

The bug being locked down: BM25/Dense used to retrieve fewer chunks than Hybrid,
which - because metrics are computed per document after chunk->document dedup -
gave Hybrid a larger effective budget.
"""

from __future__ import annotations

import pytest

from scripts.evaluate_retrieval import resolve_budgets, run_config

SPEC = {
    "id": "T01",
    "category": "code",
    "query": "回购 进展",
    "stock_code": "000725",
    "doc_type": "announcement",
    "relevant_document_ids": ["d1"],
    "relevant_count": 1,
}


# ------------------------------------------------------------------ budgets --
def test_budgets_default_fusion_to_candidate() -> None:
    budgets = resolve_budgets(20, None, 20, (5, 10))

    assert budgets == {"candidate_k": 20, "fusion_k": 20, "final_k": 20}


def test_budgets_accept_explicit_fusion_larger_than_final() -> None:
    budgets = resolve_budgets(20, 15, 10, (5, 10))

    assert budgets["fusion_k"] == 15 and budgets["final_k"] == 10


@pytest.mark.parametrize(
    "candidate_k, fusion_k, final_k, k_values, message",
    [
        (0, None, 10, (5,), "candidate-k"),
        (10, None, 0, (5,), "final-k"),
        (10, 5, 10, (5,), "fusion-k"),          # fusion below final
        (5, None, 10, (5,), "candidate-k"),     # candidate below final
        (20, None, 10, (5, 20), "k-values"),    # metric looks past the budget
        (20, None, 10, (), "k-values"),
    ],
)
def test_budgets_reject_inconsistent_settings(
    candidate_k, fusion_k, final_k, k_values, message
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_budgets(candidate_k, fusion_k, final_k, k_values)


# --------------------------------------------------------------- run_config --
class FakeEmbedder:
    dim = 4
    version = "test"

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_one(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FakeStore:
    """Records the requested depth; returns chunks from one shared document."""

    dim = 4

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, vector, top_k=5, stock_code=None, doc_type=None,
               date_from=None, date_to=None):
        self.calls.append({"top_k": top_k, "stock_code": stock_code, "doc_type": doc_type})
        return [
            {
                "chunk_id": f"c{i}",
                "document_id": "docA",
                "point_id": i,
                "score": 0.9 - i * 0.01,
                "content": "正文",
                "stock_code": stock_code,
                "doc_type": doc_type,
            }
            for i in range(1, top_k + 1)
        ]


class FakeBM25:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def search(self, query, top_k=20, stock_code=None, doc_type=None,
               date_from=None, date_to=None):
        self.calls.append(top_k)
        return [
            {
                "chunk_id": f"b{i}",
                "document_id": "docB",
                "point_id": 100 + i,
                "bm25_score": 5.0 - i * 0.1,
                "stock_code": stock_code,
                "doc_type": doc_type,
            }
            for i in range(1, top_k + 1)
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def rerank(self, query, candidates, top_k=None):
        self.calls.append(len(candidates))
        ranked = [{**c, "rerank_score": 1.0} for c in candidates]
        return (ranked[:top_k] if top_k else ranked), ""


def _call(name: str, budgets: dict, db_env):
    store, index, reranker = FakeStore(), FakeBM25(), FakeReranker()
    doc_ids, unique_docs, _ = run_config(
        name, SPEC, embedder=FakeEmbedder(), store=store, index=index,
        reranker=reranker, budgets=budgets,
    )
    return doc_ids, unique_docs, store, index, reranker


def test_single_source_configs_use_the_shared_final_budget(db_env) -> None:
    budgets = resolve_budgets(20, None, 20, (5, 10))

    _, _, store_bm25, index, _ = _call("bm25", budgets, db_env)
    assert index.calls == [20]          # exactly the shared budget

    _, _, store_dense, _, _ = _call("dense", budgets, db_env)
    assert store_dense.calls[-1]["top_k"] == 20


def test_hybrid_uses_candidate_depth_then_shared_final_budget(db_env) -> None:
    budgets = resolve_budgets(candidate_k=20, fusion_k=15, final_k=10, k_values=(5, 10))

    doc_ids, unique_docs, store, index, reranker = _call("hybrid", budgets, db_env)

    assert store.calls[-1]["top_k"] == 20   # per-retriever candidate depth
    assert index.calls == [20]
    assert len(doc_ids) <= 10               # shared final budget
    assert unique_docs <= 10


def test_hybrid_rerank_sees_fusion_depth_and_returns_final_budget(db_env) -> None:
    budgets = resolve_budgets(candidate_k=20, fusion_k=15, final_k=10, k_values=(5, 10))

    doc_ids, _, _, _, reranker = _call("hybrid_rerank", budgets, db_env)

    assert reranker.calls[-1] <= 15         # reranker candidate set = fusion-k
    assert len(doc_ids) <= 10


def test_filters_are_forwarded_unless_ignored(db_env) -> None:
    budgets = resolve_budgets(10, None, 10, (5,))

    _, _, store, _, _ = _call("dense", budgets, db_env)
    assert store.calls[-1]["stock_code"] == "000725"

    store2 = FakeStore()
    run_config("dense", SPEC, embedder=FakeEmbedder(), store=store2, index=FakeBM25(),
               reranker=FakeReranker(), budgets=budgets, ignore_filters=True)
    assert store2.calls[-1]["stock_code"] is None
