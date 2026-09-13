"""Unit tests for the retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from app.rag.eval_metrics import aggregate, median_latency, score_run


def test_perfect_ranking_scores_one() -> None:
    scores = score_run(["d1", "d2"], {"d1", "d2"}, k_values=(2,))

    assert scores["recall@2"] == 1.0
    assert scores["precision@2"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["ndcg@2"] == 1.0


def test_ranking_is_collapsed_to_distinct_documents() -> None:
    # three chunks from d1 and one from d2 collapse to [d1, d2]
    scores = score_run(["d1", "d1", "d1", "d2"], {"d1", "d2"}, k_values=(4,))

    assert scores["recall@4"] == 1.0
    assert scores["precision@4"] == 0.5  # 2 distinct docs / k=4
    assert scores["ndcg@4"] == pytest.approx(1.0)


def test_precision_penalises_irrelevant_chunks() -> None:
    scores = score_run(["d1", "x", "x", "x"], {"d1"}, k_values=(4,))

    assert scores["recall@4"] == 1.0
    assert scores["precision@4"] == 0.25
    assert scores["mrr"] == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    # duplicates collapse, so the first relevant document sits at rank 2
    scores = score_run(["x", "x", "d1"], {"d1"})

    assert scores["mrr"] == pytest.approx(0.5)
    assert scores["recall@10"] == 1.0


def test_missing_relevant_document_gives_zero() -> None:
    scores = score_run(["x", "y"], {"d1"}, k_values=(5,))

    assert scores["recall@5"] == 0.0
    assert scores["precision@5"] == 0.0
    assert scores["mrr"] == 0.0
    assert scores["ndcg@5"] == 0.0


def test_ndcg_rewards_early_relevant_hits() -> None:
    early = score_run(["d1", "x"], {"d1"}, k_values=(2,))
    late = score_run(["x", "d1"], {"d1"}, k_values=(2,))

    assert early["ndcg@2"] > late["ndcg@2"]
    assert late["ndcg@2"] == pytest.approx(1 / 1.5849625007211563)  # 1/log2(3), IDCG=1


def test_ndcg_normalises_by_number_of_relevant_documents() -> None:
    # only one of two relevant documents retrieved, but at rank 1
    scores = score_run(["d1", "x"], {"d1", "d2"}, k_values=(2,))

    assert scores["recall@2"] == 0.5
    assert scores["ndcg@2"] == pytest.approx(1.0 / (1 + 1 / 1.5849625007211563))


def test_empty_relevant_set_returns_no_metrics() -> None:
    assert score_run(["d1"], set()) == {}


def test_recall_at_k_respects_k() -> None:
    # k counts distinct ranked documents, so the relevant one must fall outside
    scores = score_run(["x1", "x2", "d1"], {"d1"}, k_values=(2, 3))

    assert scores["recall@2"] == 0.0
    assert scores["recall@3"] == 1.0


def test_duplicate_chunks_cannot_inflate_ndcg() -> None:
    # a single relevant document repeated across the whole top-k must not push
    # NDCG above 1 (regression: gains used to be counted per chunk)
    scores = score_run(["d1"] * 10, {"d1"}, k_values=(5, 10))

    assert scores["ndcg@5"] == pytest.approx(1.0)
    assert scores["ndcg@10"] == pytest.approx(1.0)
    assert scores["recall@10"] == 1.0


def test_aggregate_means_and_groups() -> None:
    rows = [
        {"id": "1", "category": "code", "recall@5": 1.0, "mrr": 1.0, "latency_ms": 10.0},
        {"id": "2", "category": "code", "recall@5": 0.0, "mrr": 0.5, "latency_ms": 30.0},
        {"id": "3", "category": "semantic", "recall@5": 0.5, "mrr": 0.25, "latency_ms": 20.0},
    ]

    overall = aggregate(rows, metric_keys=["recall@5", "mrr"])
    grouped = aggregate(rows, metric_keys=["recall@5", "mrr"], group_key="category")

    assert overall["recall@5"] == pytest.approx(0.5)
    assert overall["queries"] == 3
    assert grouped["code"]["recall@5"] == pytest.approx(0.5)
    assert grouped["semantic"]["mrr"] == pytest.approx(0.25)
    assert grouped["code"]["queries"] == 2
    assert grouped["code"]["latency_ms"] == pytest.approx(20.0)  # median of 10/30


def test_median_latency() -> None:
    assert median_latency([30.0, 10.0, 20.0]) == 20.0
    assert median_latency([]) == 0.0
