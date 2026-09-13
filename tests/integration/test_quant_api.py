import pytest

from tests.conftest import STOCK_A, STOCK_B


def test_health(app_client):
    resp = app_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert body["db_tables"]["stock_basic"] >= 2


def test_factors_endpoint(app_client):
    resp = app_client.get(f"/api/stocks/{STOCK_A}/factors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == STOCK_A
    for key in ["ret_5d", "ret_20d", "ma20_bias", "volatility_20d", "drawdown_20d"]:
        assert key in body
    assert body["as_of"] <= "2023-12-01"


def test_prediction_endpoint(app_client):
    resp = app_client.get(f"/api/stocks/{STOCK_A}/prediction")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == STOCK_A
    assert "pred_return_5d" in body
    assert "score" in body
    assert body["is_historical_recomputation"] is False


def test_ranking_endpoint_sorted_and_limited(app_client):
    resp = app_client.get("/api/stocks/ranking?top_n=3")
    assert resp.status_code == 200
    items = resp.json()
    assert 1 <= len(items) <= 3
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)


def test_ranking_board_filter(app_client):
    resp = app_client.get("/api/stocks/ranking?top_n=10&board=growth")
    assert resp.status_code == 200
    for item in resp.json():
        assert item["board"] == "growth"


def test_invalid_stock_code(app_client):
    resp = app_client.get("/api/stocks/abc/factors")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "InvalidStockCode"


def test_unknown_stock_not_found(app_client):
    resp = app_client.get("/api/stocks/600999/factors")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "StockNotFound"


def test_as_of_before_data_returns_nodata(app_client):
    resp = app_client.get(f"/api/stocks/{STOCK_A}/factors?as_of=2022-12-30")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NoData"


def test_ranking_before_data_returns_nodata(app_client):
    resp = app_client.get("/api/stocks/ranking?as_of=2022-12-30")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NoData"


def test_empty_db_returns_nodata_never_fake(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import Config
    from app.db import init_db
    from app.main import app

    db_path = tmp_path / "empty.db"
    monkeypatch.setattr(Config, "DB_PATH", db_path)
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(Config, "LOG_DIR", tmp_path / "logs")
    init_db()

    client = TestClient(app)
    resp = client.get(f"/api/stocks/{STOCK_A}/factors")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] in {"NoData", "StockNotFound"}

    resp = client.get("/api/stocks/ranking")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NoData"

def test_rag_search_endpoint_returns_evidence(app_client, monkeypatch):
    """POST /api/rag/search exposes the tool contract (components faked)."""
    from app.tools import rag_tools

    class FakeEmbedder:
        version = "test-v1"
        dim = 4

        def encode(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def encode_one(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    class FakeStore:
        dim = 4

        def search(self, *args, **kwargs):
            return [
                {
                    "chunk_id": "news-eastmoney-000725-abc:c000",
                    "document_id": "news-eastmoney-000725-abc",
                    "point_id": 1,
                    "stock_code": "000725",
                    "stock_name": "京东方A",
                    "doc_type": "news",
                    "title": "京东方A回购进展",
                    "source": "eastmoney",
                    "publish_time": "2026-08-20 10:00:00",
                    "url": "http://example.com/a",
                    "content": "公司回购进展顺利。",
                    "score": 0.7,
                }
            ]

    class FakeBM25:
        def search(self, *args, **kwargs):
            return []

    class FakeReranker:
        def rerank(self, query, candidates, top_k=None):
            return list(candidates)[:top_k], ""

    monkeypatch.setattr(
        rag_tools,
        "get_components",
        lambda **kwargs: {
            "embedder": FakeEmbedder(),
            "store": FakeStore(),
            "bm25": FakeBM25(),
            "reranker": FakeReranker(),
        },
    )

    response = app_client.post("/api/rag/search", json={"query": "回购进展", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["has_evidence"] is True
    evidence = body["results"][0]
    assert evidence["url"] and evidence["publish_time"] and evidence["source"]
    assert evidence["dense_score"] == 0.7


def test_rag_search_endpoint_validates_arguments(app_client):
    """Invalid arguments must come back as structured 400 errors, not 500s."""
    response = app_client.post("/api/rag/search", json={"query": "x"})
    assert response.status_code == 422  # pydantic min_length

    response = app_client.post(
        "/api/rag/search", json={"query": "回购", "doc_type": "video"}
    )
    assert response.status_code == 422
