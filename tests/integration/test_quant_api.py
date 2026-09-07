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
    assert resp.json()["detail"]["code"] == "NoData"

    resp = client.get("/api/stocks/ranking")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NoData"
