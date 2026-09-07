import pytest

from app.errors import ModelUnavailable, NoData, StockNotFound
from app.schemas.prediction import StockPrediction
from app.services.prediction_service import predict_stock
from tests.conftest import STOCK_A, STOCK_B


def test_predict_stock_latest(populated_db):
    result = predict_stock(STOCK_A)
    assert isinstance(result, StockPrediction)
    assert result.code == STOCK_A
    assert result.is_historical_recomputation is False
    assert result.score == result.score  # finite
    assert 0.0 <= result.strong_prob <= 1.0
    assert result.pred_return_5d is not None
    assert result.model_version


def test_predict_stock_as_of_is_historical_and_bounded(populated_db):
    result = predict_stock(STOCK_A, as_of="2023-06-30")
    assert result.is_historical_recomputation is True
    assert result.as_of <= "2023-06-30"
    assert result.code == STOCK_A


def test_predict_stock_as_of_before_data_raises_nodata(populated_db):
    with pytest.raises(NoData):
        predict_stock(STOCK_A, as_of="2022-12-30")


def test_predict_stock_missing_stock_raises_not_found(populated_db):
    with pytest.raises(StockNotFound):
        predict_stock("600999")


def test_predict_stock_missing_model_raises_model_unavailable(populated_db, monkeypatch):
    from pathlib import Path

    from app.config import Config
    from app.quant import predictor

    fake = Path("C:/definitely/missing/reg.pkl")
    monkeypatch.setattr(Config, "MODEL_REGRESSOR_PATH", fake)
    monkeypatch.setattr(Config, "MODEL_CLASSIFIER_PATH", fake)
    predictor.load_models.cache_clear()
    try:
        with pytest.raises(ModelUnavailable):
            predict_stock(STOCK_A)
    finally:
        predictor.load_models.cache_clear()


def test_no_data_never_uses_demo_fallback(populated_db, monkeypatch):
    """When the DB is empty, services must raise, never fabricate rows."""
    import pandas as pd

    from app.services.stock_service import get_factor_row
    from app.db import connect

    with connect() as conn:
        for table in ["stock_basic", "stock_daily", "factor_values", "prediction_result"]:
            conn.execute(f"DELETE FROM {table}")
    with pytest.raises(NoData):
        get_factor_row(STOCK_A)
