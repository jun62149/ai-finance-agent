from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query

from app.schemas.prediction import StockPrediction, StockRankingItem
from app.schemas.stock import StockFactors
from app.services import prediction_service, stock_service


router = APIRouter(prefix="/api")


def _as_of_str(as_of: date | None) -> str | None:
    return as_of.isoformat() if as_of else None


@router.get("/health")
def health() -> dict:
    from app.config import Config
    from app.db import database_exists, table_count

    return {
        "status": "ok",
        "service": "ai-finance-agent-quant",
        "model_version": Config.MODEL_VERSION,
        "db_exists": database_exists(),
        "db_tables": {
            "stock_basic": table_count("stock_basic"),
            "stock_daily": table_count("stock_daily"),
            "factor_values": table_count("factor_values"),
            "prediction_result": table_count("prediction_result"),
            "backtest_result": table_count("backtest_result"),
        },
    }


@router.get("/stocks/{code}/factors", response_model=StockFactors)
def get_stock_factors(code: str, as_of: date | None = Query(default=None)) -> StockFactors:
    row = stock_service.get_factor_row(code, _as_of_str(as_of))
    return StockFactors(**stock_service.factor_row_to_dict(row.iloc[0]))


@router.get("/stocks/{code}/prediction", response_model=StockPrediction)
def get_stock_prediction(
    code: str, as_of: date | None = Query(default=None)
) -> StockPrediction:
    return prediction_service.predict_stock(code, _as_of_str(as_of))


@router.get("/stocks/ranking", response_model=list[StockRankingItem])
def get_stock_ranking(
    top_n: int = Query(default=10, ge=1, le=200),
    as_of: date | None = Query(default=None),
    board: Literal["main", "growth", "star", "bj"] | None = Query(default=None),
) -> list[StockRankingItem]:
    return prediction_service.rank_stocks(top_n, _as_of_str(as_of), board)
