"""Prediction & ranking service: the stable Quant API contract."""

from __future__ import annotations

import pandas as pd

from app.config import Config
from app.errors import NoData
from app.quant.pool import add_pool_flags, stock_board
from app.quant.predictor import predict_on_factor_rows
from app.quant.ranking import apply_score_sort, factor_rows_on_date, target_factor_date
from app.schemas.prediction import StockPrediction, StockRankingItem
from app.services.stock_service import (
    factor_row_to_dict,
    get_factor_row,
    get_stock_basic,
    validate_code,
)


def predict_stock(code: str, as_of: str | None = None) -> StockPrediction:
    code = validate_code(code)
    get_stock_basic(code)
    row = get_factor_row(code, as_of)
    predicted = predict_on_factor_rows(row).iloc[0]

    values = factor_row_to_dict(predicted)
    return StockPrediction(
        code=values["code"],
        name=values["name"],
        as_of=values["as_of"],
        pred_return_5d=float(predicted["pred_return_5d"]),
        strong_prob=float(predicted["strong_prob"]),
        risk_score=float(predicted["risk_score"]),
        score=float(predicted["score"]),
        model_version=Config.MODEL_VERSION,
        is_historical_recomputation=as_of is not None,
        close=values["close"],
        pct_change=values["pct_change"],
        board=stock_board(predicted["stock_code"]),
        main_factor=str(predicted.get("main_factor") or ""),
    )


def rank_stocks(top_n: int = 10, as_of: str | None = None, board: str | None = None) -> list[StockRankingItem]:
    target_date = target_factor_date(as_of)
    if not target_date:
        scope = f" as_of={as_of}" if as_of else ""
        raise NoData(f"no factor data{scope}")

    frame = factor_rows_on_date(target_date)
    if frame.empty:
        raise NoData(f"no factor data on {target_date}")

    flagged = add_pool_flags(frame)
    if board:
        pool = flagged[flagged["board"] == board]
    else:
        pool = flagged[flagged["is_recommendable_regular"]]

    if pool.empty:
        return []

    predicted = predict_on_factor_rows(pool.copy())
    ranked = apply_score_sort(predicted).head(top_n)

    items: list[StockRankingItem] = []
    for _, row in ranked.iterrows():
        items.append(
            StockRankingItem(
                code=str(row["stock_code"]).zfill(6),
                name=str(row.get("stock_name") or ""),
                rank=int(row["rank"]),
                score=float(row["score"]),
                pred_return_5d=float(row["pred_return_5d"]),
                risk_score=float(row["risk_score"]),
                strong_prob=float(row["strong_prob"]),
                board=str(row.get("board") or ""),
                main_factor=str(row.get("main_factor") or ""),
            )
        )
    return items
