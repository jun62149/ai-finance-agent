"""Cross-sectional ranking built on stored factor rows."""

from __future__ import annotations

import pandas as pd

from app.db import read_sql
from app.quant import scoring
from app.quant.features import FEATURE_COLUMNS
from app.quant.pool import add_pool_flags, stock_board


def target_factor_date(as_of: str | None = None) -> str | None:
    """Latest trade date in factor_values (<= as_of when provided)."""
    sql = "SELECT MAX(trade_date) AS d FROM factor_values"
    params: tuple | None = None
    if as_of:
        sql += " WHERE trade_date <= ?"
        params = (as_of,)
    df = read_sql(sql, params)
    value = df.iloc[0]["d"] if not df.empty else None
    return str(value) if value is not None and pd.notna(value) else None


def factor_rows_on_date(trade_date: str) -> pd.DataFrame:
    sql = """
    SELECT f.*, b.stock_name, b.list_date
    FROM factor_values f
    LEFT JOIN stock_basic b ON f.stock_code = b.stock_code
    WHERE f.trade_date = ?
    """
    df = read_sql(sql, (trade_date,))
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    df = df[df["stock_code"].map(lambda code: stock_board(code) != "other")]
    return df.reset_index(drop=True)


def apply_score_sort(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked
