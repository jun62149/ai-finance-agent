"""Stock & factor service: stable contract between DB/quant and API/tools."""

from __future__ import annotations

import re

import pandas as pd

from app.db import read_sql
from app.errors import InvalidStockCode, NoData, StockNotFound
from app.quant.features import FEATURE_COLUMNS
from app.quant.pool import stock_board


CODE_RE = re.compile(r"^\d{6}$")


def validate_code(code: str) -> str:
    normalized = str(code).strip()
    if not CODE_RE.match(normalized):
        raise InvalidStockCode(f"invalid stock code: {code!r} (expected 6 digits)")
    return normalized


def get_stock_basic(code: str) -> dict:
    code = validate_code(code)
    df = read_sql(
        "SELECT stock_code, stock_name, market, list_date, is_active FROM stock_basic WHERE stock_code = ?",
        (code,),
    )
    if df.empty:
        raise StockNotFound(f"stock not found in stock_basic: {code}")
    row = df.iloc[0].to_dict()
    row["stock_code"] = str(row["stock_code"]).zfill(6)
    return row


def get_factor_row(code: str, as_of: str | None = None) -> pd.DataFrame:
    """Return the latest factor row (<= as_of when given). Never falls back to latest."""
    code = validate_code(code)
    sql = """
    SELECT f.*, b.stock_name, b.list_date, b.is_active
    FROM factor_values f
    LEFT JOIN stock_basic b ON f.stock_code = b.stock_code
    WHERE f.stock_code = ?
    """
    params: list = [code]
    if as_of:
        sql += " AND f.trade_date <= ?"
        params.append(as_of)
    sql += " ORDER BY f.trade_date DESC LIMIT 1"
    df = read_sql(sql, tuple(params))
    if df.empty:
        scope = f" as_of={as_of}" if as_of else ""
        raise NoData(f"no factor data for stock {code}{scope}")
    row = df.iloc[0].to_dict()
    row["stock_code"] = str(row["stock_code"]).zfill(6)
    row["trade_date"] = pd.to_datetime(row["trade_date"]).strftime("%Y-%m-%d")
    return pd.DataFrame([row])


def factor_row_to_dict(row: pd.Series) -> dict:
    data = {
        "code": str(row["stock_code"]).zfill(6),
        "name": str(row.get("stock_name") or ""),
        "as_of": pd.to_datetime(row["trade_date"]).strftime("%Y-%m-%d"),
        "board": stock_board(row["stock_code"]),
    }
    for col in FEATURE_COLUMNS + ["close", "pct_change"]:
        data[col] = None if pd.isna(row.get(col)) else float(row[col])
    return data
