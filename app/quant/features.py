"""Factor engineering, migrated unchanged from legacy B factor_service.

Kept identical on purpose: same feature columns, same labels, same pool filter,
so the Quant Engine reproduces the graduation-project numbers on the same data.
"""

from __future__ import annotations

import pandas as pd

from app.config import Config
from app.db import read_sql, upsert_dataframe
from app.log import write_log
from app.quant.pool import apply_model_pool_filter


FEATURE_COLUMNS = [
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ma5_bias",
    "ma20_bias",
    "volatility_10d",
    "volatility_20d",
    "volume_ratio_5d",
    "amount_mean_20d",
    "turnover_mean_20d",
    "drawdown_20d",
]


def load_daily_with_basic() -> pd.DataFrame:
    sql = """
    SELECT d.*, b.stock_name, b.market, b.list_date, b.is_active
    FROM stock_daily d
    LEFT JOIN stock_basic b ON d.stock_code = b.stock_code
    ORDER BY d.stock_code, d.trade_date
    """
    df = read_sql(sql)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["stock_code", "trade_date", "close"])


def build_factor_frame(df: pd.DataFrame | None = None, horizon: int | None = None) -> pd.DataFrame:
    horizon = horizon or Config.PREDICTION_HORIZON_DAYS
    data = load_daily_with_basic() if df is None else df.copy()
    if data.empty:
        return data

    data = data.sort_values(["stock_code", "trade_date"]).copy()
    grouped = data.groupby("stock_code", group_keys=False)

    data["ret_5d"] = grouped["close"].pct_change(5)
    data["ret_10d"] = grouped["close"].pct_change(10)
    data["ret_20d"] = grouped["close"].pct_change(20)

    data["ma5"] = grouped["close"].transform(lambda s: s.rolling(5).mean())
    data["ma10"] = grouped["close"].transform(lambda s: s.rolling(10).mean())
    data["ma20"] = grouped["close"].transform(lambda s: s.rolling(20).mean())
    data["ma5_bias"] = data["close"] / data["ma5"] - 1
    data["ma20_bias"] = data["close"] / data["ma20"] - 1

    daily_ret = grouped["close"].pct_change()
    data["volatility_10d"] = daily_ret.groupby(data["stock_code"]).transform(lambda s: s.rolling(10).std())
    data["volatility_20d"] = daily_ret.groupby(data["stock_code"]).transform(lambda s: s.rolling(20).std())

    data["volume_mean_5d"] = grouped["volume"].transform(lambda s: s.rolling(5).mean())
    data["volume_mean_20d"] = grouped["volume"].transform(lambda s: s.rolling(20).mean())
    data["volume_ratio_5d"] = data["volume_mean_5d"] / data["volume_mean_20d"]
    data["amount_mean_20d"] = grouped["amount"].transform(lambda s: s.rolling(20).mean())
    data["turnover_mean_20d"] = grouped["turnover"].transform(lambda s: s.rolling(20).mean())

    roll_high = grouped["high"].transform(lambda s: s.rolling(20).max())
    data["drawdown_20d"] = data["close"] / roll_high - 1

    data["future_close"] = grouped["close"].shift(-horizon)
    data["future_return_5d"] = data["future_close"] / data["close"] - 1
    daily_mean = data.groupby("trade_date")["future_return_5d"].transform("mean")
    data["future_excess_return_5d"] = data["future_return_5d"] - daily_mean
    rank = data.groupby("trade_date")["future_return_5d"].rank(pct=True)
    data["strong_label"] = (rank >= 0.7).astype(int)

    keep_cols = [
        "stock_code",
        "stock_name",
        "market",
        "list_date",
        "is_active",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_change",
        "turnover",
        *FEATURE_COLUMNS,
        "ma5",
        "ma10",
        "ma20",
        "future_return_5d",
        "future_excess_return_5d",
        "strong_label",
    ]
    return data[keep_cols].replace([float("inf"), float("-inf")], pd.NA)


def rebuild_factors() -> pd.DataFrame:
    factors = build_factor_frame()
    if factors.empty:
        write_log("factor rebuild skipped: no daily data", level="WARN")
        return factors

    store_cols = [
        "stock_code",
        "trade_date",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ma5",
        "ma10",
        "ma20",
        "ma5_bias",
        "ma20_bias",
        "volatility_10d",
        "volatility_20d",
        "volume_ratio_5d",
        "amount_mean_20d",
        "turnover_mean_20d",
        "drawdown_20d",
        "future_return_5d",
        "future_excess_return_5d",
        "strong_label",
    ]
    clean = factors.dropna(subset=FEATURE_COLUMNS)[store_cols].copy()
    clean["trade_date"] = pd.to_datetime(clean["trade_date"]).dt.strftime("%Y-%m-%d")
    count = upsert_dataframe(clean, "factor_values", ["stock_code", "trade_date"])
    write_log(f"factors rebuilt: {count} rows")
    return factors


def load_factor_dataset(require_label: bool = True) -> pd.DataFrame:
    factors = build_factor_frame()
    if factors.empty:
        return factors
    subset = FEATURE_COLUMNS + (["future_excess_return_5d", "strong_label"] if require_label else [])
    clean = factors.dropna(subset=subset).copy()
    clean = apply_model_pool_filter(clean)
    counts = clean.groupby("stock_code")["trade_date"].transform("count")
    clean = clean[counts >= Config.MIN_TRAINING_OBSERVATIONS_PER_STOCK]
    return clean.reset_index(drop=True)


def latest_factor_rows() -> pd.DataFrame:
    factors = build_factor_frame()
    if factors.empty:
        return factors
    factors = factors.dropna(subset=FEATURE_COLUMNS).copy()
    factors = apply_model_pool_filter(factors)
    return factors.sort_values(["stock_code", "trade_date"]).groupby("stock_code").tail(1).reset_index(drop=True)
