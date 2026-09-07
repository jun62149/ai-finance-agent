from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import Config
from app.db import init_db, upsert_dataframe
from app.quant.features import build_factor_frame


STOCK_A = "600001"
STOCK_B = "300001"


def _make_daily(stock_code: str, name: str, start: str, days: int, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=days)
    close = 10 + np.cumsum(rng.normal(0, 0.3, size=days))
    close = np.maximum(close, 1.0)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    open_ = prev_close * (1 + rng.normal(0, 0.005, size=days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, size=days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, size=days))
    volume = rng.uniform(1e7, 3e7, size=days)
    amount = volume * close  # > MIN_AMOUNT (50M)
    pct_change = close / prev_close - 1
    turnover = rng.uniform(0.5, 3.0, size=days)
    return pd.DataFrame(
        {
            "stock_code": stock_code,
            "stock_name": name,
            "trade_date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "pct_change": pct_change,
            "turnover": turnover,
            "source": "test",
        }
    )


def build_daily_frame(days: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = _make_daily(STOCK_A, "测试股份A", "2023-01-02", days, rng)
    b = _make_daily(STOCK_B, "测试股份B", "2023-01-02", days, rng)
    return pd.concat([a, b], ignore_index=True)


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Point Config at a temporary SQLite file and create empty schema."""
    db_path = tmp_path / "test_stock.db"
    monkeypatch.setattr(Config, "DB_PATH", db_path)
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(Config, "LOG_DIR", tmp_path / "logs")
    init_db()
    return db_path


@pytest.fixture()
def populated_db(db_env):
    """Insert stock_basic + stock_daily for two synthetic stocks and store factors."""
    daily = build_daily_frame()
    basic = (
        daily[["stock_code", "stock_name"]]
        .drop_duplicates("stock_code")
        .assign(market=lambda d: d["stock_code"].map(lambda c: "sh" if c.startswith("6") else "sz"))
        .assign(list_date="2019-01-01", is_active=1)
    )
    upsert_dataframe(basic, "stock_basic", ["stock_code"])
    upsert_dataframe(daily, "stock_daily", ["stock_code", "trade_date"])
    from app.quant.features import rebuild_factors

    rebuild_factors()
    return daily


@pytest.fixture()
def app_client(populated_db):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
