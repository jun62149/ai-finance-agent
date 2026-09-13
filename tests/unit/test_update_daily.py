"""Unit tests for the incremental daily-bar updater (offline; baostock is faked)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.db import read_sql, upsert_dataframe
from scripts.update_daily import (
    DAILY_COLUMNS,
    BaoStockSession,
    _clean,
    fetch_daily_akshare,
    last_trade_date,
    target_stocks,
)


class FakeResult:
    """Mimics baostock's result object (fields + next()/get_row_data())."""

    error_code = "0"
    fields = ["date", "code", "open", "high", "low", "close", "volume", "amount", "pctChg", "turn"]

    def __init__(self, rows):
        self._rows = list(rows)
        self._index = -1

    def next(self):
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self):
        return self._rows[self._index]


class FakeErrorResult:
    """A baostock result carrying an error code instead of rows."""

    fields: list = []

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        self.error_msg = "boom"

    def next(self) -> bool:
        return False


class FakeBaoStock:
    def __init__(self, rows, error_code="0"):
        self.rows = rows
        self.error_code = error_code
        self.requests: list[tuple] = []

    def login(self):
        return type("R", (), {"error_code": "0", "error_msg": ""})()

    def logout(self):
        return None

    def query_history_k_data_plus(self, code, fields, **kwargs):
        self.requests.append((code, fields, kwargs))
        if self.error_code != "0":
            return FakeErrorResult(self.error_code)
        return FakeResult(self.rows)


def test_clean_normalises_dates_and_types() -> None:
    frame = pd.DataFrame(
        [
            {
                "stock_code": "000725", "trade_date": "2026/06/25",
                "open": "7.5", "high": "7.8", "low": "7.4", "close": "7.52",
                "volume": "36963548", "amount": "27560399329",
                "pct_change": "6.3649", "turnover": "10.4514", "source": "baostock",
            },
            {  # unusable row: no close
                "stock_code": "000725", "trade_date": "2026-06-26", "close": None,
                "source": "baostock",
            },
        ]
    )

    cleaned = _clean(frame)

    assert list(cleaned.columns) == DAILY_COLUMNS
    assert len(cleaned) == 1  # the row without a close is dropped
    assert cleaned.iloc[0]["trade_date"] == "2026-06-25"
    assert cleaned.iloc[0]["close"] == pytest.approx(7.52)
    assert isinstance(cleaned.iloc[0]["volume"], float)


def test_baostock_fetch_uses_market_prefix_and_unadjusted_flag() -> None:
    session = BaoStockSession()
    session._module = FakeBaoStock(
        [["2026-06-25", "sz.000725", "7.50", "7.80", "7.40", "7.52", "3696354800", "27560399329", "6.3649", "10.4514"]]
    )
    session.logged_in = True

    frame = session.fetch("000725", "2026-06-25", "2026-06-25")

    code, fields, kwargs = session._module.requests[0]
    assert code == "sz.000725"  # market prefix required by baostock
    assert kwargs["adjustflag"] == "3"  # 不复权 — must stay identical to the legacy data
    assert "pctChg" in fields and "turn" in fields
    assert frame.iloc[0]["close"] == pytest.approx(7.52)
    assert frame.iloc[0]["pct_change"] == pytest.approx(6.3649)
    # baostock returns 股, the stored convention is 手 (divide by 100)
    assert frame.iloc[0]["volume"] == pytest.approx(36963548.0)


def test_baostock_fetch_raises_on_query_error() -> None:
    session = BaoStockSession()
    session._module = FakeBaoStock([], error_code="10002007")
    session.logged_in = True

    with pytest.raises(RuntimeError, match="baostock query failed"):
        session.fetch("000725", "2026-06-25", "2026-06-25")


def test_target_stocks_prefers_explicit_codes() -> None:
    assert target_stocks(["000725", "600519"], limit=None) == ["000725", "600519"]
    assert target_stocks(["000725", "600519"], limit=1) == ["000725"]


def test_last_trade_date_reads_the_db(populated_db) -> None:
    frame = pd.DataFrame(
        [
            {"stock_code": "600001", "trade_date": "2026-06-24", "close": 10.0, "source": "test"},
            {"stock_code": "600001", "trade_date": "2026-06-25", "close": 10.5, "source": "test"},
        ]
    )
    upsert_dataframe(frame, "stock_daily", ["stock_code", "trade_date"])

    assert last_trade_date("600001") == "2026-06-25"
    assert last_trade_date("999999") is None


def test_repeated_upsert_of_the_same_bar_is_idempotent(populated_db) -> None:
    """A re-run must not duplicate bars - this is what makes the updater resumable."""
    frame = pd.DataFrame(
        [{"stock_code": "600001", "trade_date": "2026-06-25", "close": 10.5, "source": "test"}]
    )
    upsert_dataframe(frame, "stock_daily", ["stock_code", "trade_date"])
    upsert_dataframe(frame, "stock_daily", ["stock_code", "trade_date"])

    count = read_sql(
        "SELECT COUNT(*) AS n FROM stock_daily WHERE stock_code = '600001' AND trade_date = '2026-06-25'"
    ).iloc[0]["n"]
    assert int(count) == 1


def test_akshare_mapping_matches_the_legacy_convention(monkeypatch) -> None:
    """akshare columns are Chinese; the mapping must match the legacy pipeline."""
    import akshare as ak

    raw = pd.DataFrame(
        [{"日期": "2026-06-25", "开盘": 7.5, "最高": 7.8, "最低": 7.4, "收盘": 7.52,
          "成交量": 36963548, "成交额": 27560399329, "涨跌幅": 6.36, "换手率": 10.45}]
    )
    monkeypatch.setattr(ak, "stock_zh_a_hist", lambda **kwargs: raw)

    frame = fetch_daily_akshare("000725", "2026-06-25", "2026-06-25")

    assert frame.iloc[0]["trade_date"] == "2026-06-25"
    assert frame.iloc[0]["close"] == pytest.approx(7.52)
    assert frame.iloc[0]["turnover"] == pytest.approx(10.45)
    assert frame.iloc[0]["source"] == "akshare"
    # akshare volume is already in 手 — no conversion
    assert frame.iloc[0]["volume"] == pytest.approx(36963548)
