import pandas as pd
import pytest

from app.quant.features import FEATURE_COLUMNS, build_factor_frame
from tests.conftest import STOCK_A, build_daily_frame


def test_build_factor_frame_columns_and_clean_values():
    daily = build_daily_frame(days=80)
    frame = build_factor_frame(daily)
    assert not frame.empty
    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert frame[FEATURE_COLUMNS].isna().sum().sum() >= 0


def test_ret_5d_matches_manual_calculation():
    daily = build_daily_frame(days=60)
    frame = build_factor_frame(daily)
    stock = frame[frame["stock_code"] == STOCK_A].sort_values("trade_date").reset_index(drop=True)
    manual = daily[daily["stock_code"] == STOCK_A].sort_values("trade_date")["close"].pct_change(5)
    manual.index = range(len(manual))
    expected = manual.iloc[5:]
    actual = stock["ret_5d"].dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(actual.astype(float), expected.astype(float), check_names=False)


def test_drawdown_never_positive():
    daily = build_daily_frame(days=80)
    frame = build_factor_frame(daily)
    assert (frame["drawdown_20d"].dropna() <= 1e-9).all()


def test_volatility_non_negative_and_finite():
    daily = build_daily_frame(days=80)
    frame = build_factor_frame(daily)
    for col in ["volatility_10d", "volatility_20d"]:
        values = frame[col].dropna()
        assert (values >= 0).all()
        assert values.abs().sum() < float("inf")
