from __future__ import annotations

from pydantic import BaseModel, Field


class StockFactors(BaseModel):
    code: str
    name: str
    as_of: str = Field(description="Actual factor trade date (<= requested as_of)")
    board: str | None = None

    # trend / momentum
    ret_5d: float | None = None
    ret_10d: float | None = None
    ret_20d: float | None = None
    ma5_bias: float | None = None
    ma20_bias: float | None = None

    # volatility
    volatility_10d: float | None = None
    volatility_20d: float | None = None

    # volume / liquidity
    volume_ratio_5d: float | None = None
    amount_mean_20d: float | None = None
    turnover_mean_20d: float | None = None

    # drawdown
    drawdown_20d: float | None = None

    close: float | None = None
    pct_change: float | None = None
