from __future__ import annotations

from pydantic import BaseModel, Field


class StockPrediction(BaseModel):
    code: str
    name: str
    as_of: str = Field(description="Actual factor trade date used")
    pred_return_5d: float = Field(description="Model predicted 5-day excess return")
    strong_prob: float = Field(
        description="Auxiliary classifier probability; low-recall model, treat as weak signal only"
    )
    risk_score: float
    score: float
    model_version: str
    is_historical_recomputation: bool = Field(
        description="True when computed on a historical factor row rather than the latest date"
    )
    close: float | None = None
    pct_change: float | None = None
    board: str | None = None
    main_factor: str | None = None


class StockRankingItem(BaseModel):
    code: str
    name: str
    rank: int
    score: float
    pred_return_5d: float
    risk_score: float
    strong_prob: float
    board: str | None = None
    main_factor: str | None = None
