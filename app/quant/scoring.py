"""Scoring logic, migrated unchanged from legacy B model_service."""

from __future__ import annotations

import pandas as pd

from app.config import Config


def risk_score_from_factors(frame: pd.DataFrame) -> pd.Series:
    """Risk score from volatility_20d and |drawdown_20d| (legacy formula)."""
    return (
        frame[["volatility_20d", "drawdown_20d"]]
        .assign(drawdown_20d=frame["drawdown_20d"].abs())
        .mean(axis=1)
        .fillna(0)
    )


def calculate_score(pred_return: pd.Series, strong_prob: pd.Series, risk_score: pd.Series) -> pd.Series:
    return (
        pred_return * 100 * Config.SCORE_W_PRED_RETURN
        + strong_prob * Config.SCORE_W_STRONG_PROB
        - risk_score * Config.SCORE_W_RISK
    )
