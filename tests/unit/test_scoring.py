import numpy as np
import pandas as pd

from app.quant.scoring import calculate_score, risk_score_from_factors


def test_calculate_score_formula():
    pred = pd.Series([0.01])
    prob = pd.Series([0.5])
    risk = pd.Series([0.1])
    score = calculate_score(pred, prob, risk)
    expected = 0.01 * 100 * 0.75 + 0.5 * 15 - 0.1 * 10
    np.testing.assert_allclose(score, [expected])


def test_risk_score_from_factors_uses_abs_drawdown():
    frame = pd.DataFrame({"volatility_20d": [0.02], "drawdown_20d": [-0.01]})
    result = risk_score_from_factors(frame)
    np.testing.assert_allclose(result, [0.015])


def test_score_orders_higher_pred_return_first():
    pred = pd.Series([0.001, 0.01])
    prob = pd.Series([0.3, 0.3])
    risk = pd.Series([0.1, 0.1])
    score = calculate_score(pred, prob, risk)
    assert score.iloc[1] > score.iloc[0]
