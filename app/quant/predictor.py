"""Model loading & prediction, migrated from legacy B model_service.

Strictly no auto-retrain and no demo fallback: missing model files raise
ModelUnavailable.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from app.config import Config
from app.db import upsert_dataframe
from app.errors import ModelUnavailable, NoData
from app.log import write_log
from app.quant import scoring
from app.quant.features import FEATURE_COLUMNS, latest_factor_rows
from app.quant.pool import add_pool_flags


_MAIN_FACTOR_MAPPING = {
    "ret_20d": "trend momentum",
    "ma20_bias": "moving average bias",
    "volume_ratio_5d": "volume confirmation",
    "volatility_20d": "low volatility",
    "drawdown_20d": "drawdown repair",
}


def _require_model_files() -> None:
    missing = [
        str(path)
        for path in (Config.MODEL_REGRESSOR_PATH, Config.MODEL_CLASSIFIER_PATH)
        if not path.exists()
    ]
    if missing:
        raise ModelUnavailable(f"model file(s) missing: {missing}")


@lru_cache(maxsize=1)
def load_models():
    import joblib

    _require_model_files()
    reg = joblib.load(Config.MODEL_REGRESSOR_PATH)
    clf = joblib.load(Config.MODEL_CLASSIFIER_PATH)
    return reg, clf


def predict_on_factor_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Add pred_return_5d / strong_prob / risk_score / score / main_factor.

    Model output depends only on the factor row (no future labels involved), so
    applying it to a historical row is a point-in-time recomputation.
    """
    if rows.empty:
        return rows
    reg, clf = load_models()
    x = rows[FEATURE_COLUMNS].astype(float)
    out = rows.copy()
    out["pred_return_5d"] = reg.predict(x)
    if hasattr(clf, "predict_proba"):
        out["strong_prob"] = clf.predict_proba(x)[:, 1]
    else:
        out["strong_prob"] = clf.predict(x)
    out["risk_score"] = scoring.risk_score_from_factors(out)
    out["score"] = scoring.calculate_score(
        out["pred_return_5d"], out["strong_prob"], out["risk_score"]
    )
    out["main_factor"] = _main_factor(out)
    return out


def _main_factor(df: pd.DataFrame) -> pd.Series:
    candidates = list(_MAIN_FACTOR_MAPPING.keys())
    idx = df[candidates].abs().idxmax(axis=1)
    return idx.map(_MAIN_FACTOR_MAPPING)


def predict_latest() -> pd.DataFrame:
    """Legacy-equivalent latest-prediction path (per-stock latest factor row)."""
    latest = latest_factor_rows()
    if latest.empty:
        raise NoData("no factor rows available")  # no fake data ever

    latest = latest.copy()
    latest = predict_on_factor_rows(latest)
    latest = add_pool_flags(latest)
    latest["is_recommendable"] = latest["is_recommendable_regular"].astype(int)
    latest["trade_date"] = pd.to_datetime(latest["trade_date"]).dt.strftime("%Y-%m-%d")

    output = latest[
        [
            "trade_date",
            "stock_code",
            "stock_name",
            "pred_return_5d",
            "strong_prob",
            "risk_score",
            "score",
            "close",
            "pct_change",
            "board",
            "main_factor",
            "is_recommendable",
        ]
    ].copy()
    upsert_dataframe(output, "prediction_result", ["trade_date", "stock_code"])
    write_log(f"latest prediction generated: {len(output)} rows")
    return output.sort_values("score", ascending=False).reset_index(drop=True)


def feature_importance() -> list[dict]:
    """Feature importance from the shipped model (informational only)."""
    _require_model_files()
    try:
        import json

        metrics = json.loads(Config.MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        return metrics.get("feature_importance", [])
    except Exception:
        return []


def model_metrics_snapshot() -> dict:
    try:
        import json

        return json.loads(Config.MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModelUnavailable(f"metrics file missing: {Config.MODEL_METRICS_PATH}") from None
