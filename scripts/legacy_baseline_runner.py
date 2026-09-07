"""Run the LEGACY (B-version) predict path against a copied database.

Used by scripts/verify_consistency.py to produce the pre-migration baseline.
This runner imports the legacy package from its own root and mutates only the
given DB copy; the original project directory is never touched.

Usage:
    python scripts/legacy_baseline_runner.py LEGACY_ROOT LEGACY_DB OUT_JSON
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def main() -> None:
    legacy_root, legacy_db, out_json = (Path(x) for x in sys.argv[1:4])

    # The legacy app package imports flask at package import time; this runner
    # only needs pure services, so provide a stub to avoid the dependency.
    flask = types.ModuleType("flask")
    flask.Flask = object
    sys.modules.setdefault("flask", flask)

    sys.path.insert(0, str(legacy_root))

    from app.config import Config

    Config.DB_PATH = legacy_db
    Config.DATA_DIR = legacy_db.parent
    Config.LOG_DIR = legacy_db.parent / "logs"

    import app.services.factor_service as legacy_factors
    import app.services.model_service as legacy_model

    # Compute the full factor frame once and reuse it for both the factor
    # snapshot and the prediction path (predict_latest -> latest_factor_rows).
    real_build = legacy_factors.build_factor_frame
    cache: dict = {}

    def cached_build(df=None, horizon=None):
        if df is not None or horizon is not None:
            return real_build(df=df, horizon=horizon)
        if "full" not in cache:
            cache["full"] = real_build()
        return cache["full"].copy()

    legacy_factors.build_factor_frame = cached_build

    prediction = legacy_model.predict_latest()
    latest_factors = legacy_factors.latest_factor_rows()

    factor_by_code = {
        str(row["stock_code"]).zfill(6): row
        for _, row in latest_factors.iterrows()
    }
    pred_by_code = {
        str(row["stock_code"]).zfill(6): row
        for _, row in prediction.iterrows()
    }

    codes = sorted(set(factor_by_code) & set(pred_by_code))
    records = []
    for code in codes:
        f = factor_by_code[code]
        p = pred_by_code[code]
        records.append(
            {
                "code": code,
                "name": str(f.get("stock_name") or ""),
                "trade_date": str(f["trade_date"])[:10],
                "factors": {col: _num(f.get(col)) for col in legacy_factors.FEATURE_COLUMNS},
                "pred_return_5d": _num(p.get("pred_return_5d")),
                "strong_prob": _num(p.get("strong_prob")),
                "risk_score": _num(p.get("risk_score")),
                "score": _num(p.get("score")),
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"count": len(records), "records": records}, ensure_ascii=False), encoding="utf-8")
    print(f"legacy baseline records: {len(records)}")


def _num(value):
    try:
        import pandas as pd

        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
