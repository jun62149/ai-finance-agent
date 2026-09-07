"""Run the latest prediction for the whole eligible pool (legacy parity path)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.quant.predictor import predict_latest


def main() -> None:
    result = predict_latest()
    print(f"predictions: {len(result)} rows")
    top = result.head(10)[["trade_date", "stock_code", "stock_name", "score", "pred_return_5d"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
