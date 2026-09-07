"""Initialize / migrate the SQLite schema (idempotent, adds missing columns)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db, table_count


def main() -> None:
    init_db()
    for table in ["stock_basic", "stock_daily", "factor_values", "prediction_result", "backtest_result"]:
        print(f"{table}: {table_count(table)}")


if __name__ == "__main__":
    main()
