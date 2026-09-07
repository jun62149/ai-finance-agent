"""Rebuild factor_values for the whole stock_daily universe (legacy semantics)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db
from app.quant.features import rebuild_factors


def main() -> None:
    init_db()
    result = rebuild_factors()
    print(f"factor rebuild done: {len(result)} rows computed")


if __name__ == "__main__":
    main()
