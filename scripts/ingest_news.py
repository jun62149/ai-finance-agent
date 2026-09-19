"""Phase 1: fetch East Money per-stock news and ingest into documents/chunks.

Usage:
    python scripts/ingest_news.py                          # corpus mode (30 names)
    python scripts/ingest_news.py --codes 600519,000333    # pilot mode
    python scripts/ingest_news.py --days 365 --pages 6
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import read_sql
from app.rag.ingestion import build_corpus, ingest_news_for_stocks
from app.timeutil import market_today


FLAGSHIP_CODES = [
    "600519",  # 贵州茅台
    "601318",  # 中国平安
    "600036",  # 招商银行
    "000858",  # 五粮液
    "601012",  # 隆基绿能
    "300750",  # 宁德时代
    "002594",  # 比亚迪
]


def _resolve_names(codes: list[str]) -> list[tuple[str, str]]:
    basic = read_sql(
        "SELECT stock_code, stock_name FROM stock_basic WHERE is_active = 1"
    )
    names = {str(r.stock_code).zfill(6): str(r.stock_name) for r in basic.itertuples(index=False)}
    return [(c, names.get(c, "")) for c in codes if c in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest East Money stock news")
    parser.add_argument("--codes", help="comma-separated 6-digit codes (overrides corpus mode)")
    parser.add_argument("--days", type=int, default=365, help="keep news newer than N days")
    parser.add_argument("--pages", type=int, default=6, help="max news pages per stock")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--corpus-size", type=int, default=25)
    parser.add_argument("--no-flagships", action="store_true")
    args = parser.parse_args()

    # A-share market calendar day (UTC+8), not the process-local date: the fetch
    # window must be the same whatever timezone the host or container runs in.
    start_date = (market_today() - timedelta(days=args.days)).isoformat()
    if args.codes:
        stock_list = _resolve_names([c.strip() for c in args.codes.split(",") if c.strip()])
    else:
        stock_list = build_corpus(
            size=args.corpus_size,
            flagship_codes=() if args.no_flagships else FLAGSHIP_CODES,
        )
    if not stock_list:
        print("no valid stock codes", file=sys.stderr)
        raise SystemExit(2)

    print(f"start_date={start_date} stocks={len(stock_list)}")
    t0 = time.time()

    def progress(code: str, docs: int, chunks: int, index: int) -> None:
        tag = f"{docs} docs / {chunks} chunks" if docs >= 0 else "error"
        print(f"[{index}/{len(stock_list)}] {code} {tag}")

    totals = ingest_news_for_stocks(
        stock_list,
        start_date=start_date,
        max_pages=args.pages,
        page_size=args.page_size,
        progress=progress,
    )
    print(
        f"done in {time.time() - t0:.1f}s: "
        f"new_docs={totals['docs']} new_chunks={totals['chunks']} errors={totals['errors']}"
    )


if __name__ == "__main__":
    main()
