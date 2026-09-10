"""Phase 1: fetch announcements (list + body) and ingest.

Sources:
    cninfo    (default) 巨潮资讯网 official disclosure platform — list via
              new/hisAnnouncement/query, body = disclosure PDF -> text (pypdf).
              Not affected by East Money's IP-level notice throttling.
    eastmoney (legacy)  East Money notice API; kept as fallback, but its
              content host throttles/blocks aggressive clients.

Resumable + idempotent: rows already stored for (code, title, publish_time)
are skipped before any body request, so re-running the same command only
fetches genuinely new bodies and reports 0 new when the range is complete.

Usage:
    python scripts/ingest_announcements.py                                   # corpus, cninfo
    python scripts/ingest_announcements.py --source eastmoney                # legacy source
    python scripts/ingest_announcements.py --codes 600519 --days 30 --max-bodies 5

Tuning knobs:
    --workers N                 concurrent body fetches per stock (default 4)
    --delay SECONDS             min interval between requests (default 0.15)
    --max-consecutive-fail-stocks N
                                stop after N stocks in a row fail EVERY request
    --refetch-existing          ignore already-ingested rows and refetch bodies

Switching source: the resume key is (code, title, publish_time) and different
sources title the same announcement differently, so purge the old source first
(scripts/purge_documents.py) to avoid keeping two copies of one announcement.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import read_sql
from app.rag.ingestion import (
    ANNOUNCEMENT_SOURCES,
    build_corpus,
    ingest_announcements_for_stocks,
)

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


def _print_stats_table(stats_rows: list[dict]) -> None:
    if not stats_rows:
        return
    header = ("code", "name", "list", "new", "chunks", "chars", "skip", "err", "empty")
    widths = {"code": 6, "name": 10, "list": 5, "new": 5, "chunks": 6,
              "chars": 9, "skip": 5, "err": 4, "empty": 5}
    print("per-code detail:")
    print(
        "  " + "  ".join(h.ljust(widths[h]) for h in header)
    )
    for s in stats_rows:
        cells = (
            str(s["code"]).ljust(widths["code"]),
            str(s["name"])[:10].ljust(widths["name"]),
            str(s["list_rows"]).ljust(widths["list"]),
            str(s["new_docs"]).ljust(widths["new"]),
            str(s["new_chunks"]).ljust(widths["chunks"]),
            str(s.get("chars", 0)).ljust(widths["chars"]),
            str(s["skipped"]).ljust(widths["skip"]),
            str(s["errors"]).ljust(widths["err"]),
            str(s["empty"]).ljust(widths["empty"]),
        )
        print("  " + "  ".join(cells))
        if s["sample_error"]:
            print(f"    ^ error sample: {s['sample_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest announcements (resumable)")
    parser.add_argument("--source", default="cninfo", choices=list(ANNOUNCEMENT_SOURCES),
                        help="announcement source (default: cninfo, official disclosure)")
    parser.add_argument("--codes", help="comma-separated 6-digit codes (overrides corpus mode)")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-bodies", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.15,
                        help="min interval seconds between body requests")
    parser.add_argument("--page-size", type=int, default=50,
                        help="cninfo list page size")
    parser.add_argument("--max-consecutive-fail-stocks", type=int, default=3,
                        help="stop early after this many consecutive fully-failed stocks")
    parser.add_argument("--refetch-existing", action="store_true",
                        help="refetch bodies even for already-ingested rows")
    parser.add_argument("--corpus-size", type=int, default=25)
    parser.add_argument("--no-flagships", action="store_true")
    args = parser.parse_args()

    end_date = date.today().isoformat()
    begin_date = (date.today() - timedelta(days=args.days)).isoformat()
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

    print(
        f"source={args.source} range={begin_date}..{end_date} stocks={len(stock_list)} "
        f"max_bodies_per_code={args.max_bodies} workers={args.workers} "
        f"delay={args.delay}s refetch_existing={args.refetch_existing}"
    )
    t0 = time.time()
    total_stocks = len(stock_list)

    def progress(code: str, stats: dict, index: int) -> None:
        line = (
            f"[{index}/{total_stocks}] {code} list={stats['list_rows']} "
            f"new={stats['new_docs']}/{stats['new_chunks']} skip={stats['skipped']} "
            f"err={stats['errors']} empty={stats['empty']}"
        )
        if stats["sample_error"]:
            line += f" | {stats['sample_error']}"
        print(line, flush=True)

    totals = ingest_announcements_for_stocks(
        stock_list,
        begin_date=begin_date,
        end_date=end_date,
        source=args.source,
        max_bodies_per_code=args.max_bodies,
        workers=args.workers,
        request_delay=args.delay,
        max_consecutive_fail_stocks=args.max_consecutive_fail_stocks,
        list_page_size=args.page_size,
        refetch_existing=args.refetch_existing,
        progress=progress,
    )

    print("")
    _print_stats_table(totals["per_code"])
    print("")
    print(
        f"done in {time.time() - t0:.1f}s | source={totals.get('source', args.source)} "
        f"new_docs={totals['docs']} new_chunks={totals['chunks']} "
        f"new_chars={totals.get('chars', 0)} "
        f"skipped_existing={totals['skipped_existing']} "
        f"errors={totals['errors']} empty_bodies={totals['empty_bodies']} "
        f"list_rows={totals['list_rows']}"
    )

    if totals["stopped_early"]:
        print("")
        print(f"[!] stopped early: {totals['stopped_reason']}")
    elif totals["docs"] == 0 and totals["errors"] == 0:
        print("")
        print("[ok] 0 new docs and no errors: the range is fully ingested "
              "(rerun-idempotent).")

    print("")
    print("next QA step: rerun the same command to confirm it reports 0 new docs "
          "and 0 errors.")


if __name__ == "__main__":
    main()
