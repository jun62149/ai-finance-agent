"""Incremental daily-bar updater (Quant data freshness).

Why this exists: the quant data migrated from the graduation project stops at its
last crawl date, while the news/announcement corpus is crawled "last N days" — so
the two drift apart in time, which looks broken in a demo (and the Agent has to
apologise for it in every report).

Design decisions (all backed by evidence from the legacy project):

* **baostock first, akshare as fallback** — in the migrated database 70.4% of rows
  (1,086,398) came from baostock and only 29.6% from akshare, even though the
  legacy code tried akshare first. baostock is stable but must be retried on
  login (observed a transient 10002007 "network receive error" that succeeds on
  the next attempt).
* **One login for the whole run** — the legacy code did login/logout per stock
  (~1,000 handshakes wasted).
* **Resumable by construction** — each stock resumes from its own MAX(trade_date),
  so the script can be interrupted and re-run; a finished run reports 0 new rows.
* **Unit/format parity with the existing data** (do not change these, the model
  was trained on this convention):
  - price adjustment: baostock `adjustflag="3"`, akshare `adjust=""` (both 不复权);
  - baostock volume is divided by 100 (股 -> 手) to match akshare's 手;
  - baostock code format `sz.000725` / `sh.600519` via `infer_market()`.

Usage:
    # small sample first (recommended)
    python scripts/update_daily.py --codes 000725,600519,000333 --end 2026-09-12

    # full incremental update (1,043 stocks) - safe to re-run until 0 new rows
    python scripts/update_daily.py --sleep 0.15

    # after the bars land:
    python scripts/rebuild_factors.py
    python scripts/predict_latest.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db, read_sql, upsert_dataframe  # noqa: E402
from app.quant.pool import infer_market, normalize_stock_code  # noqa: E402
from app.timeutil import market_today  # noqa: E402

BAOSTOCK_FIELDS = "date,code,open,high,low,close,volume,amount,pctChg,turn"
DAILY_COLUMNS = [
    "stock_code", "trade_date", "open", "high", "low", "close",
    "volume", "amount", "pct_change", "turnover", "source",
]
AKSHARE_MAPPING = {
    "日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume", "成交额": "amount",
    "涨跌幅": "pct_change", "换手率": "turnover",
}


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Same normalisation as the legacy pipeline (types + date format)."""
    result = frame.copy()
    for column in DAILY_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["trade_date", "close"])
    return result[DAILY_COLUMNS]


class BaoStockSession:
    """One login per run, retried; queries are sequential (the client is not thread safe)."""

    def __init__(self, retries: int = 3, backoff: float = 2.0) -> None:
        self.retries = retries
        self.backoff = backoff
        self._module = None
        self.logged_in = False

    def login(self) -> bool:
        import baostock as bs

        self._module = bs
        last_error = ""
        for attempt in range(1, self.retries + 1):
            with contextlib.redirect_stdout(io.StringIO()):
                result = bs.login()
            if result.error_code == "0":
                self.logged_in = True
                return True
            last_error = f"{result.error_code}: {result.error_msg}"
            time.sleep(self.backoff * attempt)
        print(f"[warn] baostock login failed after {self.retries} attempts ({last_error}); "
              "falling back to akshare", file=sys.stderr)
        return False

    def fetch(self, stock_code: str, start: str, end: str) -> pd.DataFrame:
        if not self.logged_in:
            raise RuntimeError("baostock session is not logged in")
        bs_code = f"{infer_market(stock_code)}.{normalize_stock_code(stock_code)}"
        result = self._module.query_history_k_data_plus(
            bs_code, BAOSTOCK_FIELDS,
            start_date=start, end_date=end, frequency="d", adjustflag="3",
        )
        rows = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise RuntimeError(f"baostock query failed: {result.error_code} {result.error_msg}")
        if not rows:
            return pd.DataFrame(columns=DAILY_COLUMNS)
        frame = pd.DataFrame(rows, columns=result.fields).rename(
            columns={"date": "trade_date", "pctChg": "pct_change", "turn": "turnover"}
        )
        frame["stock_code"] = normalize_stock_code(stock_code)
        frame["source"] = "baostock"
        cleaned = _clean(frame)
        cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce") / 100  # 股 -> 手
        return cleaned

    def close(self) -> None:
        if self.logged_in and self._module is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                self._module.logout()
            self.logged_in = False


def fetch_daily_akshare(stock_code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_a_hist(
        symbol=normalize_stock_code(stock_code),
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="",
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    frame = raw.rename(columns={k: v for k, v in AKSHARE_MAPPING.items() if k in raw.columns})
    frame["stock_code"] = normalize_stock_code(stock_code)
    frame["source"] = "akshare"
    return _clean(frame)


def last_trade_date(stock_code: str) -> str | None:
    rows = read_sql(
        "SELECT MAX(trade_date) AS last_date FROM stock_daily WHERE stock_code = ?",
        (stock_code,),
    )
    value = rows.iloc[0]["last_date"] if not rows.empty else None
    return str(value) if value else None


def target_stocks(codes: list[str] | None, limit: int | None) -> list[str]:
    if codes:
        stocks = [normalize_stock_code(code) for code in codes]
    else:
        frame = read_sql("SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code")
        stocks = [normalize_stock_code(code) for code in frame["stock_code"].tolist()]
    return stocks[:limit] if limit else stocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental daily-bar update (baostock first)")
    parser.add_argument("--codes", default=None, help="comma-separated codes (small sample)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N stocks")
    parser.add_argument("--start", default=None, help="override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="end date (default: today)")
    parser.add_argument("--sleep", type=float, default=0.15, help="seconds between stocks")
    parser.add_argument("--retries", type=int, default=2, help="per-source query retries")
    parser.add_argument("--primary", default="baostock", choices=["baostock", "akshare"])
    parser.add_argument("--dry-run", action="store_true", help="show the plan, fetch nothing")
    args = parser.parse_args()

    init_db()
    # Market calendar day (UTC+8): the default end must not depend on the process
    # timezone. Line 212's strptime/date arithmetic stays untouched -- it is pure
    # date math on a DB value and carries no timezone.
    end_date = args.end or market_today().isoformat()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    stocks = target_stocks(codes, args.limit)
    if not stocks:
        print("no stocks to update", file=sys.stderr)
        raise SystemExit(2)

    order = ["baostock", "akshare"] if args.primary == "baostock" else ["akshare", "baostock"]
    print(
        f"stocks={len(stocks)} end={end_date} source_order={order} sleep={args.sleep}s "
        f"dry_run={args.dry_run}"
    )

    session = BaoStockSession()
    if not args.dry_run and "baostock" in order:
        session.login()

    started = time.time()
    total_rows = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    by_source: dict[str, int] = {}

    try:
        for index, stock_code in enumerate(stocks, start=1):
            last = last_trade_date(stock_code)
            start = args.start or (
                (datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
                if last else "2020-01-01"
            )
            if start > end_date:
                skipped += 1
                continue
            if args.dry_run:
                print(f"[{index}/{len(stocks)}] {stock_code} would fetch {start}..{end_date} (last={last})")
                continue

            frame = pd.DataFrame(columns=DAILY_COLUMNS)
            errors: list[str] = []
            for source in order:
                for attempt in range(1, args.retries + 1):
                    try:
                        if source == "baostock":
                            if not session.logged_in and not session.login():
                                raise RuntimeError("baostock unavailable")
                            frame = session.fetch(stock_code, start, end_date)
                        else:
                            frame = fetch_daily_akshare(stock_code, start, end_date)
                        errors = []
                        break
                    except Exception as exc:  # noqa: BLE001 - try the next source/attempt
                        errors.append(f"{source}#{attempt}: {type(exc).__name__}: {exc}")
                        time.sleep(0.5 * attempt)
                if not frame.empty:
                    break

            if frame.empty:
                if errors:
                    failures.append((stock_code, "; ".join(errors[-2:])))
                    print(f"[{index}/{len(stocks)}] {stock_code} FAILED {errors[-1]}")
                else:
                    print(f"[{index}/{len(stocks)}] {stock_code} no rows in {start}..{end_date}")
            else:
                written = upsert_dataframe(frame, "stock_daily", ["stock_code", "trade_date"])
                total_rows += written
                source = str(frame["source"].iloc[0])
                by_source[source] = by_source.get(source, 0) + written
                print(
                    f"[{index}/{len(stocks)}] {stock_code} +{written} rows "
                    f"({frame['trade_date'].min()}..{frame['trade_date'].max()}, {source})"
                )
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        session.close()

    elapsed = time.time() - started
    print("")
    print(
        f"done in {elapsed:.1f}s | stocks={len(stocks)} rows_added={total_rows} "
        f"already_current={skipped} failures={len(failures)} by_source={by_source}"
    )
    if failures:
        print("failed stocks (re-run to retry, the script resumes per stock):")
        for stock_code, reason in failures[:20]:
            print(f"  {stock_code}: {reason}")
    if not args.dry_run:
        print("")
        print("next: python scripts/rebuild_factors.py && python scripts/predict_latest.py")


if __name__ == "__main__":
    main()
