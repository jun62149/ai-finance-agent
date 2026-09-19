"""Ingestion service: raw items -> cleaned docs -> semantic chunks.

Idempotency: a document is keyed by `content_hash` = sha256(stock_code |
title | body). Re-running ingestion skips existing hashes and only inserts
new rows, so the pipeline is resumable.

Scope decision (2026-09-08): dedupe is PER-STOCK on purpose. Market-wide
articles (e.g. capital-flow roundups) legitimately relate to many stocks;
keeping one copy per tagged stock makes stock_code-filtered retrieval work.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Iterable

from app.db import (
    existing_announcement_keys,
    existing_content_hashes,
    insert_chunks,
    insert_documents,
)
from app.rag.chunker import Chunk, chunk_text
from app.rag.sources import (
    fetch_announcement_body,
    fetch_announcement_list,
    fetch_cninfo_announcement_body,
    fetch_cninfo_announcement_list,
    fetch_news_pages,
)
from app.timeutil import now_stamp


ProgressFn = Callable[[str, int, int, int], None]

AnnouncementStats = dict[str, Any]
AnnouncementProgressFn = Callable[[str, AnnouncementStats, int], None]

ANNOUNCEMENT_SOURCES = ("eastmoney", "cninfo")


def _eastmoney_body(row: dict[str, Any]) -> tuple[str, str] | None:
    return fetch_announcement_body(row["art_code"])


def _cninfo_body(row: dict[str, Any]) -> tuple[str, str] | None:
    return fetch_cninfo_announcement_body(row)


def _row_key(row: dict[str, Any]) -> str:
    """Stable per-announcement key used to attach fetched bodies to list rows."""
    return str(row.get("art_code") or row.get("url") or "")


def _announcement_adapters(
    source: str,
    *,
    page_size: int = 50,
    max_rows: int | None = None,
) -> tuple[Callable[..., list[dict[str, Any]]], Callable[[dict[str, Any]], tuple[str, str] | None]]:
    """Resolve (list_fn, body_fn) for an announcement source.

    Module attributes are looked up at call time so tests can monkeypatch them.
    """
    if source == "cninfo":
        return (
            partial(
                fetch_cninfo_announcement_list,
                page_size=page_size,
                max_rows=max_rows,
            ),
            _cninfo_body,
        )
    if source == "eastmoney":
        return fetch_announcement_list, _eastmoney_body
    raise ValueError(
        f"unknown announcement source {source!r}; expected one of {ANNOUNCEMENT_SOURCES}"
    )


class _RateLimiter:
    """Process-wide min-interval pacing between outgoing requests."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._next_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_at - now
            self._next_at = max(now, self._next_at) + self._min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def _now() -> str:
    """Audit/display stamp only (Beijing wall clock), for created_at rows.

    Never used for filtering, ranking or attribution, and unrelated to
    `publish_ts` or the market-date arithmetic.
    """
    return now_stamp()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document_hash(code: str, title: str, content: str) -> str:
    return _hash_text(f"{code}|{title}|{content}")


def save_documents(
    docs: list[dict[str, Any]],
    *,
    source: str,
    doc_type: str,
    code: str,
    stock_name: str,
) -> tuple[int, int]:
    """Dedupe + insert documents and their chunks. Returns (new_docs, new_chunks)."""
    prepared = []
    for item in docs:
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title or len(content) < 20:
            continue
        content_hash = _document_hash(code, title, content)
        publish_time = str(item.get("publish_time") or "").strip()
        doc_id = f"{doc_type}-{source}-{code}-{content_hash[:16]}"
        prepared.append(
            {
                "document_id": doc_id,
                "stock_code": code,
                "stock_name": stock_name or None,
                "doc_type": doc_type,
                "title": title,
                "source": source,
                "publish_time": publish_time or None,
                "content": content,
                "url": str(item.get("url") or "").strip() or None,
                "content_hash": content_hash,
                "created_at": _now(),
            }
        )
    if not prepared:
        return 0, 0

    existing = existing_content_hashes([d["content_hash"] for d in prepared])
    new_docs = [d for d in prepared if d["content_hash"] not in existing]
    insert_documents(new_docs)

    chunk_rows = []
    now = _now()
    for doc in new_docs:
        chunks: list[Chunk] = chunk_text(doc["content"])
        for chunk in chunks:
            chunk_rows.append(
                {
                    "chunk_id": f"{doc['document_id']}:c{chunk.chunk_index:03d}",
                    "document_id": doc["document_id"],
                    "stock_code": code,
                    "stock_name": doc["stock_name"],
                    "doc_type": doc_type,
                    "title": doc["title"],
                    "source": source,
                    "publish_time": doc["publish_time"],
                    "url": doc["url"],
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "chunk_hash": chunk.chunk_hash,
                    "embedding_version": None,
                    "created_at": now,
                }
            )
    insert_chunks(chunk_rows)
    return len(new_docs), len(chunk_rows)


def ingest_news_for_stocks(
    stock_list: list[tuple[str, str]],
    *,
    start_date: str,
    max_pages: int = 6,
    page_size: int = 50,
    progress: ProgressFn | None = None,
) -> dict[str, int]:
    """Fetch + ingest news for many stocks. Returns total {docs, chunks, errors}."""
    totals = {"docs": 0, "chunks": 0, "errors": 0}
    for index, (code, name) in enumerate(stock_list, start=1):
        try:
            items = fetch_news_pages(
                code,
                page_size=page_size,
                max_pages=max_pages,
                min_publish_date=start_date,
            )
            in_range = [i for i in items if i["publish_time"] >= start_date]
            docs, chunks = save_documents(
                in_range,
                source="eastmoney",
                doc_type="news",
                code=code,
                stock_name=name,
            )
            totals["docs"] += docs
            totals["chunks"] += chunks
            if progress:
                progress(code, docs, chunks, index)
        except Exception:  # noqa: BLE001 - one bad source must not kill the run
            totals["errors"] += 1
            if progress:
                progress(code, -1, 0, index)
    return totals


def ingest_announcements_for_stocks(
    stock_list: list[tuple[str, str]],
    *,
    begin_date: str,
    end_date: str,
    source: str = "eastmoney",
    max_bodies_per_code: int = 300,
    workers: int = 4,
    request_delay: float = 0.15,
    max_consecutive_fail_stocks: int = 3,
    list_page_size: int = 50,
    refetch_existing: bool = False,
    progress: AnnouncementProgressFn | None = None,
) -> dict[str, Any]:
    """Fetch announcement metadata + bodies and ingest. Resumable + resilient.

    ``source`` selects the adapter: ``cninfo`` (巨潮资讯网, official, default
    for the CLI) or ``eastmoney`` (legacy, subject to IP-level throttling).

    Resume / idempotency:
    - Rows already stored for the same (code, title, publish_time) are skipped
      BEFORE any body request, so reruns only fetch genuinely new bodies
      (after the ``max_bodies_per_code`` cap, keeping reruns stable at 0 new).
    - ``save_documents`` still dedupes by content_hash as a final guard.

    Resilience:
    - Body requests are paced process-wide by ``request_delay`` seconds.
    - Per-stock failures are collected and surfaced through ``progress`` and
      the returned totals (errors vs. empty bodies vs. skipped).
    - When ``max_consecutive_fail_stocks`` stocks in a row fail EVERY request
      (likely upstream blocking / throttling), the run stops early with
      ``totals["stopped_early"]`` instead of hammering the endpoint.
    """
    list_fn, body_fn = _announcement_adapters(
        source,
        page_size=list_page_size,
        max_rows=max_bodies_per_code or None,
    )
    totals: dict[str, Any] = {
        "source": source,
        "docs": 0,
        "chunks": 0,
        "chars": 0,
        "errors": 0,
        "empty_bodies": 0,
        "skipped_existing": 0,
        "list_rows": 0,
        "stopped_early": False,
        "stopped_reason": "",
        "per_code": [],
    }
    limiter = _RateLimiter(request_delay)
    consecutive_fail_stocks = 0

    def fetch_body_paced(row: dict[str, Any]) -> tuple[str, str] | None:
        limiter.wait()
        return body_fn(row)

    for index, (code, name) in enumerate(stock_list, start=1):
        stats: AnnouncementStats = {
            "code": code,
            "name": name,
            "list_rows": 0,
            "new_docs": 0,
            "new_chunks": 0,
            "chars": 0,
            "skipped": 0,
            "errors": 0,
            "empty": 0,
            "sample_error": "",
        }
        try:
            meta = list_fn(code, begin_date, end_date)
            if max_bodies_per_code:
                meta = meta[:max_bodies_per_code]
            stats["list_rows"] = len(meta)
            if not refetch_existing and meta:
                existing = existing_announcement_keys(code)
                pending = [
                    row for row in meta
                    if (row["title"], row["publish_time"]) not in existing
                ]
                stats["skipped"] = len(meta) - len(pending)
                meta = pending

            bodies: dict[str, tuple[str, str]] = {}
            if meta:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    future_map = {
                        pool.submit(fetch_body_paced, row): row for row in meta
                    }
                    for future in as_completed(future_map):
                        row = future_map[future]
                        try:
                            body = future.result()
                        except Exception as exc:  # noqa: BLE001 - count + continue
                            stats["errors"] += 1
                            if not stats["sample_error"]:
                                stats["sample_error"] = (
                                    f"{type(exc).__name__}: {str(exc)[:140]}"
                                )
                            continue
                        if body:
                            bodies[_row_key(row)] = body
                        else:
                            stats["empty"] += 1

                items = []
                for row in meta:
                    body = bodies.get(_row_key(row))
                    if not body:
                        continue
                    title, content = body
                    items.append(
                        {
                            "title": title or row["title"],
                            "content": content,
                            "publish_time": row["publish_time"],
                            "url": row["url"],
                        }
                    )
                docs, chunks = save_documents(
                    items,
                    source=source,
                    doc_type="announcement",
                    code=code,
                    stock_name=name,
                )
                stats["new_docs"], stats["new_chunks"] = docs, chunks
                stats["chars"] = sum(len(item["content"]) for item in items)
        except Exception as exc:  # noqa: BLE001 - one bad stock must not kill run
            stats["errors"] += 1
            if not stats["sample_error"]:
                stats["sample_error"] = f"{type(exc).__name__}: {str(exc)[:140]}"

        totals["docs"] += stats["new_docs"]
        totals["chunks"] += stats["new_chunks"]
        totals["chars"] += stats["chars"]
        totals["errors"] += stats["errors"]
        totals["empty_bodies"] += stats["empty"]
        totals["skipped_existing"] += stats["skipped"]
        totals["list_rows"] += stats["list_rows"]
        totals["per_code"].append(stats)
        if progress:
            progress(code, stats, index)

        stock_fully_failed = (
            stats["new_docs"] == 0
            and stats["errors"] > 0
            and (stats["list_rows"] == 0 or stats["errors"] >= stats["list_rows"])
        )
        consecutive_fail_stocks = consecutive_fail_stocks + 1 if stock_fully_failed else 0
        if consecutive_fail_stocks >= max_consecutive_fail_stocks:
            totals["stopped_early"] = True
            totals["stopped_reason"] = (
                f"{consecutive_fail_stocks} consecutive stocks failed every request "
                f"(last sample: {stats['sample_error'] or 'n/a'}); the upstream may be "
                "blocking/throttling. Progress is saved — rerun the same command later "
                "to resume (already-ingested rows are skipped)."
            )
            break
    return totals


def build_corpus(
    size: int = 25,
    flagship_codes: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Pick the most liquid stocks from local factor data, plus flagship names.

    Corpus decision (2026-09-08): top-liquid active names from the local DB so
    the knowledge base overlaps the Quant pool, plus well-known names without
    daily bars (e.g. 600519) that demo queries need.
    """
    import pandas as pd

    from app.db import read_sql

    codes: list[str] = []
    try:
        df = read_sql(
            """
            SELECT f.stock_code, f.amount_mean_20d
            FROM factor_values f
            JOIN (SELECT MAX(trade_date) AS trade_date FROM factor_values) latest
              ON f.trade_date = latest.trade_date
            WHERE f.amount_mean_20d > 50000000
            ORDER BY f.amount_mean_20d DESC
            LIMIT ?
            """,
            (size,),
        )
        codes = [str(c).zfill(6) for c in df["stock_code"].tolist()]
    except Exception:  # noqa: BLE001 - corpus builder must never break scripts
        codes = []

    basic = read_sql(
        "SELECT stock_code, stock_name FROM stock_basic WHERE is_active = 1"
    )
    names = {
        str(r.stock_code).zfill(6): str(r.stock_name)
        for r in basic.itertuples(index=False)
    }
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for code in codes + [c for c in flagship_codes if c not in codes]:
        if code in seen or code not in names:
            continue
        seen.add(code)
        selected.append((code, names[code]))
    return selected
