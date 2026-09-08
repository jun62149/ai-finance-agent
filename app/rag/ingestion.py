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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Iterable

from app.db import (
    existing_content_hashes,
    insert_chunks,
    insert_documents,
)
from app.rag.chunker import Chunk, chunk_text
from app.rag.sources import (
    fetch_announcement_body,
    fetch_announcement_list,
    fetch_news_pages,
)


ProgressFn = Callable[[str, int, int, int], None]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    max_bodies_per_code: int = 300,
    workers: int = 8,
    progress: ProgressFn | None = None,
) -> dict[str, int]:
    """Fetch announcement metadata + bodies and ingest. Returns totals."""
    totals = {"docs": 0, "chunks": 0, "errors": 0}
    for index, (code, name) in enumerate(stock_list, start=1):
        code_docs = 0
        code_chunks = 0
        try:
            meta = fetch_announcement_list(code, begin_date, end_date)
            if max_bodies_per_code:
                meta = meta[:max_bodies_per_code]
            bodies: dict[str, tuple[str, str]] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(fetch_announcement_body, row["art_code"]): row
                    for row in meta
                }
                for future in as_completed(future_map):
                    row = future_map[future]
                    try:
                        body = future.result()
                    except Exception:  # noqa: BLE001
                        totals["errors"] += 1
                        continue
                    if body:
                        bodies[row["art_code"]] = body
            items = []
            for row in meta:
                body = bodies.get(row["art_code"])
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
                source="eastmoney",
                doc_type="announcement",
                code=code,
                stock_name=name,
            )
            totals["docs"] += docs
            totals["chunks"] += chunks
            code_docs, code_chunks = docs, chunks
        except Exception:  # noqa: BLE001
            totals["errors"] += 1
        if progress:
            progress(code, code_docs, code_chunks, index)
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
