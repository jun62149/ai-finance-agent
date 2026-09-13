"""SQLite access layer, migrated unchanged from the legacy B codebase.

Schema and upsert semantics intentionally match the graduation project so the
Quant Engine remains byte-for-byte reproducible on the same data.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from app.config import Config
from app.log import ensure_dirs


SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_basic (
    stock_code TEXT PRIMARY KEY,
    stock_name TEXT,
    market TEXT,
    list_date TEXT,
    is_active INTEGER DEFAULT 1,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS stock_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    amount REAL,
    pct_change REAL,
    turnover REAL,
    source TEXT,
    created_at TEXT,
    UNIQUE(stock_code, trade_date)
);

CREATE TABLE IF NOT EXISTS factor_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    ret_5d REAL,
    ret_10d REAL,
    ret_20d REAL,
    ma5 REAL,
    ma10 REAL,
    ma20 REAL,
    ma5_bias REAL,
    ma20_bias REAL,
    volatility_10d REAL,
    volatility_20d REAL,
    volume_ratio_5d REAL,
    amount_mean_20d REAL,
    turnover_mean_20d REAL,
    drawdown_20d REAL,
    future_return_5d REAL,
    future_excess_return_5d REAL,
    strong_label INTEGER,
    UNIQUE(stock_code, trade_date)
);

CREATE TABLE IF NOT EXISTS prediction_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    pred_return_5d REAL,
    strong_prob REAL,
    risk_score REAL,
    score REAL,
    close REAL,
    pct_change REAL,
    board TEXT,
    main_factor TEXT,
    is_recommendable INTEGER DEFAULT 1,
    created_at TEXT,
    UNIQUE(trade_date, stock_code)
);

CREATE TABLE IF NOT EXISTS backtest_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL UNIQUE,
    top10_return REAL,
    top10_return_before_cost REAL,
    benchmark_return REAL,
    excess_return REAL,
    hit_rate REAL,
    nav_model REAL,
    nav_base REAL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    doc_type TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    publish_time TEXT,
    content TEXT NOT NULL,
    url TEXT,
    content_hash TEXT UNIQUE,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_stock_publish
    ON documents(stock_code, publish_time);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    doc_type TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    publish_time TEXT,
    url TEXT,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    embedding_version TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_stock_publish
    ON chunks(stock_code, publish_time);
"""


def init_db() -> None:
    ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)
        _ensure_column(conn, "prediction_result", "board", "TEXT")
        _ensure_column(conn, "backtest_result", "top10_return_before_cost", "REAL")
        _ensure_column(conn, "backtest_result", "top_n", "INTEGER")
        _ensure_column(conn, "backtest_result", "include_growth", "INTEGER")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def read_sql(query: str, params: tuple | dict | None = None) -> pd.DataFrame:
    init_db()
    with connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


def upsert_dataframe(df: pd.DataFrame, table: str, key_columns: list[str]) -> int:
    if df.empty:
        return 0

    init_db()
    rows = df.copy()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if table == "stock_basic" and "updated_at" not in rows.columns:
        rows["updated_at"] = now
    if table in {"stock_daily", "prediction_result"} and "created_at" not in rows.columns:
        rows["created_at"] = now

    columns = list(rows.columns)
    placeholders = ", ".join(["?"] * len(columns))
    update_columns = [c for c in columns if c not in key_columns]
    updates = ", ".join([f"{c}=excluded.{c}" for c in update_columns])
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({', '.join(key_columns)}) DO UPDATE SET {updates}"
    )

    with connect() as conn:
        conn.executemany(sql, rows[columns].itertuples(index=False, name=None))
    return len(rows)


def table_count(table: str) -> int:
    init_db()
    with connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])


def existing_content_hashes(content_hashes: list[str]) -> set[str]:
    """Return the subset of content_hashes already stored in `documents`."""
    if not content_hashes:
        return set()
    init_db()
    found: set[str] = set()
    with connect() as conn:
        for i in range(0, len(content_hashes), 400):
            batch = content_hashes[i : i + 400]
            placeholders = ", ".join(["?"] * len(batch))
            rows = conn.execute(
                f"SELECT content_hash FROM documents WHERE content_hash IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            found.update(r["content_hash"] for r in rows)
    return found


def existing_announcement_keys(stock_code: str) -> set[tuple[str, str]]:
    """Return {(title, publish_time)} already stored for a stock's announcements.

    Used by the announcement ingest pipeline to skip re-fetching bodies for
    documents that are already in the knowledge base (resume / rerun support).
    """
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT title, publish_time FROM documents
            WHERE stock_code = ? AND doc_type = 'announcement'
              AND content IS NOT NULL AND length(content) > 0
            """,
            (stock_code,),
        ).fetchall()
    return {(str(r["title"]), str(r["publish_time"] or "")) for r in rows}


def purge_documents(doc_type: str, source: str | None = None) -> tuple[int, int]:
    """Delete documents of a doc_type (optionally one source) plus their chunks.

    Used when switching an ingestion source (e.g. eastmoney -> cninfo) so the
    knowledge base does not keep two copies of the same announcement.
    Returns (deleted_documents, deleted_chunks).
    """
    init_db()
    where = "doc_type = ?" + (" AND source = ?" if source else "")
    params: tuple = (doc_type, source) if source else (doc_type,)
    with connect() as conn:
        ids = [
            str(r["document_id"])
            for r in conn.execute(
                f"SELECT document_id FROM documents WHERE {where}", params
            ).fetchall()
        ]
        if not ids:
            return 0, 0
        chunks_deleted = 0
        for i in range(0, len(ids), 400):
            batch = ids[i : i + 400]
            placeholders = ", ".join(["?"] * len(batch))
            chunks_deleted += conn.execute(
                f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
                tuple(batch),
            ).rowcount
        docs_deleted = conn.execute(
            f"DELETE FROM documents WHERE {where}", params
        ).rowcount
    return int(docs_deleted), int(chunks_deleted)


def insert_documents(rows: list[dict]) -> int:
    """Idempotent bulk insert (INSERT OR IGNORE) into documents."""
    if not rows:
        return 0
    init_db()
    columns = [
        "document_id",
        "stock_code",
        "stock_name",
        "doc_type",
        "title",
        "source",
        "publish_time",
        "content",
        "url",
        "content_hash",
        "created_at",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT OR IGNORE INTO documents ({', '.join(columns)}) VALUES ({placeholders})"
    with connect() as conn:
        conn.executemany(
            sql, [tuple(r.get(c) for c in columns) for r in rows]
        )
    return len(rows)


def insert_chunks(rows: list[dict]) -> int:
    """Idempotent bulk insert (INSERT OR IGNORE) into chunks."""
    if not rows:
        return 0
    init_db()
    columns = [
        "chunk_id",
        "document_id",
        "stock_code",
        "stock_name",
        "doc_type",
        "title",
        "source",
        "publish_time",
        "url",
        "content",
        "chunk_index",
        "chunk_hash",
        "embedding_version",
        "created_at",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT OR IGNORE INTO chunks ({', '.join(columns)}) VALUES ({placeholders})"
    with connect() as conn:
        conn.executemany(
            sql, [tuple(r.get(c) for c in columns) for r in rows]
        )
    return len(rows)


def clear_table(table: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute(f"DELETE FROM {table}")


def fetch_chunks_for_embedding(
    version: str,
    *,
    limit: int | None = None,
    doc_type: str | None = None,
    after_id: int = 0,
) -> list[dict]:
    """Chunks that still need a vector for `version` (pending or outdated).

    `after_id` makes the scan resumable by ascending chunk id.
    """
    init_db()
    where = ["id > ?", "(embedding_version IS NULL OR embedding_version != ?)"]
    params: list = [after_id, version]
    if doc_type:
        where.append("doc_type = ?")
        params.append(doc_type)
    sql = f"""
        SELECT id, chunk_id, document_id, stock_code, stock_name, doc_type,
               title, source, publish_time, url, content, chunk_index
        FROM chunks
        WHERE {' AND '.join(where)}
        ORDER BY id
    """
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    with connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def set_chunks_embedding_version(ids: list[int], version: str) -> int:
    """Mark chunks as embedded with `version` (called after a successful upsert)."""
    if not ids:
        return 0
    init_db()
    updated = 0
    with connect() as conn:
        for i in range(0, len(ids), 400):
            batch = ids[i : i + 400]
            placeholders = ", ".join(["?"] * len(batch))
            updated += conn.execute(
                f"UPDATE chunks SET embedding_version = ? WHERE id IN ({placeholders})",
                (version, *batch),
            ).rowcount
    return int(updated)


def fetch_chunks_for_bm25(version: str | None = None) -> list[dict]:
    """Chunk rows used to build the BM25 index (same corpus as the vectors)."""
    init_db()
    where = ""
    params: tuple = ()
    if version:
        where = "WHERE embedding_version = ?"
        params = (version,)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, chunk_id, document_id, stock_code, stock_name, doc_type,
                   title, source, publish_time, url, chunk_index, content
            FROM chunks
            {where}
            ORDER BY id
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_chunk_contents(ids: list[int]) -> dict[int, str]:
    """Map chunk row ids -> content (Qdrant stores metadata; text stays here)."""
    if not ids:
        return {}
    init_db()
    out: dict[int, str] = {}
    with connect() as conn:
        for i in range(0, len(ids), 400):
            batch = ids[i : i + 400]
            placeholders = ", ".join(["?"] * len(batch))
            for row in conn.execute(
                f"SELECT id, content FROM chunks WHERE id IN ({placeholders})",
                tuple(batch),
            ).fetchall():
                out[int(row["id"])] = str(row["content"])
    return out


def embedding_stats(version: str) -> dict[str, dict[str, int]]:
    """Per doc_type chunk counts: total / embedded with `version` / pending."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT doc_type,
                   COUNT(*) AS total,
                   SUM(CASE WHEN embedding_version = ? THEN 1 ELSE 0 END) AS embedded
            FROM chunks
            GROUP BY doc_type
            """,
            (version,),
        ).fetchall()
    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        total = int(row["total"] or 0)
        embedded = int(row["embedded"] or 0)
        stats[str(row["doc_type"])] = {
            "total": total,
            "embedded": embedded,
            "pending": total - embedded,
        }
    return stats


def latest_trade_date() -> str:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT MAX(trade_date) AS trade_date FROM stock_daily").fetchone()
    return row["trade_date"] or "-"


def database_exists() -> bool:
    return Path(Config.DB_PATH).exists()
