"""BM25 lexical retrieval over the chunk corpus (Phase 3).

Why lexical retrieval next to dense vectors: financial queries are full of
exact tokens that embeddings blur — stock codes (600519), dates, legal
announcement terms (重大资产重组), company short names. BM25 nails those.

Implementation notes:
- Tokenization: jieba (search mode) so Chinese text is segmented; tokens are
  lowercased and punctuation-only tokens are dropped (numbers/codes survive).
- `bm25s` keeps the index as a sparse matrix, which stays cheap at ~100k chunks.
- Metadata filtering is post-hoc (bm25s has no filter support): when a filter is
  given we retrieve a wider candidate window and filter afterwards.
- The index is persisted to `Config.BM25_PATH` (index + chunk ids + metadata)
  so the API/Agent can load it without re-tokenizing the corpus.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from app.config import Config
from app.errors import IndexOutOfSync
from app.log import get_logger
from app.timeutil import now_stamp

_TOKEN_KEEP = re.compile(r"[\w\u4e00-\u9fff]")
META_FILE = "chunks_meta.pkl"
MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_LOGGER = get_logger(__name__)


def chunk_id_hash(chunk_ids: Iterable[str]) -> str:
    """Order-independent fingerprint of a chunk-id set (first 16 hex chars)."""
    joined = "|".join(sorted(str(cid) for cid in chunk_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> list[str]:
    """Segment text for BM25 (jieba search mode, lowercased, punctuation out)."""
    import jieba

    tokens: list[str] = []
    for raw in jieba.cut_for_search(str(text or "")):
        token = raw.strip().lower()
        if token and _TOKEN_KEEP.search(token):
            tokens.append(token)
    return tokens


class BM25Index:
    """Lexical index over chunks, persisted on disk."""

    def __init__(
        self,
        retriever: Any,
        chunk_ids: list[str],
        metadatas: list[dict[str, Any]],
        *,
        embedding_version: str | None = None,
        all_chunks: bool = False,
    ) -> None:
        self.retriever = retriever
        self.chunk_ids = chunk_ids
        self.metadatas = metadatas
        # Provenance of the corpus this index was built from, persisted in the
        # manifest so a later load can prove the index still matches the data.
        self.embedding_version = embedding_version
        self.all_chunks = all_chunks

    def __len__(self) -> int:
        return len(self.chunk_ids)

    # -- build / persist ---------------------------------------------------- #
    @classmethod
    def build(
        cls,
        rows: list[dict[str, Any]],
        *,
        embedding_version: str | None = None,
        all_chunks: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> "BM25Index":
        """Build from chunk rows (needs `chunk_id`, `content`, metadata keys)."""
        import bm25s

        tokenized: list[list[str]] = []
        chunk_ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            tokenized.append(tokenize(row.get("content") or ""))
            chunk_ids.append(str(row["chunk_id"]))
            metadatas.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": row.get("document_id"),
                    "stock_code": row.get("stock_code"),
                    "stock_name": row.get("stock_name"),
                    "doc_type": row.get("doc_type"),
                    "title": row.get("title"),
                    "source": row.get("source"),
                    "publish_time": row.get("publish_time"),
                    "url": row.get("url"),
                    "chunk_index": row.get("chunk_index"),
                    "point_id": row.get("id"),
                }
            )
            if progress and index % 5000 == 0:
                progress(index, total)

        retriever = bm25s.BM25()
        retriever.index(tokenized)
        return cls(
            retriever, chunk_ids, metadatas,
            embedding_version=embedding_version, all_chunks=all_chunks,
        )

    def manifest(self) -> dict[str, Any]:
        """Small provenance record written next to the index."""
        try:
            import bm25s

            bm25s_version = getattr(bm25s, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - version is informational only
            bm25s_version = "unknown"
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "embedding_version": self.embedding_version,
            "all_chunks": self.all_chunks,
            "chunk_count": len(self.chunk_ids),
            "chunk_id_hash": chunk_id_hash(self.chunk_ids),
            "collection": Config.QDRANT_COLLECTION,
            "tokenizer": "jieba-cut_for_search",
            "bm25s_version": bm25s_version,
            # Provenance/display stamp only (Beijing wall clock). Nothing in
            # verify() reads it back: corpus identity is chunk_id_hash.
            "built_at": now_stamp(),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or Config.BM25_PATH)
        target.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(target / "index"))
        with (target / META_FILE).open("wb") as handle:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "metadatas": self.metadatas}, handle
            )
        (target / MANIFEST_FILE).write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def verify(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Compare the manifest with the current chunk corpus in SQLite.

        Returns a report dict; never raises (callers decide what a mismatch means).
        """
        from app.db import fetch_chunks_for_bm25

        version = manifest.get("embedding_version")
        if manifest.get("all_chunks"):
            version = None
        current = [str(row["chunk_id"]) for row in fetch_chunks_for_bm25(version)]
        current_hash = chunk_id_hash(current)
        expected_count = int(manifest.get("chunk_count") or 0)
        expected_hash = str(manifest.get("chunk_id_hash") or "")
        return {
            "ok": len(current) == expected_count and current_hash == expected_hash,
            "expected_count": expected_count,
            "current_count": len(current),
            "expected_hash": expected_hash,
            "current_hash": current_hash,
            "embedding_version": version,
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BM25Index":
        import bm25s

        target = Path(path or Config.BM25_PATH)
        index_dir = target / "index"
        meta_file = target / META_FILE
        if not index_dir.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"no BM25 index at {target}; build it with scripts/build_bm25.py"
            )
        retriever = bm25s.BM25.load(str(index_dir))
        with meta_file.open("rb") as handle:
            payload = pickle.load(handle)
        manifest_file = target / MANIFEST_FILE
        if not manifest_file.exists():
            _LOGGER.warning(
                "BM25 manifest is missing at %s: the index cannot be checked against "
                "the chunk corpus (rebuild with scripts/build_bm25.py to enable the guard)",
                target,
            )
            return cls(retriever, payload["chunk_ids"], payload["metadatas"])

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        index = cls(
            retriever, payload["chunk_ids"], payload["metadatas"],
            embedding_version=manifest.get("embedding_version"),
            all_chunks=bool(manifest.get("all_chunks")),
        )
        report = index.verify(manifest)
        if report["ok"]:
            _LOGGER.info(
                "BM25 index verified: %d chunks (hash %s, embedding_version=%s)",
                report["current_count"], report["current_hash"],
                report["embedding_version"],
            )
            return index

        message = (
            f"BM25 index is out of sync: built with {report['expected_count']} chunks "
            f"(hash {report['expected_hash']}) but the corpus now has "
            f"{report['current_count']} (hash {report['current_hash']}, "
            f"embedding_version={report['embedding_version']}); "
            "rebuild it with `python scripts/build_bm25.py`"
        )
        if Config.BM25_ALLOW_STALE:
            _LOGGER.warning("%s (BM25_ALLOW_STALE=1, continuing anyway)", message)
            return index
        raise IndexOutOfSync(message)

    @classmethod
    def exists(cls, path: str | Path | None = None) -> bool:
        target = Path(path or Config.BM25_PATH)
        return (target / "index").exists() and (target / META_FILE).exists()

    # -- query -------------------------------------------------------------- #
    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        stock_code: str | None = None,
        doc_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k chunks for a query; `bm25_score` is attached to each result."""
        hits, _ = self.search_with_diagnostics(
            query, top_k=top_k, stock_code=stock_code, doc_type=doc_type,
            date_from=date_from, date_to=date_to,
        )
        return hits

    def search_with_diagnostics(
        self,
        query: str,
        *,
        top_k: int = 20,
        stock_code: str | None = None,
        doc_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Top-k chunks plus the counters that explain how they were obtained.

        Filtering semantics (fixed in review batch 1, Issue 1): BM25 scores the
        **whole corpus** first and applies the metadata filters **afterwards**
        (`BM25_WINDOW=0`, the default). The previous code scored only a
        `max(top_k*10, 200)` window before filtering, so a filter such as
        `stock_code + doc_type` could silently hide documents that ranked outside
        that window (measured: 2 hits instead of 356 on the real corpus).

        Setting `BM25_WINDOW>0` restores an explicit, logged truncation for very
        large corpora. Diagnostics are returned instead of stored on the
        instance, so concurrent requests cannot read each other's counters.
        """
        corpus_size = len(self.chunk_ids)
        limit = int(Config.BM25_WINDOW)
        window = corpus_size if limit <= 0 else min(corpus_size, limit)
        diagnostics: dict[str, Any] = {
            "window": window,
            "unlimited": limit <= 0,
            "truncated": window < corpus_size,
            "corpus": corpus_size,
            "scanned": 0,
            "after_filter": 0,
            "returned": 0,
            "elapsed_ms": 0.0,
        }

        if not self.chunk_ids:
            return [], diagnostics
        tokens = tokenize(query)
        if not tokens:
            return [], diagnostics

        started = time.perf_counter()
        results, scores = self.retriever.retrieve(
            [tokens], k=window, sorted=True, show_progress=False
        )
        hits: list[dict[str, Any]] = []
        for position, doc_index in enumerate(results[0]):
            idx = int(doc_index)
            score = float(scores[0][position])
            if score <= 0:
                # bm25s pads the ranking with zero-score docs; they are noise
                # for retrieval (and would pollute RRF), so stop here.
                break
            diagnostics["scanned"] += 1
            meta = dict(self.metadatas[idx])
            if stock_code and meta.get("stock_code") != stock_code:
                continue
            if doc_type and meta.get("doc_type") != doc_type:
                continue
            published = str(meta.get("publish_time") or "")[:10]
            if date_from and published and published < date_from:
                continue
            if date_to and published and published > date_to:
                continue
            diagnostics["after_filter"] += 1
            meta["bm25_score"] = score
            hits.append(meta)
            if len(hits) >= top_k:
                break

        diagnostics["returned"] = len(hits)
        diagnostics["elapsed_ms"] = (time.perf_counter() - started) * 1000
        prefix = "window=ALL" if diagnostics["unlimited"] else f"window={window}"
        if diagnostics["truncated"]:
            prefix += " (TRUNCATED)"
        _LOGGER.info(
            "BM25 search: %s scanned=%d matched=%d returned=%d elapsed_ms=%.1f",
            prefix, diagnostics["scanned"], diagnostics["after_filter"],
            diagnostics["returned"], diagnostics["elapsed_ms"],
        )
        if diagnostics["truncated"]:
            _LOGGER.warning(
                "BM25 candidate truncation is active (BM25_WINDOW=%d < corpus=%d): "
                "results with metadata filters can be silently incomplete",
                limit, corpus_size,
            )
        return hits, diagnostics
