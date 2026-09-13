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

import json
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from app.config import Config

_TOKEN_KEEP = re.compile(r"[\w\u4e00-\u9fff]")
META_FILE = "chunks_meta.pkl"


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
    ) -> None:
        self.retriever = retriever
        self.chunk_ids = chunk_ids
        self.metadatas = metadatas

    def __len__(self) -> int:
        return len(self.chunk_ids)

    # -- build / persist ---------------------------------------------------- #
    @classmethod
    def build(
        cls,
        rows: list[dict[str, Any]],
        *,
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
        return cls(retriever, chunk_ids, metadatas)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or Config.BM25_PATH)
        target.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(target / "index"))
        with (target / META_FILE).open("wb") as handle:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "metadatas": self.metadatas}, handle
            )
        return target

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
        return cls(retriever, payload["chunk_ids"], payload["metadatas"])

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
        """Top-k chunks for a query; `bm25_score` is attached to each result.

        Metadata filters (stock / type / date) are applied after retrieval,
        because bm25s has no filtering support; a wider candidate window is
        scanned when any filter is active.
        """
        if not self.chunk_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        filtered = bool(stock_code or doc_type or date_from or date_to)
        window = min(len(self.chunk_ids), max(top_k * 10, 200) if filtered else top_k)
        results, scores = self.retriever.retrieve([tokens], k=window, sorted=True)
        hits: list[dict[str, Any]] = []
        for position, doc_index in enumerate(results[0]):
            idx = int(doc_index)
            score = float(scores[0][position])
            if score <= 0:
                # bm25s pads the ranking with zero-score docs; they are noise
                # for retrieval (and would pollute RRF), so stop here.
                break
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
            meta["bm25_score"] = score
            hits.append(meta)
            if len(hits) >= top_k:
                break
        return hits
