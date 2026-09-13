"""Embedding + vector-store orchestration (Phase 2, P2).

`build_embeddings` walks the chunks that have no vector for the current
`EMBEDDING_VERSION`, embeds them, upserts into Qdrant and only then marks the
chunks as embedded in SQLite. Ordering matters: SQLite is the source of truth
for "already embedded", so a crash between upsert and marking simply re-does
that batch (Qdrant point ids are the SQLite chunk ids, so upserts are
idempotent).

`semantic_search` is the minimal query -> chunk path used to verify P2.
"""

from __future__ import annotations

from typing import Any, Callable

from app.db import (
    fetch_chunk_contents,
    fetch_chunks_for_embedding,
    set_chunks_embedding_version,
)

ProgressFn = Callable[[int, int | None], None]


def build_embeddings(
    embedder: Any,
    store: Any,
    *,
    version: str | None = None,
    limit: int | None = None,
    doc_type: str | None = None,
    scan_batch: int = 256,
    recreate: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Embed pending chunks and index them. Returns a run summary."""
    version = version or embedder.version
    store.ensure_collection(recreate=recreate)

    processed = 0
    batches = 0
    last_id = 0
    dim_checked = False

    while True:
        remaining = None if limit is None else max(0, limit - processed)
        if remaining == 0:
            break
        take = scan_batch if remaining is None else min(scan_batch, remaining)
        rows = fetch_chunks_for_embedding(
            version, limit=take, doc_type=doc_type, after_id=last_id
        )
        if not rows:
            break

        vectors = embedder.encode([row["content"] for row in rows])
        if not dim_checked:
            if vectors and len(vectors[0]) != store.dim:
                raise ValueError(
                    f"embedding dimension mismatch: model produced {len(vectors[0])} "
                    f"but the collection expects {store.dim}; set EMBEDDING_DIM"
                )
            dim_checked = True

        payloads = [{**row, "embedding_version": version} for row in rows]
        store.upsert_chunks([row["id"] for row in rows], vectors, payloads)
        set_chunks_embedding_version([row["id"] for row in rows], version)

        processed += len(rows)
        batches += 1
        last_id = max(int(row["id"]) for row in rows)
        if progress:
            progress(processed, limit)

    return {
        "version": version,
        "processed": processed,
        "batches": batches,
        "indexed_points": store.count(),
    }


def semantic_search(
    embedder: Any,
    store: Any,
    query: str,
    *,
    top_k: int = 5,
    stock_code: str | None = None,
    doc_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and return the most similar chunks with their metadata.

    Qdrant holds metadata only; chunk text is hydrated from SQLite (the single
    source of truth) so the two stores cannot drift.
    """
    vector = embedder.encode_one(query)
    if len(vector) != store.dim:
        raise ValueError(
            f"embedding dimension mismatch: model produced {len(vector)} "
            f"but the collection expects {store.dim}; set EMBEDDING_DIM"
        )
    hits = store.search(
        vector, top_k=top_k, stock_code=stock_code, doc_type=doc_type,
        date_from=date_from, date_to=date_to,
    )
    contents = fetch_chunk_contents([int(hit["point_id"]) for hit in hits])
    for hit in hits:
        hit["content"] = contents.get(int(hit["point_id"]), "")
    return hits
