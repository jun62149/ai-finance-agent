"""Move an existing vector collection between Qdrant targets.

Motivation: Phase 2 builds vectors in Qdrant **local (embedded) mode**, which
is only recommended up to ~20k points and holds a single-process file lock.
Before wiring the API/Agent (which needs concurrent access), the collection is
moved to a Qdrant server (Docker). Vectors are copied point-for-point, so the
embedding model never runs again.

Both the SQLite `chunks.embedding_version` stamps and the payload
`embedding_version` travel with the points, so the two stores stay consistent.
"""

from __future__ import annotations

from typing import Any, Callable

ProgressFn = Callable[[int, int], None]


def copy_points(
    source: Any,
    target: Any,
    *,
    batch_size: int = 256,
    recreate: bool = False,
    expected_total: int | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, int]:
    """Copy every point from `source` into `target`. Returns copy statistics."""
    target.ensure_collection(recreate=recreate)
    source_total = source.count() if expected_total is None else int(expected_total)

    copied = 0
    batches = 0
    for ids, vectors, payloads in source.scroll_points(batch_size=batch_size):
        target.upsert_chunks(ids, vectors, payloads, batch_size=batch_size)
        copied += len(ids)
        batches += 1
        if progress:
            progress(copied, source_total)

    return {
        "copied": copied,
        "batches": batches,
        "source_total": source_total,
        "target_total": target.count(),
    }
