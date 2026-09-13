"""Tests for moving a vector collection between Qdrant targets (local -> local).

The same code path is used for local -> Docker server; only the client target
differs, so a second local store keeps the test offline.
"""

from __future__ import annotations

import pytest

from app.rag.migrate import copy_points
from app.rag.vector_store import ChunkVectorStore

DIM = 4


def _store(tmp_path, name: str) -> ChunkVectorStore:
    return ChunkVectorStore(path=str(tmp_path / name), collection="chunks", dim=DIM)


@pytest.fixture()
def stores(tmp_path):
    source = _store(tmp_path, "src")
    target = _store(tmp_path, "dst")
    ids = [1, 2, 3]
    vectors = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
    payloads = [
        {"chunk_id": "c1", "stock_code": "600519", "doc_type": "announcement", "title": "A"},
        {"chunk_id": "c2", "stock_code": "000725", "doc_type": "announcement", "title": "B"},
        {"chunk_id": "c3", "stock_code": "000725", "doc_type": "news", "title": "C"},
    ]
    source.ensure_collection()
    source.upsert_chunks(ids, vectors, payloads)
    yield source, target
    source.close()
    target.close()


def test_copy_points_moves_everything(stores) -> None:
    source, target = stores

    summary = copy_points(source, target, batch_size=2)

    assert summary["copied"] == 3
    assert summary["target_total"] == 3
    assert summary["source_total"] == 3
    assert target.count() == 3


def test_copied_points_keep_vectors_and_payload(stores) -> None:
    source, target = stores
    copy_points(source, target)

    hits = target.search([1.0, 0, 0, 0], top_k=1)

    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["stock_code"] == "600519"
    assert hits[0]["title"] == "A"


def test_copy_is_idempotent(stores) -> None:
    source, target = stores
    copy_points(source, target)
    copy_points(source, target)

    assert target.count() == 3  # same point ids overwritten, not duplicated


def test_copy_supports_filters_on_target(stores) -> None:
    source, target = stores
    copy_points(source, target)

    hits = target.search([0, 1.0, 0, 0], top_k=5, doc_type="announcement")

    assert hits
    assert {h["doc_type"] for h in hits} == {"announcement"}


def test_copy_with_recreate_replaces_target(stores) -> None:
    source, target = stores
    target.ensure_collection()
    target.upsert_chunks([99], [[0, 0, 0, 1.0]], [{"chunk_id": "stale"}])

    copy_points(source, target, recreate=True)

    assert target.count() == 3  # stale point is gone


def test_ensure_collection_recreate_purges_local_mode(tmp_path) -> None:
    """Regression: local mode resurrects points on delete+create, so an
    explicit filter-delete must run on recreate."""
    store = _store(tmp_path, "recreate")
    try:
        store.ensure_collection()
        store.upsert_chunks([1], [[1.0, 0, 0, 0]], [{"chunk_id": "c1"}])
        assert store.count() == 1

        store.ensure_collection(recreate=True)

        assert store.count() == 0
        assert list(store.scroll_points()) == []
    finally:
        store.close()
