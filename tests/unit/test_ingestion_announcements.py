"""Unit tests for the resumable/resilient announcement ingestion pipeline.

Network access is fully mocked: fetch_announcement_list / fetch_announcement_body
are patched, so tests never touch East Money.
"""

from __future__ import annotations

import pytest

from app.db import insert_documents
from app.rag.ingestion import ingest_announcements_for_stocks

ROW1 = {
    "art_code": "AN111",
    "title": "既有公告标题",
    "publish_time": "2026-01-05",
    "url": "https://data.eastmoney.com/notices/detail/600001/AN111.html",
}
ROW2 = {
    "art_code": "AN222",
    "title": "新公告标题",
    "publish_time": "2026-02-01",
    "url": "https://data.eastmoney.com/notices/detail/600001/AN222.html",
}
BODY2 = ("新公告标题", "这是新公告的正文内容，长度足以通过最小长度校验。")


@pytest.fixture()
def stored_doc(db_env) -> None:
    """Pre-insert one announcement row exactly matching ROW1."""
    insert_documents(
        [
            {
                "document_id": "announcement-eastmoney-600001-aaaa",
                "stock_code": "600001",
                "stock_name": "测试股份A",
                "doc_type": "announcement",
                "title": ROW1["title"],
                "source": "eastmoney",
                "publish_time": ROW1["publish_time"],
                "content": "既有公告正文，已经入库的内容。",
                "url": ROW1["url"],
                "content_hash": "fake-hash-1",
                "created_at": "2026-01-05 09:00:00",
            }
        ]
    )


def test_skip_existing_and_ingest_only_new(db_env, stored_doc, monkeypatch) -> None:
    """Rerun behaviour: stored rows are skipped, only unseen bodies fetched."""
    from app.rag import ingestion as ing

    fetched: list[str] = []
    monkeypatch.setattr(ing, "fetch_announcement_list", lambda *a, **k: [ROW1, ROW2])
    monkeypatch.setattr(
        ing,
        "fetch_announcement_body",
        lambda art_code: fetched.append(art_code) or BODY2,
    )

    totals = ingest_announcements_for_stocks(
        [("600001", "测试股份A")],
        begin_date="2025-01-01",
        end_date="2026-12-31",
        max_bodies_per_code=100,
        workers=2,
        request_delay=0,
    )

    assert totals["skipped_existing"] == 1  # ROW1 matched the stored doc
    assert totals["docs"] == 1  # only ROW2 was new
    assert totals["errors"] == 0
    assert fetched == ["AN222"]  # body of the existing row was never fetched
    assert not totals["stopped_early"]


def test_stops_early_after_consecutive_fully_failed_stocks(db_env, monkeypatch) -> None:
    """When every body request fails for several stocks, the run stops."""
    from app.rag import ingestion as ing

    def always_fail(art_code: str):
        raise ConnectionError("Server disconnected without sending a response")

    monkeypatch.setattr(
        ing,
        "fetch_announcement_list",
        lambda *a, **k: [dict(ROW1, art_code="AN111")],
    )
    monkeypatch.setattr(ing, "fetch_announcement_body", always_fail)

    totals = ingest_announcements_for_stocks(
        [("600001", "A"), ("600002", "B"), ("600003", "C")],
        begin_date="2025-01-01",
        end_date="2026-12-31",
        max_bodies_per_code=100,
        workers=2,
        request_delay=0,
        max_consecutive_fail_stocks=2,
    )

    assert totals["stopped_early"] is True
    assert totals["stopped_reason"]
    assert len(totals["per_code"]) == 2  # stopped before touching the 3rd stock
    assert totals["errors"] == 2
    assert totals["docs"] == 0
    assert "disconnected" in totals["per_code"][0]["sample_error"]
