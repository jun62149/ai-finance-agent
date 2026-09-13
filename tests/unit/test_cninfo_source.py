"""Unit tests for the cninfo (巨潮资讯网) announcement source adapter.

Network is fully mocked: the cninfo query seam, the binary fetch seam and the
PDF text extractor are patched, so tests never touch cninfo or East Money.
"""

from __future__ import annotations

import pytest

from app.rag import sources as src
from app.rag.ingestion import _announcement_adapters, ingest_announcements_for_stocks

CODE = "600519"
ORG = "gssh0600519"


def _item(announcement_id: str, title: str, ms: int, adjunct: str, adjunct_type: str = "PDF"):
    return {
        "secCode": CODE,
        "secName": "贵州茅台",
        "announcementId": announcement_id,
        "announcementTitle": title,
        "announcementTime": ms,
        "adjunctUrl": adjunct,
        "adjunctType": adjunct_type,
        "announcementTypeName": "半年报",
    }


# cninfo announcementTime is a Unix millisecond timestamp (local tz)
def _ms(year: int, month: int, day: int) -> int:
    from datetime import datetime

    return int(datetime(year, month, day).timestamp() * 1000)


TS_A = _ms(2026, 8, 15)
TS_B = _ms(2026, 8, 16)


@pytest.fixture()
def cninfo_env(monkeypatch):
    monkeypatch.setattr(src, "_cninfo_org_id", lambda code: ORG)
    return monkeypatch


def test_list_filters_by_date_and_builds_pdf_url(cninfo_env) -> None:
    pages = {
        1: {
            "announcements": [
                _item("1", "半年度报告", TS_A, "finalpage/2026-08-15/1.PDF"),
                _item("2", "股东大会决议", TS_B, "finalpage/2026-08-16/2.PDF"),
            ],
            "hasMore": False,
        }
    }
    cninfo_env.setattr(
        src, "_cninfo_query", lambda code, org, b, e, page, size: pages[page]
    )

    rows = src.fetch_cninfo_announcement_list(
        CODE, "2026-08-15", "2026-08-15", page_size=50
    )

    assert len(rows) == 1  # TS_B row is outside the requested range
    assert rows[0]["title"] == "半年度报告"
    assert rows[0]["publish_time"] == "2026-08-15"
    assert rows[0]["url"] == "http://static.cninfo.com.cn/finalpage/2026-08-15/1.PDF"
    assert rows[0]["art_code"] == "1"


def test_list_paginates_and_sorts_newest_first(cninfo_env) -> None:
    pages = {
        1: {
            "announcements": [_item("1", "旧公告", TS_A, "finalpage/a.PDF")],
            "hasMore": True,
        },
        2: {
            "announcements": [_item("2", "新公告", TS_B, "finalpage/b.PDF")],
            "hasMore": False,
        },
    }
    calls: list[int] = []

    def fake_query(code, org, b, e, page, size):
        calls.append(page)
        return pages[page]

    cninfo_env.setattr(src, "_cninfo_query", fake_query)

    rows = src.fetch_cninfo_announcement_list(CODE, "2026-08-01", "2026-08-31")

    assert calls == [1, 2]  # followed hasMore
    assert [r["title"] for r in rows] == ["新公告", "旧公告"]  # newest first


def test_list_respects_max_rows(cninfo_env) -> None:
    # cninfo returns announcements newest-first; the cap keeps the newest rows
    page = {
        "announcements": [
            _item("2", "B", TS_B, "finalpage/b.PDF"),
            _item("1", "A", TS_A, "finalpage/a.PDF"),
        ],
        "hasMore": True,
    }
    cninfo_env.setattr(src, "_cninfo_query", lambda *a, **k: page)

    rows = src.fetch_cninfo_announcement_list(
        CODE, "2026-08-01", "2026-08-31", max_rows=1
    )

    assert len(rows) == 1
    assert rows[0]["title"] == "B"  # newest kept


def test_list_requires_resolvable_orgid(monkeypatch) -> None:
    monkeypatch.setattr(src, "_cninfo_org_id", lambda code: None)
    with pytest.raises(RuntimeError, match="orgId"):
        src.fetch_cninfo_announcement_list(CODE, "2026-08-01", "2026-08-31")


def test_list_clamps_page_size_to_server_max(cninfo_env) -> None:
    """pageSize > 30 makes cninfo ignore pageNum, so it must be clamped."""
    sizes: list[int] = []

    def fake_query(code, org, b, e, page, size):
        sizes.append(size)
        return {
            "announcements": [_item("1", "A", TS_A, "finalpage/a.PDF")],
            "hasMore": False,
        }

    cninfo_env.setattr(src, "_cninfo_query", fake_query)

    src.fetch_cninfo_announcement_list(CODE, "2026-08-01", "2026-08-31", page_size=50)

    assert sizes == [src.CNINFO_MAX_PAGE_SIZE]


def test_list_stops_when_server_repeats_a_page(cninfo_env) -> None:
    """A repeated page (pageNum ignored) must not duplicate rows or loop on."""
    page = {
        "announcements": [_item("1", "A", TS_A, "finalpage/a.PDF")],
        "hasMore": True,
    }
    calls: list[int] = []

    def fake_query(code, org, b, e, page_num, size):
        calls.append(page_num)
        return page

    cninfo_env.setattr(src, "_cninfo_query", fake_query)

    rows = src.fetch_cninfo_announcement_list(CODE, "2026-08-01", "2026-08-31")

    assert [r["art_code"] for r in rows] == ["1"]  # de-duplicated
    assert calls == [1, 2]  # stopped as soon as a page brought nothing new


def test_body_extracts_pdf_text(monkeypatch) -> None:
    monkeypatch.setattr(src, "_fetch_bytes", lambda url: b"%PDF-fake")
    monkeypatch.setattr(src, "pdf_bytes_to_text", lambda data: "公告正文" * 20)

    body = src.fetch_cninfo_announcement_body(
        {"title": "半年度报告", "url": "http://static.cninfo.com.cn/x.PDF", "adjunct_type": "PDF"}
    )

    assert body is not None
    title, content = body
    assert title == "半年度报告"
    assert "公告正文" in content


def test_body_skips_non_pdf_and_empty(monkeypatch) -> None:
    monkeypatch.setattr(src, "_fetch_bytes", lambda url: b"%PDF-fake")
    monkeypatch.setattr(src, "pdf_bytes_to_text", lambda data: "")

    assert src.fetch_cninfo_announcement_body(
        {"title": "t", "url": "http://x/y.DOC", "adjunct_type": "DOC"}
    ) is None
    assert src.fetch_cninfo_announcement_body({"title": "t", "url": ""}) is None
    # PDF link but no extractable text -> None (counted as "empty" by ingestion)
    assert src.fetch_cninfo_announcement_body(
        {"title": "t", "url": "http://x/y.PDF", "adjunct_type": "PDF"}
    ) is None


def test_unknown_source_rejected() -> None:
    with pytest.raises(ValueError, match="unknown announcement source"):
        _announcement_adapters("bloomberg")


def test_ingest_with_cninfo_source(db_env, monkeypatch) -> None:
    """The pipeline runs end-to-end against the cninfo adapter."""
    from app.rag import ingestion as ing

    rows = [
        {
            "art_code": "1",
            "title": "半年度报告",
            "publish_time": "2026-08-15",
            "url": "http://static.cninfo.com.cn/a.PDF",
            "adjunct_type": "PDF",
        }
    ]
    fetched: list[str] = []

    monkeypatch.setattr(ing, "fetch_cninfo_announcement_list", lambda *a, **k: rows)
    monkeypatch.setattr(
        ing,
        "fetch_cninfo_announcement_body",
        lambda row: fetched.append(row["art_code"])
        or (row["title"], "巨潮公告正文内容，长度足够通过最小校验。" * 3),
    )

    totals = ingest_announcements_for_stocks(
        [("600519", "贵州茅台")],
        begin_date="2026-08-01",
        end_date="2026-08-31",
        source="cninfo",
        max_bodies_per_code=10,
        workers=1,
        request_delay=0,
    )

    assert fetched == ["1"]
    assert totals["docs"] == 1
    assert totals["source"] == "cninfo"
    assert totals["chars"] > 0
    assert totals["errors"] == 0
