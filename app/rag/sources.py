"""Financial text source adapters (news / announcements).

Sources:
- news: East Money search-api (paginated, same public source AkShare wraps);
- announcements (default): **cninfo / 巨潮资讯网**, the official disclosure
  platform — list via `new/hisAnnouncement/query`, body via the disclosure
  PDF (`static.cninfo.com.cn`) parsed with pypdf;
- announcements (legacy/fallback): East Money notice API
  (`fetch_announcement_list` + `fetch_announcement_body`), kept because East
  Money applies connection-level IP throttling to its notice host.

The adapters share one return contract (list rows + `(title, content)` bodies)
so a new source can be added without touching the ingestion pipeline.
"""

from __future__ import annotations

import io
import json
import re
import time
from datetime import datetime
from typing import Any

import httpx

from app.rag.cleaner import clean_text, clean_title
from app.timeutil import MARKET_TZ


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)
NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
ANN_CONTENT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
ART_CODE_RE = re.compile(r"(AN\d+)")

# cninfo (巨潮资讯网, official disclosure platform)
CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_ORGID_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_STATIC_BASE = "http://static.cninfo.com.cn/"
CNINFO_REFERER = "https://www.cninfo.com.cn/new/disclosure"
# The query endpoint serves at most 30 rows per page; larger pageSize values
# make the server ignore pageNum (measured 2026-09-10).
CNINFO_MAX_PAGE_SIZE = 30

# orgId is heterogeneous (gssh0600519 / 9900005965 / GD165627) and cannot be
# derived from the code — resolve it from cninfo's own stock list once per run.
_CNINFO_ORGID_CACHE: dict[str, str] = {}


def _client(referer: str = "https://data.eastmoney.com/") -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Referer": referer},
        timeout=20.0,
        follow_redirects=True,
    )


def _jsonp_load(text: str) -> dict:
    start, end = text.find("("), text.rfind(")")
    if start < 0 or end <= start:
        raise ValueError("unexpected JSONP payload")
    return json.loads(text[start + 1 : end])


def _safe_get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - retry then report upstream
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_error}") from last_error


def _safe_post(
    client: httpx.Client, url: str, data: dict[str, str]
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.post(url, data=data)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - retry then report upstream
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed: {last_error}") from last_error


def fetch_news_pages(
    symbol: str,
    *,
    page_size: int = 50,
    max_pages: int = 6,
    min_publish_date: str | None = None,
) -> list[dict[str, Any]]:
    """Paginated East Money news for one stock code."""
    items: list[dict[str, Any]] = []
    with _client() as client:
        for page_index in range(1, max_pages + 1):
            inner_param = {
                "uid": "",
                "keyword": symbol,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "searchScope": "default",
                        "sort": "default",
                        "pageIndex": page_index,
                        "pageSize": page_size,
                        "preTag": "",
                        "postTag": "",
                    }
                },
            }
            params = {
                "cb": "cb_query",
                "param": json.dumps(inner_param, ensure_ascii=False),
                "_": str(int(time.time() * 1000)),
            }
            resp = _safe_get(client, NEWS_URL, params)
            payload = _jsonp_load(resp.text)
            result = (payload.get("result") or {}).get("cmsArticleWebOld") or []
            page_dates = []
            for row in result:
                publish = str(row.get("date") or "").strip()
                page_dates.append(publish)
                code = str(row.get("code") or "")
                items.append(
                    {
                        "title": clean_title(row.get("title") or ""),
                        "content": clean_text(row.get("content") or ""),
                        "publish_time": publish,
                        "media": str(row.get("mediaName") or "").strip(),
                        "url": f"http://finance.eastmoney.com/a/{code}.html" if code else "",
                    }
                )
            if len(result) < page_size:
                break
            if min_publish_date and page_dates and all(
                d < min_publish_date for d in page_dates if d
            ):
                break
            time.sleep(0.15)
    return items


def fetch_announcement_list(
    code: str,
    begin_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """Announcement metadata rows for one stock code (East Money via AkShare)."""
    import akshare as ak

    begin = begin_date.replace("-", "")
    end = end_date.replace("-", "")
    df = ak.stock_individual_notice_report(
        security=code,
        symbol="全部",
        begin_date=begin,
        end_date=end,
    )
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        url = str(row.get("网址") or "").strip()
        art = ART_CODE_RE.search(url)
        title = clean_title(str(row.get("公告标题") or ""))
        publish = str(row.get("公告日期") or "").strip()
        if not art or not title:
            continue
        rows.append(
            {
                "code": str(row.get("代码") or code).zfill(6),
                "name": str(row.get("名称") or "").strip(),
                "title": title,
                "category": str(row.get("公告类型") or "").strip(),
                "publish_time": publish,
                "url": url,
                "art_code": art.group(1),
            }
        )
    rows.sort(key=lambda r: r["publish_time"], reverse=True)
    return rows


def fetch_announcement_body(art_code: str) -> tuple[str, str] | None:
    """Fetch announcement title + cleaned body. Returns None when body is empty."""
    with _client() as client:
        resp = _safe_get(
            client,
            ANN_CONTENT_URL,
            {"art_code": art_code, "client_source": "web", "page_index": "1"},
        )
    data = (resp.json() or {}).get("data") or {}
    title = clean_title(data.get("notice_title") or "")
    content = clean_text(data.get("notice_content") or "")
    if len(content) < 20:
        return None
    return title, content


# --------------------------------------------------------------------------- #
# cninfo (巨潮资讯网) — official disclosure platform, announcement source
# --------------------------------------------------------------------------- #


def _cninfo_org_id(code: str) -> str | None:
    """Resolve the cninfo orgId for a stock code (cached per process)."""
    global _CNINFO_ORGID_CACHE
    if not _CNINFO_ORGID_CACHE:
        with _client(CNINFO_REFERER) as client:
            resp = _safe_get(client, CNINFO_ORGID_URL, {})
        payload = resp.json() or {}
        _CNINFO_ORGID_CACHE = {
            str(item.get("code")): str(item.get("orgId"))
            for item in payload.get("stockList") or []
            if item.get("code") and item.get("orgId")
        }
    return _CNINFO_ORGID_CACHE.get(code)


def _cninfo_query(
    code: str,
    org_id: str,
    begin_date: str,
    end_date: str,
    page_num: int,
    page_size: int,
) -> dict[str, Any]:
    """One page of cninfo's announcement query (seam for tests)."""
    payload = {
        "stock": f"{code},{org_id}",
        "tabName": "fulltext",
        "pageSize": str(page_size),
        "pageNum": str(page_num),
        "column": "",
        "category": "",
        "plate": "",
        "seDate": f"{begin_date}~{end_date}",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    headers = {
        "User-Agent": UA,
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": CNINFO_REFERER,
        "Origin": "https://www.cninfo.com.cn",
    }
    with httpx.Client(headers=headers, timeout=25.0, follow_redirects=True) as client:
        resp = _safe_post(client, CNINFO_QUERY_URL, payload)
    return resp.json() or {}


def _cninfo_ts_to_date(value: Any) -> str:
    """cninfo announcementTime is a Unix millisecond timestamp.

    Interpreted in the market zone (UTC+8): converting it in the process-local
    timezone would move an early-morning announcement to the previous day in a
    UTC container.
    """
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=MARKET_TZ).strftime("%Y-%m-%d")
    return str(value or "")[:10]


def fetch_cninfo_announcement_list(
    code: str,
    begin_date: str,
    end_date: str,
    *,
    page_size: int = CNINFO_MAX_PAGE_SIZE,
    max_pages: int = 60,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Announcement metadata rows for one stock from cninfo (official).

    Returns rows shaped like the East Money adapter's, plus a direct PDF
    ``url`` (static.cninfo.com.cn) used to fetch the body.

    Pagination note: the endpoint serves at most ``CNINFO_MAX_PAGE_SIZE`` (30)
    rows per page and **ignores ``pageNum`` when a larger pageSize is sent**
    (measured 2026-09-10: pageSize=50 returned page 1 again for pageNum=2, so a
    client would silently see only the newest ~30 announcements). The size is
    therefore clamped, pages are de-duplicated by announcementId, and
    ``hasMore`` / ``totalRecordNum`` drive the stop condition — ``totalpages``
    under-reports and must not be trusted.
    """
    org_id = _cninfo_org_id(code)
    if not org_id:
        raise RuntimeError(f"cninfo orgId not found for stock {code}")

    page_size = max(1, min(int(page_size), CNINFO_MAX_PAGE_SIZE))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for page_num in range(1, max_pages + 1):
        data = _cninfo_query(code, org_id, begin_date, end_date, page_num, page_size)
        items = data.get("announcements") or []
        if not items:
            break
        fresh = 0
        for item in items:
            art_code = str(item.get("announcementId") or "")
            key = art_code or f"{item.get('announcementTitle')}|{item.get('announcementTime')}"
            if key in seen:
                continue
            seen.add(key)
            fresh += 1
            publish = _cninfo_ts_to_date(item.get("announcementTime"))
            title = clean_title(str(item.get("announcementTitle") or ""))
            adjunct = str(item.get("adjunctUrl") or "").strip()
            if not title or not publish or not (begin_date <= publish <= end_date):
                continue
            rows.append(
                {
                    "code": str(item.get("secCode") or code).zfill(6),
                    "name": str(item.get("secName") or "").strip(),
                    "title": title,
                    "category": str(item.get("announcementTypeName") or "").strip(),
                    "publish_time": publish,
                    "url": (
                        adjunct
                        if adjunct.startswith("http")
                        else f"{CNINFO_STATIC_BASE}{adjunct}"
                    ),
                    "art_code": art_code,
                    "adjunct_type": str(item.get("adjunctType") or "").upper(),
                }
            )
        if max_rows and len(rows) >= max_rows:
            break
        if fresh == 0:
            # the server ignored pageNum and repeated this page — stop cleanly
            break
        if not data.get("hasMore"):
            break
        total = int(data.get("totalRecordNum") or 0)
        if total and len(seen) >= total:
            break
        time.sleep(0.2)

    # newest first, matching the East Money adapter contract
    rows.sort(key=lambda r: r["publish_time"], reverse=True)
    return rows[:max_rows] if max_rows else rows


def _fetch_bytes(url: str) -> bytes:
    """Download a binary asset (announcement PDF). Seam for tests."""
    with _client(CNINFO_REFERER) as client:
        resp = _safe_get(client, url, {})
    return resp.content


def pdf_bytes_to_text(data: bytes) -> str:
    """Extract text from a PDF byte payload using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pypdf is required to parse cninfo announcement PDFs; "
            "install it with: pip install pypdf"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one broken page must not kill the doc
            continue
    return "\n".join(parts)


def fetch_cninfo_announcement_body(
    row: dict[str, Any],
) -> tuple[str, str] | None:
    """Fetch + extract one cninfo announcement body (PDF -> text).

    Returns None when the announcement has no usable body (no PDF link,
    unparsable PDF, or too little text).
    """
    url = str(row.get("url") or "").strip()
    if not url:
        return None
    if str(row.get("adjunct_type") or "PDF").upper() not in ("", "PDF"):
        return None
    data = _fetch_bytes(url)
    content = clean_text(pdf_bytes_to_text(data))
    if len(content) < 20:
        return None
    title = clean_title(str(row.get("title") or ""))
    return title, content
