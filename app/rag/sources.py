"""Financial text source adapters (news / announcements).

Phase 1 uses East Money (the same public source wrapped by AkShare):
- per-stock news via search-api (paginated);
- per-stock announcement list via AkShare `stock_individual_notice_report`;
- announcement body via the East Money notice content endpoint.

The adapter keeps the same return contract so additional sources
(cninfo, exchange websites, ...) can be added without touching ingestion.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from app.rag.cleaner import clean_text, clean_title


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)
NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
ANN_CONTENT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
ART_CODE_RE = re.compile(r"(AN\d+)")


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Referer": "https://data.eastmoney.com/"},
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
