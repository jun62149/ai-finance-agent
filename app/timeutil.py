"""Market-day time semantics (UTC+8) — the single source of truth for A-class time.

Two classes of timestamps live in this project and they must never be confused:

**A-class — semantic time.** Decides retrieval filtering and data attribution:
the `days` window, the `date_from`/`date_to` boundaries, the `publish_ts`
payload derived from `publish_time`, the cninfo millisecond timestamp ->
publication date conversion, and the ingest/update fetch windows. These are
**A-share market calendar days**, computed in UTC+8. `to_market_epoch()` and
`market_today()` are the only places allowed to turn such a value into an
instant or a date.

**B-class — audit time.** Only written down for display and tracing:
`documents/chunks.created_at`, the BM25 manifest `built_at`, the `system.log`
line stamps. It **never** participates in retrieval filtering, sorting or data
attribution decisions — `now_stamp()` is the single source for it.

The zone is a fixed ``timezone(timedelta(hours=8))`` rather than
``ZoneInfo("Asia/Shanghai")`` on purpose: mainland China has no DST, so a fixed
offset is equivalent, and it needs no ``tzdata`` package — which is absent on
Windows and in slim container images, where a named zone either raises
``ZoneInfoNotFoundError`` or silently falls back to UTC.

A market calendar day is **not** a trading day: exchange holidays are not
modelled here. ``market_today()`` is the Beijing calendar date, nothing more.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

MARKET_TZ = timezone(timedelta(hours=8))

#: Format of every B-class audit stamp (matches the pre-existing DB/log format).
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def market_today(now: datetime | None = None) -> date:
    """The Beijing calendar date for `now` (default: the current instant).

    `now` is a seam for tests: pass an aware or naive datetime to pin the
    instant instead of depending on the wall clock. A naive value is read as
    UTC+8.
    """
    if now is None:
        return datetime.now(MARKET_TZ).date()
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    return now.astimezone(MARKET_TZ).date()


def to_market_epoch(value: Any, *, end_of_day: bool = False) -> int:
    """Epoch seconds for a date/datetime string read in the market zone (UTC+8).

    Accepts ``YYYY-MM-DD`` (a market day; ``end_of_day=True`` extends a
    date-only value to 23:59:59) and ISO datetimes. A value that already carries
    an offset is respected as-is — it is never shifted a second time. A
    date-only value with ``end_of_day`` in mind must be passed as a date:
    for a value that already carries a time the flag has no effect.

    Raises ``ValueError`` when the value is empty or unparsable, so a bad
    boundary fails loudly instead of silently widening or narrowing a window.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty datetime value")

    if len(text) <= 10:
        moment = datetime.strptime(text, "%Y-%m-%d")
        if end_of_day:
            moment = datetime.combine(moment.date(), time(23, 59, 59))
    else:
        moment = datetime.fromisoformat(text)

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=MARKET_TZ)
    return int(moment.timestamp())


def now_stamp() -> str:
    """B-class audit stamp: Beijing wall clock, no offset, for records/logs only."""
    return datetime.now(MARKET_TZ).strftime(STAMP_FORMAT)
