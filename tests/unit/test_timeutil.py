"""Contract tests for market-day (UTC+8) time semantics.

These pin the *semantics* rather than the implementation: the fixed conversion
values, the rule that an offset-aware value is never shifted a second time, the
exact `end_of_day` contract, the injected-clock behaviour of `market_today()`,
the cninfo millisecond -> Beijing date rule, and the requirement that the dense
epoch window and the BM25 string window agree on one date boundary. A
source-level guard at the end keeps the forbidden patterns out of production code.

Corpus shape these tests rely on (measured 2026-09-19 over 109,554 chunks): every
`publish_time` is either `YYYY-MM-DD` (101,811) or `YYYY-MM-DD HH:MM:SS` (7,743),
none is empty and none carries a UTC offset. That is why a string-date window and
an epoch window can be required to agree exactly -- an offset-bearing value would
legitimately differ, because BM25 compares the first ten characters while dense
compares the absolute instant.
"""

from __future__ import annotations

import ast
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.rag.bm25 import BM25Index
from app.rag.sources import _cninfo_ts_to_date
from app.rag.vector_store import ChunkVectorStore
from app.timeutil import MARKET_TZ, STAMP_FORMAT, market_today, now_stamp, to_market_epoch

REPO_ROOT = Path(__file__).resolve().parents[2]

# The boundary day used throughout: 2026-04-29 is the A-share annual-report peak,
# the day with the most date-only (CST midnight) documents in the corpus.
DAY = "2026-04-29"
DAY_START = 1777392000  # 2026-04-29T00:00:00+08:00
DAY_END = 1777478399  # 2026-04-29T23:59:59+08:00
UTC_MIDNIGHT_OF_DAY = 1777420800  # the wrong boundary a UTC process computed


# --------------------------------------------------------------------------- #
# 1. fixed conversion values (the ruler's calibration)
# --------------------------------------------------------------------------- #
def test_fixed_conversion_values() -> None:
    """The three reference values must never drift."""
    assert to_market_epoch("2026-08-15") == 1786723200
    assert to_market_epoch("2026-08-15", end_of_day=True) == 1786809599
    assert to_market_epoch("2026-08-15 00:30:00") == 1786725000
    # a date object and its string form are the same market day
    assert to_market_epoch(date(2026, 8, 15)) == to_market_epoch("2026-08-15")


def test_unparsable_value_fails_loudly() -> None:
    """A bad boundary must raise, not silently widen or narrow the window."""
    for value in ("", None, "not-a-date", "15/08/2026"):
        with pytest.raises((ValueError, TypeError)):
            to_market_epoch(value)


# --------------------------------------------------------------------------- #
# 2. an offset-aware value is never shifted twice
# --------------------------------------------------------------------------- #
def test_aware_datetime_is_converted_for_its_absolute_instant() -> None:
    """Same instant, three ways of writing it -> one epoch. No extra +8 shift."""
    utc_midnight = datetime(2026, 8, 15, tzinfo=timezone.utc)
    beijing_midnight = datetime(2026, 8, 15, tzinfo=MARKET_TZ)

    assert to_market_epoch(utc_midnight) == 1786752000
    assert to_market_epoch(beijing_midnight) == 1786723200
    # the two differ by exactly the zone offset: proof that nothing was added
    assert to_market_epoch(utc_midnight) - to_market_epoch(beijing_midnight) == 28800


@pytest.mark.parametrize(
    "text",
    [
        "2026-08-15T00:00:00+08:00",
        "2026-08-14T16:00:00+00:00",
        "2026-08-14T12:00:00-04:00",
    ],
)
def test_the_same_instant_written_in_any_zone_gives_one_epoch(text: str) -> None:
    """2026-08-15T00:00:00+08:00 is the same instant however it is spelled."""
    assert to_market_epoch(text) == 1786723200


def test_naive_values_are_read_as_market_time() -> None:
    """A naive string keeps its Beijing reading (that is the whole point)."""
    assert to_market_epoch("2026-04-29") == DAY_START
    assert to_market_epoch("2026-04-29 00:00:00") == DAY_START
    assert to_market_epoch(datetime(2026, 4, 29)) == DAY_START


# --------------------------------------------------------------------------- #
# 3. end_of_day contract
# --------------------------------------------------------------------------- #
def test_end_of_day_covers_the_whole_market_day() -> None:
    start = to_market_epoch(DAY)
    end = to_market_epoch(DAY, end_of_day=True)

    assert start == DAY_START
    assert end == DAY_END
    # the contract is 23:59:59 of that day (not 24:00:00, not midnight next day)
    assert end == to_market_epoch(f"{DAY} 23:59:59")
    assert end - start == 86399


def test_end_of_day_does_not_extend_a_value_that_already_has_a_time() -> None:
    """Documented trap: the flag only applies to a date-only value."""
    assert to_market_epoch(f"{DAY} 10:00:00", end_of_day=True) == to_market_epoch(f"{DAY} 10:00:00")


def test_day_window_is_closed_at_both_ends() -> None:
    """Everything inside the market day is inside [gte, lte]; the next day is not."""
    gte, lte = DAY_START, DAY_END
    for inside in (f"{DAY} 00:00:00", f"{DAY} 00:30:00", f"{DAY} 16:00:00", f"{DAY} 23:59:59"):
        assert gte <= to_market_epoch(inside) <= lte, inside
    assert to_market_epoch("2026-04-28 23:59:59") < gte
    assert to_market_epoch("2026-04-30") > lte


# --------------------------------------------------------------------------- #
# 4. market_today() from an injected instant (no wall clock, no faketime)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "utc_now,expected",
    [
        # Beijing is UTC+8, so the Beijing date rolls over at 16:00 UTC.
        (datetime(2026, 8, 15, 15, 59, 59, tzinfo=timezone.utc), date(2026, 8, 15)),
        (datetime(2026, 8, 15, 16, 0, 0, tzinfo=timezone.utc), date(2026, 8, 16)),
        (datetime(2026, 8, 15, 23, 30, 0, tzinfo=timezone.utc), date(2026, 8, 16)),
        (datetime(2026, 8, 15, 0, 0, 0, tzinfo=timezone.utc), date(2026, 8, 15)),
    ],
)
def test_market_today_follows_the_beijing_calendar(utc_now: datetime, expected: date) -> None:
    assert market_today(now=utc_now) == expected


def test_market_today_accepts_beijing_and_naive_instants() -> None:
    aware = datetime(2026, 8, 16, 0, 30, tzinfo=MARKET_TZ)
    assert market_today(now=aware) == date(2026, 8, 16)
    # naive is read as Beijing time, not as the host's local time
    assert market_today(now=datetime(2026, 8, 15, 23, 59, 59)) == date(2026, 8, 15)
    assert market_today(now=datetime(2026, 8, 16, 0, 0, 0)) == date(2026, 8, 16)


def test_days_window_is_measured_in_market_days() -> None:
    """`days=30` must produce the same date_from whatever the process timezone."""
    anchor = datetime(2026, 8, 16, 3, 0, tzinfo=MARKET_TZ)  # 2026-08-15 19:00 UTC
    expected = (date(2026, 8, 16) - timedelta(days=30)).isoformat()
    assert (market_today(now=anchor) - timedelta(days=30)).isoformat() == expected


# --------------------------------------------------------------------------- #
# 5. cninfo millisecond timestamp -> Beijing date
# --------------------------------------------------------------------------- #
def _ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> int:
    """cninfo announcementTime: Unix milliseconds of a Beijing wall-clock moment."""
    moment = datetime(year, month, day, hour, minute, second, tzinfo=MARKET_TZ)
    return int(moment.timestamp() * 1000)


@pytest.mark.parametrize(
    "ms_value,expected",
    [
        # the case the old process-local conversion got wrong in a UTC container:
        # 2026-08-15 00:30 Beijing == 2026-08-14 16:30 UTC
        (_ms(2026, 8, 15, 0, 30), "2026-08-15"),
        (_ms(2026, 8, 15, 0, 0, 0), "2026-08-15"),          # exactly midnight
        (_ms(2026, 8, 14, 23, 59, 59), "2026-08-14"),        # one second earlier
        (_ms(2026, 8, 15, 7, 59, 59), "2026-08-15"),         # just before 08:00
        (_ms(2026, 8, 15, 8, 0, 0), "2026-08-15"),           # just after 08:00
        (_ms(2026, 8, 15, 16, 0, 0), "2026-08-15"),          # the announcement hour
        (_ms(2026, 8, 15, 23, 59, 59), "2026-08-15"),        # last second of the day
    ],
)
def test_cninfo_timestamp_maps_to_the_beijing_date(ms_value: int, expected: str) -> None:
    assert _cninfo_ts_to_date(ms_value) == expected


def test_cninfo_timestamp_is_independent_of_the_instant_it_is_viewed_from() -> None:
    """The same millisecond value always yields the same Beijing date."""
    ms_value = _ms(2026, 8, 15, 0, 30)
    assert to_market_epoch("2026-08-15 00:30:00") == ms_value // 1000
    assert _cninfo_ts_to_date(ms_value) == "2026-08-15"


def test_cninfo_non_numeric_value_is_passed_through() -> None:
    assert _cninfo_ts_to_date("2026-08-15 00:30:00") == "2026-08-15"
    assert _cninfo_ts_to_date(None) == ""


# --------------------------------------------------------------------------- #
# 6. dense and BM25 must share one date boundary
# --------------------------------------------------------------------------- #
class _StubRetriever:
    """Returns every document best-first, so the candidate window is never the limit."""

    def __init__(self, count: int) -> None:
        self._count = count

    def retrieve(self, queries, k, sorted=True, show_progress=False):  # noqa: A002 - bm25s API
        n = min(int(k), self._count)
        # bm25s returns one row per query: results[i] is the ranked doc indices.
        return [list(range(n))], [[float(n - index) for index in range(n)]]


#: Values around the boundary day, in the two shapes the corpus actually uses.
PROBE_TIMES = [
    "2026-04-28",
    "2026-04-28 23:59:59",
    "2026-04-29",
    "2026-04-29 00:00:00",
    "2026-04-29 00:30:00",
    "2026-04-29 07:59:59",
    "2026-04-29 08:00:00",
    "2026-04-29 16:00:00",
    "2026-04-29 23:59:59",
    "2026-04-30",
    "2026-04-30 00:00:00",
]


def _dense_window(date_from: str, date_to: str | None) -> tuple[float, float]:
    flt = ChunkVectorStore._build_filter(None, None, date_from=date_from, date_to=date_to)
    condition = next(c for c in flt.must if getattr(c, "key", None) == "publish_ts")
    return condition.range.gte, condition.range.lte


def _kept_by_dense(date_from: str, date_to: str | None) -> set[str]:
    gte, lte = _dense_window(date_from, date_to)
    kept = set()
    for index, publish_time in enumerate(PROBE_TIMES):
        epoch = to_market_epoch(publish_time)
        if (gte is None or epoch >= gte) and (lte is None or epoch <= lte):
            kept.add(f"c{index}")
    return kept


def _kept_by_bm25(date_from: str, date_to: str | None) -> set[str]:
    metadatas = [
        {"chunk_id": f"c{index}", "publish_time": publish_time,
         "stock_code": "000725", "doc_type": "announcement"}
        for index, publish_time in enumerate(PROBE_TIMES)
    ]
    index = BM25Index(
        _StubRetriever(len(metadatas)),
        [meta["chunk_id"] for meta in metadatas],
        metadatas,
    )
    hits, diagnostics = index.search_with_diagnostics(
        "公告", top_k=len(metadatas), date_from=date_from, date_to=date_to
    )
    assert diagnostics["truncated"] is False, (
        "BM25_WINDOW truncation would weaken this equivalence test"
    )
    return {hit["chunk_id"] for hit in hits}


@pytest.mark.parametrize(
    "date_from,date_to",
    [
        (DAY, DAY),                     # the explicit single-day window
        (DAY, None),                    # the shape `days=` produces (lower bound only)
        ("2026-04-28", DAY),            # a two-day window
        ("2026-04-29", "2026-04-30"),
    ],
)
def test_dense_and_bm25_agree_on_the_same_date_boundary(date_from: str, date_to: str | None) -> None:
    """One date boundary, two retrievers: the kept sets must be identical."""
    assert _kept_by_dense(date_from, date_to) == _kept_by_bm25(date_from, date_to)


def test_the_boundary_day_midnight_document_is_inside_the_window() -> None:
    """The exact defect: a date-only (CST midnight) document on the boundary day.

    The old code put the boundary at 08:00 Beijing in a UTC process, so this
    document fell 28,800 seconds short of the window.
    """
    gte, lte = _dense_window(DAY, DAY)
    assert gte == DAY_START
    assert lte == DAY_END
    assert gte != UTC_MIDNIGHT_OF_DAY, "the old UTC boundary must not come back"

    midnight_doc = "2026-04-29"          # date-only -> 2026-04-29T00:00:00+08:00
    assert to_market_epoch(midnight_doc) == gte
    assert "c2" in _kept_by_dense(DAY, DAY)      # index 2 == midnight_doc
    assert "c2" in _kept_by_bm25(DAY, DAY)


def test_bm25_window_would_still_agree_when_it_is_the_only_bound() -> None:
    """A lower bound alone must not shift either retriever's interpretation."""
    kept_dense = _kept_by_dense(DAY, None)
    kept_bm25 = _kept_by_bm25(DAY, None)
    assert kept_dense == kept_bm25
    assert kept_bm25 == {f"c{index}" for index in range(2, len(PROBE_TIMES))}


# --------------------------------------------------------------------------- #
# 7. source guard: forbidden patterns must stay out of production code
# --------------------------------------------------------------------------- #
# These inspect the syntax tree rather than the text, so an explanatory docstring
# that names a forbidden pattern (the auditor's anti-tautology note, this file's
# own comments) is not mistaken for a use of it.
def _python_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(sorted((REPO_ROOT / root).rglob("*.py")))
    return files


def _attribute_calls(path: Path):
    """Yield (call node, attribute name, receiver name) for every `x.y(...)` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            name = receiver.id if isinstance(receiver, ast.Name) else None
            yield node, node.func.attr, name


def test_no_process_local_time_in_production_code() -> None:
    """Process-local time must not be used for anything in app/.

    `datetime.now(tz)` with an explicit zone is fine -- the target is the
    zone-less reading: `date.today()`, `datetime.now()`, `datetime.utcnow()`.
    """
    offenders: list[str] = []
    for path in _python_files("app"):
        for node, attr, receiver in _attribute_calls(path):
            where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            if attr == "today" and receiver == "date":
                offenders.append(f"{where}: date.today()")
            elif attr == "utcnow" and receiver == "datetime":
                offenders.append(f"{where}: datetime.utcnow()")
            elif attr == "now" and receiver == "datetime":
                if not node.args and not node.keywords:
                    offenders.append(f"{where}: datetime.now() without a timezone")
    assert not offenders, (
        "process-local time in production code (use app.timeutil: market_today() / "
        f"now_stamp()): {offenders}"
    )


def test_epoch_construction_is_centralized() -> None:
    """Only app/timeutil.py may turn a datetime into an epoch.

    A second `datetime.timestamp()` conversion elsewhere is a second time
    semantics that can drift away from the market zone.
    """
    offenders: list[str] = []
    for path in _python_files("app"):
        if path.name == "timeutil.py":
            continue
        for node, attr, _receiver in _attribute_calls(path):
            if attr == "timestamp":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"epoch conversion outside app/timeutil.py: {offenders}"


def test_the_auditor_never_rebuilds_expectations_from_stored_values() -> None:
    """The anti-tautology rule: the expectation comes from the raw DB string.

    Deriving it from the stored publish_ts (`fromtimestamp(stored)`) would make
    every comparison trivially pass and hide the very defect the scan looks for.
    """
    scanner = REPO_ROOT / "scripts" / "scan_publish_ts.py"
    offenders = [
        f"{scanner.name}:{node.lineno}: fromtimestamp({ast.unparse(node.args[0])})"
        for node, attr, _receiver in _attribute_calls(scanner)
        if attr == "fromtimestamp"
        and node.args
        and re.match(r"^(stored|publish_ts|payload|point)", ast.unparse(node.args[0]))
    ]
    assert not offenders, (
        f"the expected epoch must not be derived from a stored value: {offenders}"
    )
    # positive control: the conversion really does go through app.timeutil
    assert "to_market_epoch(" in scanner.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 8. the audit stamp keeps its legacy shape
# --------------------------------------------------------------------------- #
def test_now_stamp_is_beijing_wall_clock_in_the_legacy_format() -> None:
    assert STAMP_FORMAT == "%Y-%m-%d %H:%M:%S"
    stamp = now_stamp()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stamp)
    # within a minute of the market-zone wall clock, and never the UTC reading
    # unless Beijing happens to equal it (they differ by 8 hours)
    parsed = datetime.strptime(stamp, STAMP_FORMAT)
    delta = abs((datetime.now(MARKET_TZ).replace(tzinfo=None) - parsed).total_seconds())
    assert delta < 120
