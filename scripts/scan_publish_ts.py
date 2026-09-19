"""Read-only auditor for the `publish_ts` payload (Step 5 pre-flight).

Two modes, both strictly read-only:

  MODE 1 (default)  full scan
      Compares every stored `publish_ts` against the epoch rebuilt from the
      **raw** publication string, and classifies the differences.
  MODE 2            --window D [--date-to D]
      Prints the day-window filter a production retrieval would build for that
      date, its boundary epochs, and the exact number of chunks it matches.
      Used as the before/after instrument for the UTC+8 boundary fix.

Read-only contract (audit by inspection, not by promise)
-------------------------------------------------------
* Qdrant: only ``count`` and ``scroll`` are ever called. No upsert, no
  set_payload, no delete, no index creation -- there is no write path in this
  file at all.
* SQLite: opened with ``mode=ro`` URI, so the database cannot be written even
  by accident.
* Nothing here is imported by production code; it is an audit instrument.

Anti-tautology rules (why the expectation cannot be self-fulfilling)
-------------------------------------------------------------------
R1  `expected` is built from the raw string only:
    ``expected_cst = to_market_epoch(raw)``. Deriving it from the stored
    `publish_ts` (e.g. ``fromtimestamp(stored)``) would make every comparison
    trivially pass and hide exactly the defect this scan exists to find.
R2  `raw` comes from the **DB** (`chunks.publish_time`), not from the Qdrant
    payload. The payload copy is cross-checked separately, and
    `documents.publish_time` is cross-checked against `chunks.publish_time`
    separately, so a divergence is reported instead of silently resolved.
R3  The ruler calibrates itself: the fixed conversion values are printed in the
    same output as the measurement. A scan whose calibration lines are missing
    or wrong must not be trusted.

Auxiliary check: for the date-only subset the modulo identity
``(stored + 28800) % 86400 == 0`` holds for a UTC+8 midnight without using any
of this project's conversion code. It is a cross-check only -- the primary
criterion stays DB raw string -> to_market_epoch -> compare with stored.

Usage:
    python scripts/scan_publish_ts.py
    python scripts/scan_publish_ts.py --label "host/pre-fix"
    python scripts/scan_publish_ts.py --window 2026-04-29
    python scripts/scan_publish_ts.py --window 2026-04-29 --date-to 2026-04-29

Exit code: 0 = no mismatch/missing, 2 = mismatch or missing found (report only;
this script never repairs anything), 1 = could not complete.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.rag.vector_store import ChunkVectorStore  # noqa: E402
from app.timeutil import MARKET_TZ, to_market_epoch  # noqa: E402

try:  # qdrant-client reports transport failures through this wrapper type
    from qdrant_client.http.exceptions import (  # noqa: E402
        ResponseHandlingException as _TransportError,
    )
except ImportError:  # pragma: no cover - older clients
    _TransportError = OSError

_TRANSIENT_ERRORS = (_TransportError, ConnectionError, TimeoutError)

BATCH_SIZE = 512
SAMPLE_LIMIT = 10
HISTOGRAM_TOP = 5

#: Docker Desktop on Windows restarts its engine on auto-update, which resets an
#: in-flight scroll (`ResponseHandlingException: [WinError 10054]`). Observed
#: 2026-09-19 mid-scan; retry the page instead of losing the whole run. Every
#: retry is printed, so the evidence says whether the measurement was disturbed.
SCROLL_RETRIES = 6
SCROLL_BACKOFF_SECONDS = 10.0

#: Progress line frequency (points). A long read must be observable: without it a
#: stall is indistinguishable from "still working".
PROGRESS_EVERY = 10000
#: A scroll page slower than this is reported individually with its page number.
SLOW_PAGE_SECONDS = 1.0

#: Qdrant API surface actually used by this script (for the audit trail).
QDRANT_APIS_USED = ("count", "scroll")

#: A UTC+8 midnight differs from a UTC midnight by exactly this many seconds.
TZ_SHIFT = 28800
DAY_SECONDS = 86400

#: Calibration inputs -> expected epochs (R3). Kept as literal pairs so the
#: expected values can never drift into "whatever the code happens to return".
CALIBRATION = (
    ("2026-08-15", {}, 1786723200),
    ("2026-08-15", {"end_of_day": True}, 1786809599),
    ("2026-08-15 00:30:00", {}, 1786725000),
    ("2026-08-15T00:00:00+00:00", {}, 1786752000),
)

PROVENANCE_FILES = (
    "app/timeutil.py",
    "app/rag/vector_store.py",
    "app/rag/retrieval.py",
    "scripts/scan_publish_ts.py",
)


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        head = out.stdout.strip()
        if not head:
            return "unknown (git unavailable)"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        return f"{head}{' +dirty' if dirty else ' (clean)'}"
    except (OSError, subprocess.SubprocessError):
        return "unknown (git unavailable)"


def _print_header(label: str) -> None:
    root = Path(__file__).resolve().parents[1]
    print("=" * 78)
    print(f"publish_ts read-only scan | label={label}")
    print("=" * 78)
    print(f"git_head           = {_git_head()}")
    print(f"run_at_utc         = {datetime.now(timezone.utc).isoformat()}")
    print(f"run_at_market(CST) = {datetime.now(MARKET_TZ).isoformat()}")
    # ASCII only: Windows reports a localized tz name that turns into mojibake
    # when piped through a console with a different codepage.
    _local = datetime.now().astimezone()
    _tzname = str(_local.tzname()).encode("ascii", "replace").decode("ascii")
    print(f"process_tz         = utc_offset={_local.utcoffset()} tzname={_tzname}")
    print(f"qdrant_url         = {Config.QDRANT_URL or '(unset -> LOCAL PATH MODE: ' + str(Config.QDRANT_PATH) + ')'}")
    print(f"collection         = {Config.QDRANT_COLLECTION}")
    print(f"db_path            = {Config.DB_PATH}")
    print(f"qdrant_apis_used   = {', '.join(QDRANT_APIS_USED)}  (no mutation API in this file)")
    print("file_sha256(first16):")
    for name in PROVENANCE_FILES:
        print(f"  {name:<32} {_sha256(root / name)}")


def _print_calibration() -> None:
    """R3: print the ruler's fixed values together with the measurement."""
    print("\n[ruler calibration] expected literal vs actual")
    ok = True
    for value, kwargs, expected in CALIBRATION:
        try:
            actual = to_market_epoch(value, **kwargs)
        except Exception as exc:  # noqa: BLE001 - report, never hide
            actual = f"ERROR {exc!r}"
        match = actual == expected
        ok = ok and match
        print(f"  {value:<28} {kwargs!s:<22} expected={expected} actual={actual} "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"  calibration_ok = {ok}"
          + ("" if ok else "   <-- DO NOT TRUST THIS SCAN"))


# --------------------------------------------------------------------------- #
# data access (read-only)
# --------------------------------------------------------------------------- #
def _load_db_rows() -> tuple[dict[str, dict[str, Any]], int, int, int]:
    """chunk_id -> row from SQLite, opened read-only. Plus chunk/doc counts."""
    uri = f"file:{Config.DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows: dict[str, dict[str, Any]] = {}
        for row in conn.execute(
            "SELECT c.chunk_id, c.document_id, c.publish_time AS chunk_pt,"
            "       c.doc_type, c.source, d.publish_time AS doc_pt"
            "  FROM chunks c LEFT JOIN documents d ON d.document_id = c.document_id"
        ):
            rows[str(row["chunk_id"])] = {
                "document_id": row["document_id"],
                "raw": row["chunk_pt"],
                "doc_pt": row["doc_pt"],
                "doc_type": row["doc_type"],
                "source": row["source"],
            }
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_doc_time_mismatch = conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN documents d"
            " ON d.document_id = c.document_id"
            " WHERE IFNULL(c.publish_time,'') <> IFNULL(d.publish_time,'')"
        ).fetchone()[0]
        return rows, int(n_chunks), int(n_docs), int(n_doc_time_mismatch)
    finally:
        conn.close()


def _iter_points(store: ChunkVectorStore, stats: dict[str, int]):
    """Stream every point as (point_id, payload). Payload only, no vectors.

    A page is retried on a transient transport failure (the engine can restart
    underneath us); every retry is counted in `stats` and printed, so the evidence
    records that the measurement was disturbed rather than hiding it.
    """
    offset: Any = None
    attempt = 0
    page = 0
    while True:
        page_started = time.perf_counter()
        try:
            points, offset = store.client.scroll(
                store.collection,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            attempt = 0
        except _TRANSIENT_ERRORS as exc:
            attempt += 1
            stats["scroll_retries"] += 1
            if attempt > SCROLL_RETRIES:
                print(f"[scroll] giving up after {SCROLL_RETRIES} retries: "
                      f"{type(exc).__name__}: {str(exc)[:140]}")
                raise
            print(f"[scroll] transient {type(exc).__name__} "
                  f"(attempt {attempt}/{SCROLL_RETRIES}), retrying in "
                  f"{SCROLL_BACKOFF_SECONDS:.0f}s: {str(exc)[:100]}", flush=True)
            time.sleep(SCROLL_BACKOFF_SECONDS)
            continue
        if not points:
            return
        page_seconds = time.perf_counter() - page_started
        if page_seconds > SLOW_PAGE_SECONDS:
            print(f"[scroll] page {page} took {page_seconds:.2f}s for "
                  f"{len(points)} points", flush=True)
        page += 1
        for point in points:
            yield int(point.id), dict(point.payload or {})
        # The final page still carries points but reports offset=None, and a
        # scroll with offset=None starts again from the beginning -- so terminate
        # *after* yielding that page. (Only checking `not points` loops forever:
        # found 2026-09-19 after a scan reached 1.28M points on a 109,554-point
        # collection.)
        if offset is None:
            return


# --------------------------------------------------------------------------- #
# MODE 1: full scan
# --------------------------------------------------------------------------- #
def scan_full(label: str) -> int:
    _print_header(label)
    _print_calibration()

    _db_started = time.perf_counter()
    db_rows, n_chunks, n_docs, n_doc_time_mismatch = _load_db_rows()
    print(f"\n[corpus] db_chunks={n_chunks}  db_documents={n_docs}  "
          f"db_rows_loaded={len(db_rows)}  ({time.perf_counter()-_db_started:.1f}s)")

    store = ChunkVectorStore()
    try:
        points_count = int(store.client.count(store.collection, exact=True).count)
        print(f"[corpus] qdrant_points={points_count}")

        classes: Counter[str] = Counter()
        deltas: Counter[int] = Counter()
        samples: list[dict[str, Any]] = []
        date_only = Counter()
        scanned_ids: set[str] = set()
        payload_vs_db_mismatch = 0
        payload_vs_db_samples: list[tuple[str, str, str]] = []
        stats: dict[str, int] = {"scroll_retries": 0}

        scanned = 0
        overflow = False
        _loop_started = time.perf_counter()
        for point_id, payload in _iter_points(store, stats):
            scanned += 1
            if scanned > points_count:
                print(f"\n[scan] ABORT: the walk passed the {points_count} points the "
                      f"store reports, so it is not terminating -- this run is INVALID",
                      flush=True)
                overflow = True
                break
            if scanned % PROGRESS_EVERY == 0:
                _elapsed = time.perf_counter() - _loop_started
                print(f"[scan] {scanned} points  {_elapsed:.1f}s  "
                      f"{scanned / max(_elapsed, 1e-9):,.0f} points/s", flush=True)
            chunk_id = str(payload.get("chunk_id") or f"<point:{point_id}>")
            scanned_ids.add(chunk_id)
            row = db_rows.get(chunk_id)
            stored = payload.get("publish_ts")
            payload_pt = payload.get("publish_time")

            if row is None:
                classes["point_without_db_row"] += 1
                continue

            raw = row["raw"]
            if payload_pt is not None and raw is not None and str(payload_pt) != str(raw):
                payload_vs_db_mismatch += 1
                if len(payload_vs_db_samples) < 5:
                    payload_vs_db_samples.append((chunk_id, str(raw), str(payload_pt)))

            if stored is None:
                classes["missing_publish_ts"] += 1
                continue

            try:
                expected_cst = to_market_epoch(raw)
            except (ValueError, TypeError):
                classes["unparsable_raw"] += 1
                if len(samples) < SAMPLE_LIMIT:
                    samples.append({
                        "chunk_id": chunk_id, "raw": repr(raw), "stored": stored,
                        "expected_cst": None, "delta": None,
                        "doc_type": row["doc_type"], "source": row["source"],
                    })
                continue

            delta = int(stored) - expected_cst
            if delta == 0:
                classes["ok"] += 1
            elif delta == -TZ_SHIFT:
                classes["utc_written"] += 1
                deltas[delta] += 1
                if len(samples) < SAMPLE_LIMIT:
                    samples.append({
                        "chunk_id": chunk_id, "raw": str(raw), "stored": stored,
                        "expected_cst": expected_cst, "delta": delta,
                        "doc_type": row["doc_type"], "source": row["source"],
                    })
            else:
                classes["other"] += 1
                deltas[delta] += 1
                if len(samples) < SAMPLE_LIMIT:
                    samples.append({
                        "chunk_id": chunk_id, "raw": str(raw), "stored": stored,
                        "expected_cst": expected_cst, "delta": delta,
                        "doc_type": row["doc_type"], "source": row["source"],
                    })

            if isinstance(raw, str) and len(raw.strip()) == 10:
                if (int(stored) + TZ_SHIFT) % DAY_SECONDS == 0:
                    date_only["cst_midnight_ok"] += 1
                elif int(stored) % DAY_SECONDS == 0:
                    date_only["utc_midnight_signature"] += 1
                else:
                    date_only["modulo_other"] += 1
    finally:
        store.close()

    if overflow:
        print("\n[verdict]")
        print("  RUN INVALID: the point walk exceeded the reported point count, so the "
              "termination logic is broken and every total below would be wrong.")
        return 3

    db_only = set(db_rows) - scanned_ids
    mismatch = classes["utc_written"] + classes["other"]
    missing = classes["missing_publish_ts"] + classes["unparsable_raw"]

    print(f"\n[scan] points_scanned={scanned}")
    print("[totals]")
    for key in ("ok", "utc_written", "other", "missing_publish_ts",
                "unparsable_raw", "point_without_db_row"):
        print(f"  {key:<24} {classes[key]}")
    print(f"  {'db_chunk_without_point':<24} {len(db_only)}")

    print(f"\n[delta histogram  stored - expected_cst]  top {HISTOGRAM_TOP}")
    if deltas:
        for delta, count in deltas.most_common(HISTOGRAM_TOP):
            print(f"  delta={delta:+d}  count={count}")
    else:
        print("  (empty)")

    print("\n[payload.publish_time vs db chunks.publish_time]")
    print(f"  mismatches = {payload_vs_db_mismatch}")
    for chunk_id, raw, payload_pt in payload_vs_db_samples:
        print(f"    chunk_id={chunk_id} db={raw!r} payload={payload_pt!r}")

    print("\n[documents.publish_time vs chunks.publish_time]  (DB-only check)")
    print(f"  mismatches = {n_doc_time_mismatch}")

    print(f"\n[auxiliary modulo cross-check]  date-only subset (len(raw)==10)")
    print(f"  subset_size                = {sum(date_only.values())}")
    print(f"  cst_midnight_ok            = {date_only['cst_midnight_ok']}")
    print(f"  utc_midnight_signature     = {date_only['utc_midnight_signature']}")
    print(f"  modulo_other               = {date_only['modulo_other']}")

    print(f"\n[samples] up to {SAMPLE_LIMIT} classified points")
    if not samples:
        print("  (none)")
    for s in samples:
        print(f"  chunk_id={s['chunk_id']} raw={str(s['raw'])[:26]!r} "
              f"stored={s['stored']} expected_cst={s['expected_cst']} "
              f"delta={s['delta']} doc_type={s['doc_type']} source={s['source']}")

    print("\n[verdict]")
    print(f"  mismatch (utc_written + other) = {mismatch}")
    print(f"  missing  (missing_ts + unparsable) = {missing}")
    print(f"  payload_vs_db_mismatch = {payload_vs_db_mismatch}")
    print(f"  documents_vs_chunks_mismatch = {n_doc_time_mismatch}")
    print(f"  scroll_retries = {stats['scroll_retries']}"
          + ("" if stats["scroll_retries"] == 0
             else "   <-- the engine was disturbed during this run"))
    if mismatch == 0 and missing == 0:
        print("  RESULT: 0 inconsistent points -> no backfill indicated "
              "(report only; nothing was modified)")
        return 0
    print("  RESULT: inconsistencies found -> REPORT AND STOP. This script never "
          "backfills; backfill is a separate decision.")
    return 2


# --------------------------------------------------------------------------- #
# MODE 2: day-window probe
# --------------------------------------------------------------------------- #
def probe_window(label: str, date_from: str, date_to: str | None) -> int:
    _print_header(label)
    _print_calibration()

    print("\n[mode] WINDOW PROBE (not the full scan)")
    print(f"  date_from = {date_from}")
    print(f"  date_to   = {date_to}")
    print("  filter built by the production path: "
          "app.rag.vector_store.ChunkVectorStore._build_filter")

    flt = ChunkVectorStore._build_filter(
        None, None, date_from=date_from, date_to=date_to
    )
    conditions = list(getattr(flt, "must", []) or [])
    print(f"  filter_conditions = {[getattr(c, 'key', c) for c in conditions]}")

    gte = lte = None
    for condition in conditions:
        if getattr(condition, "key", None) == "publish_ts":
            rng = condition.range
            gte, lte = rng.gte, rng.lte

    def render(epoch: int | None) -> str:
        if epoch is None:
            return "(unbounded)"
        utc = datetime.fromtimestamp(epoch, timezone.utc)
        cst = datetime.fromtimestamp(epoch, MARKET_TZ)
        return f"{epoch}  utc={utc.isoformat()}  cst={cst.isoformat()}"

    print(f"\n[boundary epochs computed by production code]")
    print(f"  gte = {render(gte)}")
    print(f"  lte = {render(lte)}")

    store = ChunkVectorStore()
    try:
        total = int(store.client.count(store.collection, exact=True).count)
        hits = int(store.client.count(
            store.collection, count_filter=flt, exact=True
        ).count)
    finally:
        store.close()

    print(f"\n[result]")
    print(f"  collection_total   = {total}")
    print(f"  window_hits(exact) = {hits}")
    print("  note: exact count, no vectors fetched, no writes")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only publish_ts auditor")
    parser.add_argument("--window", metavar="YYYY-MM-DD",
                        help="MODE 2: probe the day window starting at this date")
    parser.add_argument("--date-to", metavar="YYYY-MM-DD",
                        help="MODE 2: optional upper bound (default: none)")
    parser.add_argument("--label", default="unlabelled",
                        help="provenance label printed in the header")
    args = parser.parse_args()

    if args.window:
        return probe_window(args.label, args.window, args.date_to)
    if args.date_to:
        print("--date-to requires --window", file=sys.stderr)
        return 1
    return scan_full(args.label)


if __name__ == "__main__":
    raise SystemExit(main())
