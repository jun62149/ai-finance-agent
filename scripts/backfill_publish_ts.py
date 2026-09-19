"""Backfill the `publish_ts` payload field (epoch seconds) in Qdrant.

Why: Qdrant can only range-filter numeric/datetime fields, and the payload
stores `publish_time` as a string. The recency filter (`days` in the RAG tool,
and any date window) therefore needs `publish_ts`. Vectors are untouched — this
only updates payloads in place, so no re-embedding is involved.

The epoch is computed by `publish_ts_from` (app.timeutil, fixed UTC+8), so a value
written from a UTC container matches one written from a UTC+8 host; this script no
longer carries its own date parser.

Run it once after migrating/creating a collection, and again only if new points
were written by an older code path.

Usage:
    python scripts/backfill_publish_ts.py                 # configured store
    python scripts/backfill_publish_ts.py --dry-run
    python scripts/backfill_publish_ts.py --batch-size 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.rag.vector_store import ChunkVectorStore, publish_ts_from


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill publish_ts payloads")
    parser.add_argument("--batch-size", type=int, default=256, help="operations per request")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from qdrant_client.models import (
        SetPayload,
        SetPayloadOperation,
    )

    print(
        f"store={'url:' + Config.QDRANT_URL if Config.QDRANT_URL else 'path:' + str(Config.QDRANT_PATH)} "
        f"collection={Config.QDRANT_COLLECTION}"
    )

    with ChunkVectorStore() as store:
        total = store.count()
        print(f"points={total}")
        store.ensure_payload_indexes()

        scanned = 0
        updated = 0
        missing_time = 0
        already = 0
        operations: list = []
        started = time.time()

        for ids, _vectors, payloads in store.scroll_points(batch_size=args.batch_size):
            for point_id, payload in zip(ids, payloads):
                scanned += 1
                if payload.get("publish_ts") is not None:
                    already += 1
                    continue
                epoch = publish_ts_from(payload.get("publish_time"))
                if epoch is None:
                    missing_time += 1
                    continue
                operations.append(
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload={"publish_ts": epoch}, points=[int(point_id)]
                        )
                    )
                )
            if operations and (len(operations) >= args.batch_size or args.limit):
                if not args.dry_run:
                    store.client.batch_update_points(
                        store.collection, update_operations=operations, wait=True
                    )
                updated += len(operations)
                operations = []
                print(f"  updated {updated} points ({time.time() - started:.0f}s)", flush=True)
            if args.limit and scanned >= args.limit:
                break

        if operations and not args.dry_run:
            store.client.batch_update_points(
                store.collection, update_operations=operations, wait=True
            )
            updated += len(operations)

        print("")
        print(
            f"scanned={scanned} updated={updated} already_had_publish_ts={already} "
            f"unparsable_publish_time={missing_time} in {time.time() - started:.1f}s"
            + (" (dry run)" if args.dry_run else "")
        )
        if not args.dry_run:
            sample = store.search([1.0] + [0.0] * (store.dim - 1), top_k=1)
            if sample:
                print(f"sample payload has publish_ts: {sample[0].get('publish_ts')}")


if __name__ == "__main__":
    main()
