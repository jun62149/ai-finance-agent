"""Copy the chunk vectors from Qdrant local mode to a Qdrant server.

Phase 2 indexes into Qdrant **local (embedded) mode** (`QDRANT_PATH`), which is
not recommended beyond ~20k points and only allows one process at a time. Before
starting the API/Agent (concurrent reads), move the collection to the Docker
Qdrant service — vectors are copied as-is, **the embedding model is not rerun**.

Usage:
    # 1) start the server (see docker/docker-compose.yml)
    docker compose -f docker/docker-compose.yml up -d qdrant

    # 2) copy (dry run first)
    python scripts/migrate_qdrant.py --to-url http://localhost:6333 --dry-run
    python scripts/migrate_qdrant.py --to-url http://localhost:6333

    # 3) point the app at the server for good (or export per-shell)
    #    QDRANT_URL=http://localhost:6333

The copy is idempotent: point ids are the SQLite chunk ids, so re-running only
overwrites the same points.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.rag.migrate import copy_points
from app.rag.vector_store import ChunkVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy vectors local Qdrant -> Qdrant server")
    parser.add_argument("--from-path", default=None,
                        help=f"source local path (default: {Config.QDRANT_PATH})")
    parser.add_argument("--to-url", default=None,
                        help="target Qdrant URL, e.g. http://localhost:6333")
    parser.add_argument("--collection", default=None,
                        help=f"collection name (default: {Config.QDRANT_COLLECTION})")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recreate", action="store_true",
                        help="delete the target collection first")
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts only, copy nothing")
    args = parser.parse_args()

    to_url = args.to_url or Config.QDRANT_URL
    if not to_url:
        print("no target: pass --to-url http://localhost:6333 or set QDRANT_URL",
              file=sys.stderr)
        raise SystemExit(2)

    collection = args.collection or Config.QDRANT_COLLECTION
    source_path = str(args.from_path or Config.QDRANT_PATH)

    with ChunkVectorStore(path=source_path, collection=collection) as source:
        total = source.count()
        print(f"source: path={source_path} collection={collection} points={total}")
        if args.dry_run:
            print("dry run — nothing copied.")
            return
        if total == 0:
            print("source is empty — build embeddings first "
                  "(scripts/build_embeddings.py)", file=sys.stderr)
            raise SystemExit(3)

        t0 = time.time()
        last = [0.0]

        def progress(copied: int, expected: int) -> None:
            now = time.time()
            if now - last[0] < 5 and copied < expected:
                return
            last[0] = now
            rate = copied / max(now - t0, 1e-9)
            print(f"  copied {copied}/{expected} points "
                  f"({rate:.0f}/s, {now - t0:.0f}s)")

        with ChunkVectorStore(url=to_url, collection=collection) as target:
            summary = copy_points(
                source,
                target,
                batch_size=args.batch_size,
                recreate=args.recreate,
                expected_total=total,
                progress=progress,
            )

    print("")
    print(
        f"done in {time.time() - t0:.1f}s | copied={summary['copied']} "
        f"source_total={summary['source_total']} target_total={summary['target_total']}"
    )
    if summary["target_total"] == summary["source_total"]:
        print("[ok] point counts match — switch with QDRANT_URL and rerun a search.")
    else:
        print("[!] counts differ — rerun the command (idempotent) or check the server.")


if __name__ == "__main__":
    main()
