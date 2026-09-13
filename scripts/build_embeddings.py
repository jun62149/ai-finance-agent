"""Build embeddings for chunks and index them into Qdrant (Phase 2, P2).

Resumable: only chunks without a vector for the current EMBEDDING_VERSION are
processed, and SQLite is marked only after a successful Qdrant upsert. Rerun
the same command to continue after an interruption; a completed run reports
0 pending.

Usage:
    python scripts/build_embeddings.py                       # all pending chunks
    python scripts/build_embeddings.py --limit 200           # smoke test
    python scripts/build_embeddings.py --doc-type announcement
    python scripts/build_embeddings.py --recreate            # rebuild collection
    python scripts/build_embeddings.py --device cpu --batch-size 8

Environment:
    EMBEDDING_MODEL=BAAI/bge-m3   EMBEDDING_DEVICE=cuda   EMBEDDING_BATCH_SIZE=16
    EMBEDDING_VERSION=bge-m3-v1   EMBEDDING_DIM=1024
    QDRANT_PATH=data/qdrant       (or QDRANT_URL=http://localhost:6333)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.db import embedding_stats
from app.rag.embedding import Embedder
from app.rag.indexing import build_embeddings
from app.rag.vector_store import ChunkVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks into Qdrant (resumable)")
    parser.add_argument("--limit", type=int, default=None, help="max chunks this run")
    parser.add_argument("--doc-type", default=None, help="only news / announcement")
    parser.add_argument("--batch-size", type=int, default=None, help="model batch size")
    parser.add_argument("--scan-batch", type=int, default=256, help="chunks per DB scan")
    parser.add_argument("--device", default=None, help="cuda / cpu / auto")
    parser.add_argument("--model", default=None, help="embedding model name")
    parser.add_argument("--version", default=None, help="EMBEDDING_VERSION to stamp")
    parser.add_argument("--recreate", action="store_true", help="drop + recreate collection")
    args = parser.parse_args()

    embedder = Embedder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        version=args.version,
    )
    version = embedder.version

    print(
        f"model={embedder.model_name} device={embedder.device} "
        f"batch_size={embedder.batch_size} version={version} dim={embedder.dim}"
    )
    print(f"qdrant={'url:' + Config.QDRANT_URL if Config.QDRANT_URL else 'path:' + str(Config.QDRANT_PATH)} "
          f"collection={Config.QDRANT_COLLECTION}")

    before = embedding_stats(version)
    print("pending before:", {k: v["pending"] for k, v in before.items()} or "{}")

    t0 = time.time()
    last_report = [0.0]

    def progress(processed: int, limit: int | None) -> None:
        now = time.time()
        if now - last_report[0] < 5 and (limit is None or processed < limit):
            return
        last_report[0] = now
        elapsed = now - t0
        rate = processed / elapsed if elapsed else 0.0
        target = f"/{limit}" if limit else ""
        print(f"  embedded {processed}{target} chunks in {elapsed:.0f}s ({rate:.1f}/s)")

    with ChunkVectorStore(dim=embedder.dim) as store:
        summary = build_embeddings(
            embedder,
            store,
            version=version,
            limit=args.limit,
            doc_type=args.doc_type,
            scan_batch=args.scan_batch,
            recreate=args.recreate,
            progress=progress,
        )

    stats = embedding_stats(version)
    print("")
    print(
        f"done in {time.time() - t0:.1f}s | processed={summary['processed']} "
        f"batches={summary['batches']} indexed_points={summary['indexed_points']}"
    )
    for doc_type, row in sorted(stats.items()):
        print(f"  {doc_type}: total={row['total']} embedded={row['embedded']} pending={row['pending']}")

    pending = sum(row["pending"] for row in stats.values())
    if pending == 0:
        print("[ok] every chunk has a vector for this version (rerun-idempotent).")
    else:
        print(f"[i] {pending} chunks still pending; rerun the same command to continue.")


if __name__ == "__main__":
    main()
