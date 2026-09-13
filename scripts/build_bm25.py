"""Build the BM25 lexical index over the chunk corpus (Phase 3).

Tokenization is jieba-based and takes a few minutes for ~110k chunks, so the
index (sparse matrix + chunk ids + metadata) is persisted under `BM25_PATH`
and reused by search/eval/API without re-tokenizing.

Usage:
    python scripts/build_bm25.py                      # all chunks that have vectors
    python scripts/build_bm25.py --all-chunks         # every chunk in SQLite
    python scripts/build_bm25.py --out data/bm25_dev  # custom location
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.db import fetch_chunks_for_bm25
from app.rag.bm25 import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BM25 index")
    parser.add_argument("--out", default=None, help=f"output dir (default: {Config.BM25_PATH})")
    parser.add_argument("--version", default=None,
                        help="only chunks embedded with this version (default: EMBEDDING_VERSION)")
    parser.add_argument("--all-chunks", action="store_true",
                        help="index every chunk regardless of embedding_version")
    args = parser.parse_args()

    version = None if args.all_chunks else (args.version or Config.EMBEDDING_VERSION)
    out_dir = Path(args.out or Config.BM25_PATH)

    print(f"loading chunks from SQLite (embedding_version={version or 'any'})")
    rows = fetch_chunks_for_bm25(version)
    if not rows:
        print("no chunks found — run ingestion/build_embeddings first", file=sys.stderr)
        raise SystemExit(2)
    print(f"chunks: {len(rows)}")

    t0 = time.time()
    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] < 5:
            return
        last[0] = now
        rate = done / max(now - t0, 1e-9)
        print(f"  tokenized {done}/{total} ({rate:.0f}/s, {now - t0:.0f}s)")

    index = BM25Index.build(
        rows, embedding_version=version, all_chunks=args.all_chunks, progress=progress
    )
    target = index.save(out_dir)

    print("")
    print(f"done in {time.time() - t0:.1f}s | docs={len(index)} | saved to {target}")
    print(f"manifest: {target / 'manifest.json'} "
          f"(chunk_id_hash={index.manifest()['chunk_id_hash']})")
    print("next: python scripts/search_hybrid.py \"<query>\"  (dense + bm25 + rrf + rerank)")


if __name__ == "__main__":
    main()
