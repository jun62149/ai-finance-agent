"""Query the chunk vector store (Phase 2, P2 minimal chain).

Usage:
    python scripts/search_chunks.py "茅台 半年度 营收 增长"
    python scripts/search_chunks.py "回购 进展" --stock-code 000725 --doc-type announcement
    python scripts/search_chunks.py "分红 派息" --top-k 8 --device cpu

Prints score + traceable metadata (stock / type / date / source / url) so the
result can be cited directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config
from app.rag.embedding import Embedder
from app.rag.indexing import semantic_search
from app.rag.vector_store import ChunkVectorStore


def _preview(text: str, width: int = 220) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[:width] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over chunks")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stock-code", default=None)
    parser.add_argument("--doc-type", default=None, choices=["news", "announcement"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--preview", type=int, default=220)
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model, device=args.device, version=args.version)
    with ChunkVectorStore(dim=embedder.dim) as store:
        hits = semantic_search(
            embedder,
            store,
            args.query,
            top_k=args.top_k,
            stock_code=args.stock_code,
            doc_type=args.doc_type,
        )

    print(
        f"query={args.query!r} top_k={args.top_k} "
        f"stock_code={args.stock_code} doc_type={args.doc_type} hits={len(hits)}"
    )
    for rank, hit in enumerate(hits, start=1):
        print("")
        print(f"[{rank}] score={hit.get('score', 0):.4f} {hit.get('stock_code')} "
              f"{hit.get('stock_name')} | {hit.get('doc_type')} | {hit.get('publish_time')}")
        print(f"    title : {hit.get('title')}")
        print(f"    source: {hit.get('source')} | chunk_id={hit.get('chunk_id')}")
        print(f"    url   : {hit.get('url')}")
        print(f"    text  : {_preview(str(hit.get('content') or ''), args.preview)}")
    if not hits:
        print("no hits — is the collection built? run scripts/build_embeddings.py")


if __name__ == "__main__":
    main()
