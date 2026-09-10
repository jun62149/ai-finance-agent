"""Remove ingested documents of a given type/source from the knowledge base.

Purpose: when switching an announcement source (eastmoney -> cninfo) the same
announcement can exist under two different titles, so the resume/skip key
(code, title, publish_time) would not match and the corpus would hold two
copies. Purging the old source keeps one canonical copy per announcement.

Usage:
    python scripts/purge_documents.py --doc-type announcement --source eastmoney            # dry run
    python scripts/purge_documents.py --doc-type announcement --source eastmoney --yes      # delete
    python scripts/purge_documents.py --doc-type announcement --yes                         # all sources
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, purge_documents  # noqa: E402


def _count(doc_type: str, source: str | None) -> tuple[int, int]:
    where = "doc_type = ?" + (" AND source = ?" if source else "")
    params: tuple = (doc_type, source) if source else (doc_type,)
    with connect() as conn:
        docs = conn.execute(
            f"SELECT COUNT(*) AS n FROM documents WHERE {where}", params
        ).fetchone()["n"]
        chunks = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE document_id IN "
            f"(SELECT document_id FROM documents WHERE {where})",
            params,
        ).fetchone()["n"]
    return int(docs), int(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge documents/chunks by type and source")
    parser.add_argument("--doc-type", default="announcement",
                        help="documents.doc_type to purge (default: announcement)")
    parser.add_argument("--source", default=None,
                        help="documents.source to purge, e.g. eastmoney (default: all sources)")
    parser.add_argument("--yes", action="store_true",
                        help="actually delete; without it this is a dry run")
    args = parser.parse_args()

    docs, chunks = _count(args.doc_type, args.source)
    scope = f"doc_type={args.doc_type}" + (f" source={args.source}" if args.source else " (all sources)")
    print(f"scope: {scope}")
    print(f"matched: {docs} documents / {chunks} chunks")

    if not args.yes:
        print("dry run — nothing deleted. Re-run with --yes to delete.")
        return

    deleted_docs, deleted_chunks = purge_documents(args.doc_type, args.source)
    print(f"deleted: {deleted_docs} documents / {deleted_chunks} chunks")


if __name__ == "__main__":
    main()
