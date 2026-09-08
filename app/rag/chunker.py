"""Semantic-boundary chunking for financial text (design doc section 7).

Rules:
- prefer paragraph / sentence boundaries over fixed-character splits;
- never split a numeric table row in the middle unless it exceeds the hard
  limit (then fall back to whitespace windows so digits stay intact);
- keep paragraph content verbatim (numbers/units preserved).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")


@dataclass
class Chunk:
    content: str
    chunk_index: int
    chunk_hash: str


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hard_split_segment(segment: str, max_chars: int) -> list[str]:
    """Split one over-long segment at whitespace windows (keeps numbers whole)."""
    if len(segment) <= max_chars:
        return [segment]
    pieces: list[str] = []
    rest = segment
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = window.rfind(" ")
        if cut < max_chars * 0.5:
            cut = max_chars
        pieces.append(window[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split one paragraph into sentence-grouped units <= max_chars."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    units: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                units.append(current)
                current = ""
            units.extend(_hard_split_segment(sentence, max_chars))
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            units.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        units.append(current)
    return units


def chunk_text(
    text: str,
    *,
    max_chars: int = 600,
    min_chars: int = 100,
) -> list[Chunk]:
    """Chunk cleaned text on paragraph/sentence boundaries."""
    if not text:
        return []
    paragraphs = [
        p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()
    ]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        content = "\n\n".join(buffer)
        chunks.append(
            Chunk(content=content, chunk_index=len(chunks), chunk_hash=_hash_text(content))
        )
        buffer = []
        buffer_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            for unit in _split_paragraph(paragraph, max_chars):
                chunks.append(
                    Chunk(
                        content=unit,
                        chunk_index=len(chunks),
                        chunk_hash=_hash_text(unit),
                    )
                )
            continue
        if buffer and buffer_len + 2 + len(paragraph) > max_chars and buffer_len >= min_chars:
            flush()
        buffer.append(paragraph)
        buffer_len += len(paragraph) + (2 if len(buffer) > 1 else 0)
    flush()
    return chunks
