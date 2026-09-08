"""Text cleaning for financial news / announcements.

Goals (design doc section 7):
- strip HTML tags/scripts/styles and decode entities;
- unify whitespace, full/half-width punctuation and broken encodings;
- keep digits, units and financial numbers intact (do NOT "clean" them away).
"""

from __future__ import annotations

import html
import re


_BLOCK_RE = re.compile(r"<(script|style|iframe|noscript)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\u00a0\u3000]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+(?=[，。；：！？、,;:!?」』）】》％%])")
_SPACE_AFTER_OPEN_RE = re.compile(r"(?<=[（【「《『])\s+")


def _fullwidth_to_halfwidth(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def strip_html(raw: str) -> str:
    """Remove script/style blocks and tags, then unescape entities."""
    text = _BLOCK_RE.sub(" ", raw or "")
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def normalize_space(text: str) -> str:
    """Collapse horizontal whitespace, keep paragraph breaks, strip edges."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _fullwidth_to_halfwidth(text)
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _SPACE_BEFORE_PUNCT_RE.sub("", text)
    text = _SPACE_AFTER_OPEN_RE.sub("", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(raw: str, *, max_length: int = 200_000) -> str:
    """End-to-end cleaning for article/announcement bodies."""
    text = strip_html(raw)
    text = normalize_space(text)
    if len(text) > max_length:
        text = text[:max_length]
    return text


def clean_title(raw: str) -> str:
    return normalize_space(strip_html(raw))[:300]
