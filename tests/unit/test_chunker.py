from __future__ import annotations

from app.rag.chunker import chunk_text


def test_chunk_text_respects_size_limits() -> None:
    text = "\n\n".join(f"第{i}段：" + "内容" * 120 for i in range(30))
    chunks = chunk_text(text, max_chars=600, min_chars=100)
    assert chunks
    assert all(len(c.content) <= 600 for c in chunks)
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))
    assert len({c.chunk_hash for c in chunks}) == len(chunks)


def test_chunk_text_keeps_paragraphs_intact_when_short() -> None:
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = chunk_text(text, max_chars=600, min_chars=100)
    assert len(chunks) == 1
    assert "第一段。" in chunks[0].content
    assert "第三段。" in chunks[0].content


def test_chunk_text_does_not_break_long_numeric_segment() -> None:
    row = " ".join(str(i) for i in range(200))  # a long table-like row
    chunks = chunk_text(row, max_chars=100, min_chars=10)
    assert chunks
    for chunk in chunks:
        numbers = [t for t in chunk.content.split() if t.isdigit()]
        assert numbers  # digits preserved
    assert all(len(c.content) <= 100 for c in chunks)


def test_chunk_text_empty_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
