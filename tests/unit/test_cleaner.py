from __future__ import annotations

from app.rag.cleaner import clean_text, clean_title, normalize_space, strip_html


def test_strip_html_keeps_financial_numbers() -> None:
    raw = "<html><body><script>var x=1;</script>营收 <b>1,234.56</b> 亿元</body></html>"
    text = strip_html(raw)
    assert "1,234.56" in text
    assert "<b>" not in text
    assert "var x" not in text


def test_clean_text_normalizes_fullwidth_and_space() -> None:
    raw = "　贵州茅台：\n\n　　2026\u3000半年度净利润\uFF04445.17 亿元\n\n\nnext段"
    text = clean_text(raw)
    assert "2026 半年度净利润$445.17 亿元" in text
    assert "\n\n" in text
    assert "\n\n\n" not in text


def test_normalize_space_keeps_paragraph_breaks() -> None:
    text = normalize_space("第一段 结束。\n\n第二段开始。")
    assert text == "第一段 结束。\n\n第二段开始。"


def test_clean_title_trims_and_strips_tags() -> None:
    assert clean_title("  <em>贵州茅台</em>: 公告 ") == "贵州茅台: 公告"
