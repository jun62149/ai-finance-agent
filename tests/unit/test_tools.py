"""Unit tests for the Agent tools (P5 RAG tool + P6 quant tools + registry).

Network and models are avoided: the RAG tool is exercised with fake retrieval
components, and the quant tools run against the temporary SQLite fixture.
"""

from __future__ import annotations

import pytest

from app.errors import InvalidStockCode, StockNotFound
from app.tools import call_tool, get_tool, list_tools, tool_names, tool_schemas
from app.tools.errors import ToolArgumentError, ToolNotFound
from app.tools.rag_tools import search_financial_knowledge

from tests.conftest import STOCK_A, STOCK_NAMES


# ------------------------------------------------------------------- fakes --
class FakeEmbedder:
    version = "test-v1"
    dim = 4

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_one(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FakeStore:
    dim = 4

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, vector, top_k=5, stock_code=None, doc_type=None, date_from=None, date_to=None):
        self.calls.append(
            {"top_k": top_k, "stock_code": stock_code, "doc_type": doc_type,
             "date_from": date_from, "date_to": date_to}
        )
        return [
            {
                "chunk_id": "announcement-cninfo-000725-abc:c000",
                "document_id": "announcement-cninfo-000725-abc",
                "point_id": 1,
                "stock_code": "000725",
                "stock_name": "京东方A",
                "doc_type": "announcement",
                "title": "关于回购股份方案实施完毕暨回购实施结果的公告",
                "source": "cninfo",
                "publish_time": "2026-01-06",
                "url": "http://static.cninfo.com.cn/finalpage/2026-01-06/x.PDF",
                "chunk_index": 0,
                "content": "截至 2025 年 12 月 31 日，本次回购方案已实施完毕。",
                "score": 0.66,
            }
        ]


class FakeBM25:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query, top_k=20, stock_code=None, doc_type=None, date_from=None, date_to=None):
        self.calls.append({"query": query, "date_from": date_from, "date_to": date_to})
        return [
            {
                "chunk_id": "announcement-cninfo-000725-abc:c000",
                "document_id": "announcement-cninfo-000725-abc",
                "point_id": 1,
                "stock_code": "000725",
                "doc_type": "announcement",
                "bm25_score": 4.77,
            }
        ]


class FakeReranker:
    def rerank(self, query, candidates, top_k=None):
        ranked = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_score"] = 0.91
            ranked.append(item)
        return ranked[:top_k] if top_k else ranked, ""


@pytest.fixture()
def rag_components() -> dict:
    return {
        "embedder": FakeEmbedder(),
        "store": FakeStore(),
        "bm25_index": FakeBM25(),
        "reranker": FakeReranker(),
    }


# -------------------------------------------------------------- RAG tool ----
def test_search_tool_returns_evidence_with_metadata(db_env, rag_components) -> None:
    outcome = search_financial_knowledge(
        "回购 实施结果", stock_code="000725", doc_type="announcement",
        top_k=3, components=rag_components,
    )

    assert outcome["count"] == 1
    assert outcome["has_evidence"] is True
    evidence = outcome["results"][0]
    for field in ("chunk_id", "document_id", "stock_code", "doc_type",
                  "title", "source", "publish_time", "url"):
        assert evidence[field], f"missing traceable metadata: {field}"
    assert evidence["dense_score"] == pytest.approx(0.66)
    assert evidence["bm25_score"] == pytest.approx(4.77)
    assert evidence["rerank_score"] == pytest.approx(0.91)
    assert "rrf_score" in evidence
    assert outcome["retrieval"]["reranked"] is True
    assert evidence["citation"].startswith("关于回购股份方案实施完毕")


def test_search_tool_passes_recency_window(db_env, rag_components) -> None:
    """`days` must reach both retrievers as an explicit date window."""
    search_financial_knowledge("回购", days=30, components=rag_components)

    from datetime import date, timedelta

    expected_from = (date.today() - timedelta(days=30)).isoformat()
    assert rag_components["store"].calls[0]["date_from"] == expected_from
    assert rag_components["bm25_index"].calls[0]["date_from"] == expected_from


def test_search_tool_empty_result_is_explicit(db_env) -> None:
    class EmptyStore(FakeStore):
        def search(self, *args, **kwargs):
            return []

    class EmptyBM25(FakeBM25):
        def search(self, *args, **kwargs):
            return []

    outcome = search_financial_knowledge(
        "完全不相关的问题",
        components={
            "embedder": FakeEmbedder(), "store": EmptyStore(),
            "bm25_index": EmptyBM25(), "reranker": FakeReranker(),
        },
    )

    assert outcome["count"] == 0
    assert outcome["has_evidence"] is False
    assert outcome["results"] == []  # nothing is ever invented


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"query": ""}, "query"),
        ({"query": "x"}, "query"),
        ({"query": "回购", "top_k": 0}, "top_k"),
        ({"query": "回购", "top_k": 999}, "top_k"),
        ({"query": "回购", "doc_type": "video"}, "doc_type"),
        ({"query": "回购", "stock_code": "725"}, "stock_code"),
        ({"query": "回购", "days": 0}, "days"),
        ({"query": "回购", "days": 99999}, "days"),
    ],
)
def test_search_tool_validates_arguments(db_env, rag_components, kwargs, message) -> None:
    with pytest.raises(ToolArgumentError, match=message):
        search_financial_knowledge(components=rag_components, **kwargs)


def test_search_tool_rejects_evidence_without_metadata(db_env) -> None:
    from app.tools.errors import EvidenceMetadataMissing

    class NoMetadataStore(FakeStore):
        def search(self, *args, **kwargs):
            hits = super().search(*args, **kwargs)
            hits[0]["url"] = None  # break traceability
            return hits

    with pytest.raises(EvidenceMetadataMissing, match="url"):
        search_financial_knowledge(
            "回购",
            components={
                "embedder": FakeEmbedder(), "store": NoMetadataStore(),
                "bm25_index": FakeBM25(), "reranker": FakeReranker(),
            },
        )


# ------------------------------------------------------------ quant tools ---
def test_get_stock_profile_tool(populated_db) -> None:
    result = call_tool("get_stock_profile", code=STOCK_A)

    assert result["ok"] is True
    assert result["result"]["name"] == STOCK_NAMES[STOCK_A]
    assert result["result"]["has_factor_data"] is True


def test_get_stock_factors_tool(populated_db) -> None:
    result = call_tool("get_stock_factors", code=STOCK_A)

    assert result["ok"] is True
    factors = result["result"]
    assert factors["code"] == STOCK_A
    assert factors["as_of"]
    for field in ("ret_5d", "volatility_20d", "amount_mean_20d", "drawdown_20d"):
        assert field in factors
    assert "非投资建议" in factors["disclaimer"]
    assert "弱信号" in factors["model_note"]


def test_get_stock_factors_tool_historical_as_of(populated_db) -> None:
    result = call_tool("get_stock_factors", code=STOCK_A, as_of="2023-06-01")

    assert result["ok"] is True
    assert result["result"]["as_of"] <= "2023-06-01"


def test_quant_tool_errors_are_structured(populated_db) -> None:
    unknown = call_tool("get_stock_factors", code="999999")
    assert unknown["ok"] is False
    assert unknown["error"]["code"] == "StockNotFound"

    invalid = call_tool("get_stock_factors", code="abc")
    assert invalid["ok"] is False
    assert invalid["error"]["code"] == "InvalidStockCode"

    bad_as_of = call_tool("get_stock_factors", code=STOCK_A, as_of="not-a-date")
    assert bad_as_of["ok"] is False
    assert bad_as_of["error"]["code"] == "InvalidToolArgument"

    unknown_tool = call_tool("drop_database")
    assert unknown_tool["ok"] is False
    assert unknown_tool["error"]["code"] == "UnknownTool"


def test_quant_tool_errors_match_service_exceptions(populated_db) -> None:
    """The tool layer must not invent a different error vocabulary than the API."""
    from app.services import stock_service

    with pytest.raises(StockNotFound):
        stock_service.get_stock_basic("999999")
    with pytest.raises(InvalidStockCode):
        stock_service.validate_code("abc")


# ---------------------------------------------------------------- registry --
def test_registry_exposes_expected_tools() -> None:
    assert set(tool_names()) == {
        "search_financial_knowledge",
        "get_stock_profile",
        "get_stock_factors",
        "predict_stock",
        "rank_stocks",
    }
    assert len(list_tools()) == 5


def test_tool_schemas_are_function_calling_shaped() -> None:
    for schema in tool_schemas():
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        assert function["parameters"]["type"] == "object"
        assert "properties" in function["parameters"]


def test_no_dangerous_tools_are_registered() -> None:
    """Iron rule: the Agent gets no SQL / shell / filesystem capability."""
    forbidden_prefixes = ("sql", "shell", "bash", "exec", "run", "read_", "write_", "delete_")
    forbidden_substrings = ("_file", "file_", "read_file", "write_file")
    for name in tool_names():
        lowered = name.lower()
        assert not lowered.startswith(forbidden_prefixes), name
        assert not any(token in lowered for token in forbidden_substrings), name


def test_get_tool_unknown_raises() -> None:
    with pytest.raises(ToolNotFound, match="unknown tool"):
        get_tool("nope")


def test_call_tool_wrong_kwargs_returns_structured_error(populated_db) -> None:
    result = call_tool("get_stock_profile", wrong_argument="x")

    assert result["ok"] is False
    assert result["error"]["code"] == "InvalidToolArgument"
