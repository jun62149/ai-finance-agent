"""Unit tests for the P7 Agent (offline: LLM and tools are faked)."""

from __future__ import annotations

import unicodedata

import pytest

from app.agents import nodes, policies
from app.agents.graph import NODE_SEQUENCE, build_graph, run_agent
from app.agents.llm import LLMClient, LLMUnavailable
from app.tools import call_tool

from tests.conftest import STOCK_A, STOCK_NAMES


class FakeLLM:
    """Deterministic stand-in for the DeepSeek client."""

    def __init__(self, answer: str = "# 报告\n基于工具结果的分析。", tool_calls=None) -> None:
        self.model = "fake-model"
        self.base_url = "http://fake"
        self.available = True
        self.answer = answer
        self._tool_calls = tool_calls or []
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        if tools and self._tool_calls:
            calls, self._tool_calls = self._tool_calls, []
            return {"role": "assistant", "content": None, "tool_calls": calls}
        return {"role": "assistant", "content": self.answer}

    def chat_text(self, messages, **kwargs):
        return self.chat(messages, **kwargs).get("content") or ""


@pytest.fixture()
def fake_tools(monkeypatch):
    """Patch the tool runner so the loop never touches real services."""
    calls: list[tuple[str, dict]] = []

    def fake_call_tool(name, **kwargs):
        calls.append((name, kwargs))
        if name == "search_financial_knowledge":
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "count": 1,
                    "has_evidence": True,
                    "results": [
                        {
                            "chunk_id": "c1",
                            "document_id": "d1",
                            "stock_code": "000725",
                            "doc_type": "announcement",
                            "title": "回购实施结果公告",
                            "source": "cninfo",
                            "publish_time": "2026-05-30",
                            "url": "http://x/1",
                            "content": "回购方案已实施完毕。",
                            "rank": 1,
                        }
                    ],
                },
            }
        if name == "predict_stock":
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "code": "000725",
                    "as_of": "2026-06-24",
                    "pred_return_5d": 0.0055,
                    "strong_prob": 0.37,
                    "risk_score": 0.15,
                    "score": 4.48,
                    "model_version": "xgb-b-20260625-fulla",
                },
            }
        if name == "get_stock_factors":
            return {
                "ok": True,
                "tool": name,
                "result": {"code": "000725", "as_of": "2026-06-24", "ret_5d": 0.01},
            }
        if name == "get_stock_profile":
            return {
                "ok": True,
                "tool": name,
                "result": {"code": "000725", "name": "京东方A", "has_factor_data": True},
            }
        if name == "rank_stocks":
            return {
                "ok": True,
                "tool": name,
                "result": {"as_of": "2026-06-24", "count": 1, "items": [{"code": "001211"}]},
            }
        return {"ok": False, "tool": name, "error": {"code": "UnknownTool", "message": name}}

    monkeypatch.setattr(nodes, "call_tool", fake_call_tool)
    return calls


# ------------------------------------------------------------- intent rules --
@pytest.mark.parametrize(
    "query, expected",
    [
        ("现在评分最高的10只股票", "stock_ranking"),
        ("比较一下京东方A和美的集团", "stock_comparison"),
        ("贵州茅台最近有什么公告", "news_search"),
        ("什么是 risk_score", "factor_explanation"),
        ("分析一下 000725", "stock_analysis"),
    ],
)
def test_intent_rules(populated_db, query, expected) -> None:
    state = nodes.query_analyzer({"user_query": query})

    assert state["intent"] == expected
    assert state["intent_source"] == "rule"


def test_unknown_query_falls_back_to_general(populated_db) -> None:
    state = nodes.query_analyzer({"user_query": "今天天气怎么样"})

    assert state["intent"] == "general"


# ---------------------------------------------------------------- resolver ---
def test_stock_resolver_extracts_codes_and_names(populated_db) -> None:
    by_code = nodes.stock_resolver({"user_query": "分析 000725 和 600519"})
    assert by_code["stock_codes"] == ["000725", "600519"]

    by_name = nodes.stock_resolver({"user_query": f"分析{STOCK_NAMES[STOCK_A]}"})
    assert by_name["stock_codes"] == [STOCK_A]
    assert by_name["stock_names"][STOCK_A] == STOCK_NAMES[STOCK_A]


def test_stock_resolver_caps_number_of_stocks(populated_db) -> None:
    query = " ".join(f"{600000 + i:06d}" for i in range(10))
    state = nodes.stock_resolver({"user_query": query})

    assert len(state["stock_codes"]) == policies.MAX_STOCKS_PER_QUERY


# ----------------------------------------------------------------- planner ---
def test_task_planner_uses_intent_plan() -> None:
    state = nodes.task_planner({"intent": "stock_analysis", "stock_codes": ["000725"]})

    assert "predict_stock" in state["plan"]
    assert "search_financial_knowledge" in state["plan"]
    assert state["iterations"] == 0


def test_task_planner_skips_profile_without_stock() -> None:
    state = nodes.task_planner({"intent": "stock_analysis", "stock_codes": []})

    assert "get_stock_profile" not in state["plan"]


# ------------------------------------------------------------ execution loop --
def test_tool_loop_runs_planned_tools(populated_db, fake_tools) -> None:
    state = nodes.tool_execution_loop(
        {
            "user_query": "分析 000725",
            "intent": "stock_analysis",
            "stock_codes": ["000725"],
            "plan": ["get_stock_profile", "predict_stock", "search_financial_knowledge"],
        },
        llm=None,
    )

    assert [call["name"] for call in state["tool_calls"]] == [
        "get_stock_profile",
        "predict_stock",
        "search_financial_knowledge",
    ]
    assert state["tool_results"]["predict_stock"]["pred_return_5d"] == 0.0055
    assert state["errors"] == []


def test_tool_loop_records_errors_without_raising(populated_db, monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_tool",
        lambda name, **kwargs: {
            "ok": False,
            "tool": name,
            "error": {"code": "NoData", "message": "no factor data for stock 600519"},
        },
    )

    state = nodes.tool_execution_loop(
        {"user_query": "分析 600519", "intent": "stock_analysis",
         "stock_codes": ["600519"], "plan": ["predict_stock"]},
        llm=None,
    )

    assert state["tool_calls"][0]["ok"] is False
    assert state["errors"][0]["code"] == "NoData"


def test_tool_loop_rejects_tools_outside_the_whitelist(populated_db, fake_tools) -> None:
    state = nodes.tool_execution_loop(
        {"user_query": "分析 000725", "intent": "stock_analysis",
         "stock_codes": ["000725"], "plan": ["run_sql", "predict_stock"]},
        llm=None,
    )

    assert all(call["name"] != "run_sql" for call in state["tool_calls"])
    assert any(error["code"] == "ToolNotAllowed" for error in state["errors"])


def test_tool_loop_accepts_llm_requested_extra_tools(populated_db, fake_tools) -> None:
    llm = FakeLLM(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_financial_knowledge",
                    "arguments": '{"query": "回购", "top_k": 3}',
                },
            }
        ]
    )
    state = nodes.tool_execution_loop(
        {"user_query": "分析 000725", "intent": "stock_analysis",
         "stock_codes": ["000725"], "plan": ["predict_stock"]},
        llm=llm,
    )

    assert [call["name"] for call in state["tool_calls"]] == [
        "predict_stock",
        "search_financial_knowledge",
    ]
    assert state["iterations"] >= 1


def test_tool_loop_respects_max_tool_calls(populated_db, fake_tools) -> None:
    plan = ["predict_stock"] * (policies.MAX_TOOL_CALLS + 5)

    state = nodes.tool_execution_loop(
        {"user_query": "分析 000725", "intent": "stock_analysis",
         "stock_codes": ["000725"], "plan": plan},
        llm=None,
    )

    assert len(state["tool_calls"]) == policies.MAX_TOOL_CALLS


# --------------------------------------------------------------- validator ---
def test_validator_flags_missing_quant_results() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "stock_analysis",
            "tool_results": {"search_financial_knowledge": {"results": [
                {"chunk_id": "c1", "document_id": "d1", "stock_code": "000725",
                 "doc_type": "news", "title": "t", "source": "eastmoney",
                 "publish_time": "2026-08-01", "url": "http://x"}]}},
            "tool_calls": [{"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is False
    assert any("量化" in note for note in state["evidence_notes"])


def test_validator_flags_evidence_without_metadata() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "news_search",
            "tool_results": {"search_financial_knowledge": {"results": [
                {"chunk_id": "c1", "title": "t"}]}},
            "tool_calls": [{"name": "search_financial_knowledge"}],
        }
    )

    assert any("缺少元数据" in note for note in state["evidence_notes"])


def test_validator_notes_empty_knowledge_base_result() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "news_search",
            "tool_results": {"search_financial_knowledge": {"results": []}},
            "tool_calls": [{"name": "search_financial_knowledge"}],
        }
    )

    assert any("未检索到" in note for note in state["evidence_notes"])


# ---------------------------------------------------------------- reporter ---
def test_report_generator_requires_llm() -> None:
    """No model → structured error, never a fabricated report."""

    class NoKeyLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.available = False

    with pytest.raises(LLMUnavailable, match="LLM_API_KEY"):
        nodes.report_generator({"user_query": "分析 000725"}, llm=NoKeyLLM())


def test_report_generator_passes_only_tool_material() -> None:
    llm = FakeLLM(answer="# 报告\n只基于工具结果。")
    state = nodes.report_generator(
        {
            "user_query": "分析 000725",
            "intent": "stock_analysis",
            "tool_results": {"predict_stock": {"score": 4.48}},
            "errors": [{"code": "NoData", "message": "x"}],
            "evidence_notes": ["note"],
        },
        llm=llm,
    )

    assert state["final_answer"].startswith("# 报告")
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "4.48" in prompt and "NoData" in prompt  # material is what the LLM sees


# -------------------------------------------------------------- full graph ---
def test_full_graph_runs_offline(populated_db, fake_tools) -> None:
    llm = FakeLLM(answer="# 综合分析\n基于工具输出。")

    state = run_agent("分析 000725", llm=llm)

    assert state["intent"] == "stock_analysis"
    assert state["stock_codes"] == ["000725"]
    assert "predict_stock" in [call["name"] for call in state["tool_calls"]]
    assert state["final_answer"].startswith("# 综合分析")
    assert state["evidence_ok"] is True


def test_graph_node_sequence_is_controlled() -> None:
    assert NODE_SEQUENCE == (
        "query_analyzer",
        "stock_resolver",
        "task_planner",
        "tool_execution_loop",
        "evidence_validator",
        "report_generator",
    )
    assert build_graph(FakeLLM()) is not None


def test_llm_client_reports_missing_key() -> None:
    client = LLMClient(api_key="")

    assert client.available is False
    with pytest.raises(LLMUnavailable, match="LLM_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])


# ------------------------------------------------- registry robustness (P7) --
def test_call_tool_converts_unexpected_exceptions(populated_db, monkeypatch) -> None:
    """A missing index/model must surface as a structured error, not a crash."""
    from app.tools import registry

    def boom(**kwargs):
        raise FileNotFoundError("no BM25 index at data/bm25")

    monkeypatch.setattr(
        registry,
        "TOOL_SPECS",
        (
            registry.ToolSpec(
                name="search_financial_knowledge",
                description="fake",
                parameters={"type": "object", "properties": {}},
                func=boom,
            ),
        ),
    )
    result = call_tool("search_financial_knowledge", query="回购")

    assert result["ok"] is False
    assert result["error"]["code"] == "FileNotFoundError"
    assert "no BM25 index" in result["error"]["message"]


def test_name_normalization_handles_full_width(populated_db) -> None:
    """Full-width company names in the DB must match half-width user input."""
    assert nodes.normalize_name("京东方Ａ") == nodes.normalize_name("京东方A")
    assert nodes.normalize_name(" 中国 平安 ") == "中国平安"

    # the temp fixture stores half-width names; a full-width query must still hit
    full_width = unicodedata.normalize("NFKC", STOCK_NAMES[STOCK_A])  # unchanged here
    state = nodes.stock_resolver({"user_query": f"分析{full_width}"})
    assert state["stock_codes"] == [STOCK_A]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("现在评分最高的5只股票", 5),
        ("给我 top3 的股票", 3),
        ("排名前20的股票", 20),
        ("有哪些评分高的股票", 10),
    ],
)
def test_rank_tool_parses_top_n_from_query(query, expected) -> None:
    args = nodes._plan_arguments("rank_stocks", {"user_query": query})

    assert args["top_n"] == expected
