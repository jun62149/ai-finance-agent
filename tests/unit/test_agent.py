"""Unit tests for the P7 Agent (offline: LLM and tools are faked)."""

from __future__ import annotations

import json
import time
import unicodedata

import pytest

from app.agents import nodes, policies
from app.agents.graph import NODE_SEQUENCE, build_graph, run_agent
from app.agents.llm import LLMClient, LLMUnavailable
from app.tools import call_tool
from app.tools.rag_tools import REQUIRED_EVIDENCE_FIELDS

from tests.conftest import STOCK_A, STOCK_B, STOCK_NAMES


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


class GreedyLLM(FakeLLM):
    """Always asks for one more tool call — the loop must still terminate."""

    def __init__(self, per_round: int = 1, code: str = STOCK_A) -> None:
        super().__init__()
        self.per_round = per_round
        self.code = code
        self.rounds = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        if tools:
            self.rounds += 1
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{self.rounds}_{index}",
                        "type": "function",
                        "function": {
                            "name": "get_stock_factors",
                            "arguments": json.dumps({"code": self.code}),
                        },
                    }
                    for index in range(self.per_round)
                ],
            }
        return {"role": "assistant", "content": self.answer}


@pytest.fixture()
def fake_tools(monkeypatch):
    """Patch the tool runner so the loop never touches real services."""
    calls: list[tuple[str, dict]] = []

    def fake_call_tool(name, **kwargs):
        calls.append((name, kwargs))
        code = kwargs.get("code") or "000725"
        if name == "search_financial_knowledge":
            stock_code = kwargs.get("stock_code") or "000725"
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "count": 1,
                    "has_evidence": True,
                    "results": [
                        {
                            "chunk_id": f"c-{stock_code}",
                            "document_id": f"d-{stock_code}",
                            "stock_code": stock_code,
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
                    "code": code,
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
                "result": {"code": code, "as_of": "2026-06-24", "ret_5d": 0.01},
            }
        if name == "get_stock_profile":
            return {
                "ok": True,
                "tool": name,
                "result": {"code": code, "name": "京东方A", "has_factor_data": True},
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
                 "publish_time": "2026-08-01", "url": "http://x",
                 "content": "正文"}]}},
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
    assert state["evidence_ok"] is False  # incomplete metadata cannot pass as 证据校验


def test_validator_notes_empty_knowledge_base_result() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "news_search",
            "tool_results": {"search_financial_knowledge": {"results": []}},
            "tool_calls": [{"name": "search_financial_knowledge"}],
        }
    )

    assert any("未检索到" in note for note in state["evidence_notes"])
    assert state["evidence_ok"] is False  # zero evidence is "insufficient", not "passed"


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


def test_llm_client_retries_transient_failures(monkeypatch) -> None:
    """Container DNS hiccups must not turn into a failed answer."""
    import httpx

    from app.agents import llm as llm_module

    attempts = {"n": 0}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json():
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def fake_post(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("[Errno -5] No address associated with hostname")
        return FakeResponse()

    monkeypatch.setattr(llm_module.httpx, "post", fake_post)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_: None)

    client = LLMClient(api_key="fake", retries=2)
    message = client.chat([{"role": "user", "content": "hi"}])

    assert message["content"] == "ok"
    assert attempts["n"] == 3


def test_llm_client_does_not_retry_client_errors(monkeypatch) -> None:
    import httpx

    from app.agents import llm as llm_module

    attempts = {"n": 0}

    def fake_post(*args, **kwargs):
        attempts["n"] += 1
        request = httpx.Request("POST", "http://x")
        response = httpx.Response(401, request=request, text="bad key")
        raise httpx.HTTPStatusError("401", request=request, response=response)

    monkeypatch.setattr(llm_module.httpx, "post", fake_post)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_: None)

    client = LLMClient(api_key="fake", retries=2)
    with pytest.raises(LLMUnavailable, match="401"):
        client.chat([{"role": "user", "content": "hi"}])

    assert attempts["n"] == 1  # a bad key is not worth retrying


# ============================ Batch 2 regressions ===========================
# A1 per-stock comparison · A2 degradation · A3 no-stock plan · B1 tool timeout
# B2 MAX_ITERATIONS semantics · B3 budget overflow · C1 zero evidence
# C2 single evidence field list · C3 subject consistency

COMPARISON_PLAN = ["predict_stock", "get_stock_factors", "search_financial_knowledge"]


def _evidence(code: str, chunk_id: str = "c1") -> dict:
    """A complete piece of evidence (every field the tool layer requires)."""
    return {
        "chunk_id": chunk_id,
        "document_id": f"d-{chunk_id}",
        "stock_code": code,
        "doc_type": "news",
        "title": "t",
        "source": "eastmoney",
        "publish_time": "2026-08-01",
        "url": "http://x",
        "content": "正文",
    }


# ------------------------------------------------------- A1 / A2: comparison --
def test_comparison_expands_every_tool_for_every_stock(populated_db, fake_tools) -> None:
    """A1: 'compare two stocks' must actually query both of them."""
    state = nodes.tool_execution_loop(
        {
            "user_query": f"比较一下{STOCK_NAMES[STOCK_A]}和{STOCK_NAMES[STOCK_B]}",
            "intent": "stock_comparison",
            "stock_codes": [STOCK_A, STOCK_B],
            "plan": COMPARISON_PLAN,
        },
        llm=None,
    )

    assert [call["name"] for call in state["tool_calls"]] == COMPARISON_PLAN * 2
    assert [
        call["arguments"].get("code") or call["arguments"].get("stock_code")
        for call in state["tool_calls"]
    ] == [STOCK_A, STOCK_A, STOCK_A, STOCK_B, STOCK_B, STOCK_B]
    # the second stock must not overwrite the first: results are keyed by stock
    assert list(state["tool_results"]) == [
        f"{name}:{code}" for code in (STOCK_A, STOCK_B) for name in COMPARISON_PLAN
    ]
    assert state["tool_results"][f"predict_stock:{STOCK_B}"]["code"] == STOCK_B


def test_comparison_run_covers_both_stocks_end_to_end(populated_db, fake_tools) -> None:
    llm = FakeLLM(answer="# 对比")
    state = run_agent(f"比较一下{STOCK_NAMES[STOCK_A]}和{STOCK_NAMES[STOCK_B]}", llm=llm)

    assert state["intent"] == "stock_comparison"
    assert state["stock_codes"] == [STOCK_A, STOCK_B]
    assert state["evidence_ok"] is True
    assert {item["stock_code"] for item in state["evidence"]} == {STOCK_A, STOCK_B}
    # both stocks' material must reach the model
    payload = llm.calls[-1]["messages"][-1]["content"]
    assert f"predict_stock:{STOCK_A}" in payload
    assert f"predict_stock:{STOCK_B}" in payload


def test_comparison_degraded_to_one_stock_keeps_plain_keys(populated_db, fake_tools) -> None:
    """A2: fewer than two resolved stocks is the single-stock path, not a fake comparison."""
    state = nodes.tool_execution_loop(
        {
            "user_query": f"比较一下{STOCK_NAMES[STOCK_A]}",
            "intent": "stock_comparison",
            "stock_codes": [STOCK_A],
            "plan": COMPARISON_PLAN,
        },
        llm=None,
    )

    assert list(state["tool_results"]) == COMPARISON_PLAN
    assert state["tool_calls"][0]["arguments"] == {"code": STOCK_A}


def test_llm_requested_call_is_keyed_by_its_own_stock(populated_db, fake_tools) -> None:
    """A1: an LLM-added call for another stock cannot clobber the planned result."""
    llm = FakeLLM(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_stock_factors",
                    "arguments": json.dumps({"code": STOCK_B}),
                },
            }
        ]
    )
    state = nodes.tool_execution_loop(
        {
            "user_query": f"分析 {STOCK_A}",
            "intent": "stock_analysis",
            "stock_codes": [STOCK_A],
            "plan": ["get_stock_factors"],
        },
        llm=llm,
    )

    assert set(state["tool_results"]) == {"get_stock_factors", f"get_stock_factors:{STOCK_B}"}
    # the planned result for the queried stock keeps its historical plain key...
    assert state["tool_results"]["get_stock_factors"]["code"] == STOCK_A
    # ...and the validator notices that the extra material is about a stock the
    # user never asked about (C3 subject consistency)
    validated = nodes.evidence_validator(
        {
            "user_query": f"分析 {STOCK_A}",
            "intent": "stock_analysis",
            "stock_codes": [STOCK_A],
            "tool_results": state["tool_results"],
            "tool_calls": state["tool_calls"],
        }
    )
    assert validated["evidence_ok"] is False
    assert any(STOCK_B in note for note in validated["evidence_notes"])


# --------------------------------------------------------- A3: no stock code --
def test_planner_drops_code_scoped_tools_without_a_stock() -> None:
    state = nodes.task_planner({"intent": "stock_comparison", "stock_codes": []})

    assert state["plan"] == ["search_financial_knowledge"]


def test_loop_never_calls_code_scoped_tools_with_empty_arguments(
    populated_db, fake_tools
) -> None:
    plan = nodes.task_planner({"intent": "stock_comparison", "stock_codes": []})["plan"]

    state = nodes.tool_execution_loop(
        {
            "user_query": "比较一下这两只股票",
            "intent": "stock_comparison",
            "stock_codes": [],
            "plan": plan,
        },
        llm=None,
    )

    assert [call["name"] for call in state["tool_calls"]] == ["search_financial_knowledge"]
    assert state["errors"] == []


# ------------------------------------------------------------- B1: timeout ----
def test_tool_timeout_bounds_wall_clock(monkeypatch) -> None:
    """B1: the timeout must release the caller, not wait for the tool to finish."""

    def slow_call_tool(name, **kwargs):
        time.sleep(3.0)
        return {"ok": True, "tool": name, "result": {}}

    monkeypatch.setattr(nodes, "call_tool", slow_call_tool)

    started = time.perf_counter()
    outcome = nodes._run_tool("predict_stock", {"code": "000725"}, 1.0)
    elapsed = time.perf_counter() - started

    assert outcome["error"]["code"] == "ToolTimeout"
    assert elapsed < 2.5, f"timeout did not bound the wait (took {elapsed:.2f}s)"


# ------------------------------------------------- B2 / B3: loop boundaries ---
def test_llm_extra_rounds_are_bounded_by_max_iterations(populated_db, fake_tools) -> None:
    """B2: MAX_ITERATIONS = 6 means 1 planned batch + up to 5 LLM extra rounds."""
    llm = GreedyLLM()

    state = nodes.tool_execution_loop(
        {
            "user_query": f"分析 {STOCK_A}",
            "intent": "stock_analysis",
            "stock_codes": [STOCK_A],
            "plan": ["predict_stock"],
        },
        llm=llm,
    )

    assert policies.MAX_ITERATIONS == 6
    assert llm.rounds == policies.MAX_ITERATIONS - 1
    assert state["iterations"] == policies.MAX_ITERATIONS - 1
    assert len(state["tool_calls"]) == 1 + llm.rounds


def test_max_iterations_one_disables_llm_extra_rounds(
    populated_db, fake_tools, monkeypatch
) -> None:
    """B2 boundary: MAX_ITERATIONS = 1 → 0 LLM extra rounds, planned batch still runs."""
    monkeypatch.setattr(policies, "MAX_ITERATIONS", 1)
    llm = GreedyLLM()

    state = nodes.tool_execution_loop(
        {
            "user_query": f"分析 {STOCK_A}",
            "intent": "stock_analysis",
            "stock_codes": [STOCK_A],
            "plan": ["predict_stock"],
        },
        llm=llm,
    )

    assert llm.rounds == 0
    assert state["iterations"] == 0
    assert [call["name"] for call in state["tool_calls"]] == ["predict_stock"]


def test_tool_budget_overflow_is_reported(populated_db, fake_tools) -> None:
    """B3: calls dropped by the budget must be visible in the trace."""
    plan = ["predict_stock"] * (policies.MAX_TOOL_CALLS + 3)

    state = nodes.tool_execution_loop(
        {
            "user_query": f"分析 {STOCK_A}",
            "intent": "stock_analysis",
            "stock_codes": [STOCK_A],
            "plan": plan,
        },
        llm=None,
    )

    assert len(state["tool_calls"]) == policies.MAX_TOOL_CALLS
    overflow = [e for e in state["errors"] if e["code"] == "ToolCallBudgetExceeded"]
    assert len(overflow) == 1
    assert "3 queued tool call(s) discarded" in overflow[0]["message"]


# --------------------------------------------------- C1: zero evidence --------
def test_validator_treats_zero_evidence_as_insufficient() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "news_search",
            "tool_results": {
                "search_financial_knowledge": {"results": [], "has_evidence": False}
            },
            "tool_calls": [{"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is False
    assert any("未检索到" in note for note in state["evidence_notes"])


def test_news_search_without_evidence_is_reported_as_insufficient(
    populated_db, monkeypatch
) -> None:
    monkeypatch.setattr(
        nodes,
        "call_tool",
        lambda name, **kwargs: {
            "ok": True,
            "tool": name,
            "result": {"count": 0, "has_evidence": False, "results": []},
        },
    )

    state = run_agent(f"{STOCK_NAMES[STOCK_A]}最近有什么重大公告", llm=FakeLLM())

    assert state["intent"] == "news_search"
    assert state["evidence"] == []
    assert state["evidence_ok"] is False


# ------------------------------------------- C2: one evidence field list ------
def test_validator_requires_every_field_from_the_tool_layer_list() -> None:
    item = {field: "x" for field in REQUIRED_EVIDENCE_FIELDS if field != "content"}

    state = nodes.evidence_validator(
        {
            "intent": "news_search",
            "tool_results": {"search_financial_knowledge": {"results": [item]}},
            "tool_calls": [{"name": "search_financial_knowledge"}],
        }
    )

    assert "content" in REQUIRED_EVIDENCE_FIELDS
    assert state["evidence_ok"] is False
    assert state["evidence_notes"] == ["证据 x 缺少元数据 ['content']"]


# ------------------------------------------------- C3: subject consistency ---
def test_validator_accepts_matching_subjects() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "stock_analysis",
            "stock_codes": ["000725"],
            "tool_results": {
                "predict_stock": {"code": "000725"},
                "search_financial_knowledge": {"results": [_evidence("000725")]},
            },
            "tool_calls": [{"name": "predict_stock"}, {"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is True
    assert state["evidence_notes"] == []


def test_validator_rejects_evidence_for_another_stock() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "stock_analysis",
            "stock_codes": ["000725"],
            "tool_results": {
                "predict_stock": {"code": "000725"},
                "search_financial_knowledge": {"results": [_evidence("600519")]},
            },
            "tool_calls": [{"name": "predict_stock"}, {"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is False
    assert any("600519" in note for note in state["evidence_notes"])


def test_validator_rejects_quant_result_for_another_stock() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "stock_analysis",
            "stock_codes": ["000725"],
            "tool_results": {
                "predict_stock": {"code": "600519"},
                "search_financial_knowledge": {"results": [_evidence("000725")]},
            },
            "tool_calls": [{"name": "predict_stock"}, {"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is False
    assert any("量化结果" in note for note in state["evidence_notes"])


def test_validator_accepts_every_requested_stock() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "stock_comparison",
            "stock_codes": ["000725", "600519"],
            "tool_results": {
                "predict_stock:000725": {"code": "000725"},
                "predict_stock:600519": {"code": "600519"},
                "search_financial_knowledge:000725": {"results": [_evidence("000725", "c1")]},
                "search_financial_knowledge:600519": {"results": [_evidence("600519", "c2")]},
            },
            "tool_calls": [{"name": "predict_stock"}, {"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is True
    assert state["evidence_notes"] == []
    assert len(state["evidence"]) == 2
    assert {item["stock_code"] for item in state["evidence"]} == {"000725", "600519"}


def test_validator_does_not_invent_a_subject_for_ranking() -> None:
    state = nodes.evidence_validator(
        {
            "intent": "stock_ranking",
            "stock_codes": [],
            "tool_results": {
                "rank_stocks": {
                    "as_of": "2026-06-24",
                    "count": 1,
                    "items": [{"code": "001211"}],
                }
            },
            "tool_calls": [{"name": "rank_stocks"}],
        }
    )

    assert state["evidence_ok"] is True
    assert state["evidence_notes"] == []


def test_validator_does_not_compare_a_quant_date_with_publish_time() -> None:
    """C3 checks subjects only: a quant baseline date may differ from publish_time."""
    state = nodes.evidence_validator(
        {
            "intent": "stock_analysis",
            "stock_codes": ["000725"],
            "tool_results": {
                "predict_stock": {"code": "000725", "as_of": "2026-06-24"},
                "search_financial_knowledge": {"results": [_evidence("000725")]},
            },
            "tool_calls": [{"name": "predict_stock"}, {"name": "search_financial_knowledge"}],
        }
    )

    assert state["evidence_ok"] is True
    assert state["evidence_notes"] == []
