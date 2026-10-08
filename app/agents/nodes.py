"""Agent state and node implementations (Phase 7).

Controlled workflow (design doc §14):

    query_analyzer -> stock_resolver -> task_planner -> tool_execution_loop
                   -> evidence_validator -> report_generator

The planner is deterministic (`INTENT_TOOL_PLANS`); the LLM only *extends* the
plan with additional tool calls inside the loop, bounded by `MAX_ITERATIONS`.
Tool failures never raise into the loop: `call_tool` returns structured errors
that are recorded and later reported to the user.
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, TypedDict

from app.agents import policies
from app.agents.llm import LLMClient
from app.db import read_sql
from app.tools import call_tool, get_tool, tool_schemas
from app.tools.errors import ToolNotFound, error_payload
from app.tools.quant_tools import FACTORS_TOOL_NAME, PREDICTION_TOOL_NAME, RANKING_TOOL_NAME
from app.tools.rag_tools import REQUIRED_EVIDENCE_FIELDS, SEARCH_TOOL_NAME

CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

#: Quant tools whose presence means "there is a quantitative result to report".
#: `get_stock_profile` is deliberately not one of them: company metadata alone
#: cannot support a quant conclusion.
QUANT_RESULT_TOOLS = (PREDICTION_TOOL_NAME, FACTORS_TOOL_NAME, RANKING_TOOL_NAME)


def normalize_name(text: str) -> str:
    """Fold full-width/half-width and spacing so 京东方A matches 京东方Ａ."""
    return unicodedata.normalize("NFKC", str(text or "")).replace(" ", "").strip()


class AgentState(TypedDict, total=False):
    user_query: str
    intent: str
    intent_source: str
    stock_codes: list[str]
    stock_names: dict[str, str]
    plan: list[str]
    iterations: int
    tool_calls: list[dict[str, Any]]
    tool_results: dict[str, Any]
    errors: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    evidence_ok: bool
    evidence_notes: list[str]
    final_answer: str
    llm_used: bool


# --------------------------------------------------------------- analyzer ----
def classify_intent(query: str, has_stock: bool) -> tuple[str, str]:
    """Rule-based intent classification; returns (intent, source)."""
    text = (query or "").lower()
    for intent, keywords in policies.INTENT_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            if intent == "factor_explanation" and has_stock:
                # "什么是 risk_score" is knowledge; "茅台的风险评分是多少" is analysis
                continue
            return intent, "rule"
    if has_stock:
        return "stock_analysis", "rule"
    return "general", "rule"


def query_analyzer(state: AgentState, *, llm: LLMClient | None = None) -> AgentState:
    query = state.get("user_query", "")
    codes = CODE_RE.findall(query)
    intent, source = classify_intent(query, has_stock=bool(codes) or _mentions_company(query))

    # Only ask the LLM when the rules found nothing meaningful.
    if intent == "general" and not codes and llm is not None and llm.available:
        refined = _llm_classify(query, llm)
        if refined:
            intent, source = refined, "llm"

    return {"intent": intent, "intent_source": source}


def _mentions_company(query: str) -> bool:
    basic = read_sql("SELECT stock_name FROM stock_basic WHERE is_active = 1")
    normalized_query = normalize_name(query)
    for name in basic["stock_name"].astype(str):
        candidate = normalize_name(name)
        if candidate and candidate in normalized_query:
            return True
    return False


def _llm_classify(query: str, llm: LLMClient) -> str | None:
    prompt = (
        "你是查询意图分类器。只输出一个 JSON，形如 {\"intent\": \"...\"}。\n"
        f"可选意图: {', '.join(policies.INTENTS)}\n"
        f"用户问题: {query}"
    )
    try:
        text = llm.chat_text(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64
        )
        payload = json.loads(text[text.find("{") : text.rfind("}") + 1])
        intent = str(payload.get("intent") or "")
        return intent if intent in policies.INTENTS else None
    except Exception:  # noqa: BLE001 - classification fallback must never break the run
        return None


# --------------------------------------------------------------- resolver ----
def stock_resolver(state: AgentState, *, llm: LLMClient | None = None) -> AgentState:
    """Resolve stock codes from explicit codes or company names (no LLM needed)."""
    query = state.get("user_query", "")
    codes = list(dict.fromkeys(CODE_RE.findall(query)))

    basic = read_sql("SELECT stock_code, stock_name FROM stock_basic WHERE is_active = 1")
    normalized_query = normalize_name(query)
    names: dict[str, str] = {}
    for row in basic.itertuples(index=False):
        code = str(row.stock_code).zfill(6)
        name = str(row.stock_name or "")
        names[code] = name
        candidate = normalize_name(name)
        if candidate and candidate in normalized_query and code not in codes:
            codes.append(code)

    codes = codes[: policies.MAX_STOCKS_PER_QUERY]
    return {
        "stock_codes": codes,
        "stock_names": {code: names.get(code, "") for code in codes},
    }


# ---------------------------------------------------------------- planner ----
def task_planner(state: AgentState, *, llm: LLMClient | None = None) -> AgentState:
    intent = state.get("intent", "general")
    plan = list(policies.INTENT_TOOL_PLANS.get(intent, []))
    if not state.get("stock_codes"):
        # No subject resolved: drop the tools that need a stock code rather than
        # calling them with empty arguments (they could only answer
        # InvalidToolArgument). Knowledge search still runs, and the validator
        # reports the missing quantitative material.
        plan = [name for name in plan if name not in policies.CODE_SCOPED_TOOLS]
    # A comparison that resolved fewer than two stocks is not special-cased here:
    # `_tool_queue` already degrades it to the single-stock path (plain keys).
    return {"plan": plan, "iterations": 0, "tool_calls": [], "tool_results": {}, "errors": []}


# ------------------------------------------------------------- tool runner ---
def _run_tool(name: str, kwargs: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Execute one tool with a wall-clock budget; never raises.

    The call runs in its own thread so the loop can give up on it. Note that
    Python cannot kill a running thread: on expiry the caller returns immediately
    (the executor is shut down *without* waiting), while the abandoned call keeps
    running until its own I/O times out. That is why every tool must keep its own
    I/O timeout — the Agent never blocks on a hung tool.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(call_tool, name, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        future.cancel()  # no-op for a call that already started; see docstring
        return {
            "ok": False,
            "tool": name,
            "error": {
                "code": "ToolTimeout",
                "message": f"tool {name} exceeded {timeout:.0f}s",
            },
        }
    finally:
        pool.shutdown(wait=False)


def _plan_arguments(
    name: str, state: AgentState, code: str | None = None
) -> dict[str, Any]:
    """Deterministic arguments for the planned tools.

    `code` pins a call to one stock (used by the comparison expansion); when it is
    omitted the call falls back to the first resolved stock, which is the
    historical single-stock behaviour.
    """
    codes = state.get("stock_codes") or []
    target = code or (codes[0] if codes else None)
    query = state.get("user_query", "")

    if name in policies.CODE_SCOPED_TOOLS:
        return {"code": target} if target else {}

    if name == RANKING_TOOL_NAME:
        top_n = 10
        match = re.search(r"(?:top|前)\s*(\d{1,2})", query.lower()) or re.search(
            r"(\d{1,2})\s*(?:只|个|名|位|支)", query
        )
        if match:
            top_n = max(1, min(int(match.group(1)), policies.MAX_TOOL_CALLS * 4))
        return {"top_n": top_n}

    if name == SEARCH_TOOL_NAME:
        return {
            "query": query or "最新公告",
            "stock_code": target,
            "top_k": 5,
        }

    return {}


def _tool_queue(state: AgentState) -> list[tuple[str, dict[str, Any], str]]:
    """`(tool name, arguments, result key)` triples for the planned tools.

    Single-stock runs keep the historical plain keys (`predict_stock`) so the API,
    the UI and the LLM payload are unchanged for the common case. A comparison
    that resolved 2+ stocks expands every code-scoped tool once per stock and keys
    each result by stock (`predict_stock:600519`) — otherwise the second stock
    would overwrite the first in `tool_results`.
    """
    plan = list(state.get("plan") or [])
    codes = list(state.get("stock_codes") or [])
    multi_stock = state.get("intent") == "stock_comparison" and len(codes) >= 2

    if not multi_stock:
        return [(name, _plan_arguments(name, state), name) for name in plan]

    queue: list[tuple[str, dict[str, Any], str]] = []
    for code in codes:
        for name in plan:
            if name in policies.CODE_SCOPED_TOOLS or name == SEARCH_TOOL_NAME:
                arguments = _plan_arguments(name, state, code=code)
                queue.append((name, arguments, policies.result_key(name, code)))
    for name in plan:  # tools that are not bound to a stock still run exactly once
        if name not in policies.CODE_SCOPED_TOOLS and name != SEARCH_TOOL_NAME:
            queue.append((name, _plan_arguments(name, state), name))
    return queue


def _llm_result_key(name: str, kwargs: dict[str, Any]) -> str:
    """Result key for an LLM-requested call: pin it to the stock it names."""
    if name in policies.CODE_SCOPED_TOOLS:
        code = kwargs.get("code")
    elif name == SEARCH_TOOL_NAME:
        code = kwargs.get("stock_code")
    else:
        code = None
    return policies.result_key(name, str(code) if code else None)


def _extra_tool_calls(state: AgentState, llm: LLMClient | None) -> list[dict[str, Any]]:
    """Ask the LLM whether more tool calls are needed (bounded, optional)."""
    if llm is None or not llm.available:
        return []
    summary = {
        "query": state.get("user_query"),
        "intent": state.get("intent"),
        "stock_codes": state.get("stock_codes"),
        "already_called": [call["name"] for call in state.get("tool_calls", [])],
        "results_preview": {
            name: (str(result)[:200] if not isinstance(result, (int, float)) else result)
            for name, result in (state.get("tool_results") or {}).items()
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是金融投研 Agent 的工具调度器。只有当现有工具结果不足以回答问题时，"
                "才请求额外工具调用；否则不要调用任何工具。禁止编造数据。"
            ),
        },
        {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
    ]
    try:
        message = llm.chat(messages, tools=tool_schemas(), max_tokens=512)
    except Exception:  # noqa: BLE001 - the loop must survive an LLM hiccup
        return []
    return list(message.get("tool_calls") or [])


def tool_execution_loop(state: AgentState, *, llm: LLMClient | None = None) -> AgentState:
    """Run the planned tools, then let the LLM add calls — bounded by policy.

    A multi-stock plan is expanded per stock by `_tool_queue`, and every result is
    stored under its `result_key` (plain tool name for single-stock runs).
    """
    tool_calls = list(state.get("tool_calls") or [])
    tool_results = dict(state.get("tool_results") or {})
    errors = list(state.get("errors") or [])
    iterations = int(state.get("iterations") or 0)

    queue = _tool_queue(state)

    while True:
        while queue and len(tool_calls) < policies.MAX_TOOL_CALLS:
            name, kwargs, key = queue.pop(0)
            if name not in policies.ALLOWED_TOOLS:
                errors.append(
                    {"code": "ToolNotAllowed", "message": f"tool {name} is not allowed"}
                )
                continue
            outcome = _run_tool(name, kwargs, policies.TOOL_TIMEOUT_SECONDS)
            tool_calls.append(
                {
                    "name": name,
                    "arguments": kwargs,
                    "result_key": key,
                    "ok": bool(outcome.get("ok")),
                }
            )
            if outcome.get("ok"):
                tool_results[key] = outcome.get("result")
            else:
                errors.append(dict(outcome.get("error") or {}))

        if len(tool_calls) >= policies.MAX_TOOL_CALLS:
            if queue:
                # A queued call must never be dropped silently: the trace is the
                # only record of what the Agent was asked to do.
                errors.append(
                    {
                        "code": "ToolCallBudgetExceeded",
                        "message": (
                            f"{len(queue)} queued tool call(s) discarded after "
                            f"{policies.MAX_TOOL_CALLS} calls: "
                            + ", ".join(name for name, _kwargs, _key in queue)
                        ),
                    }
                )
            break
        if iterations >= policies.MAX_ITERATIONS - 1:
            break

        state_for_llm: AgentState = {
            **state,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "errors": errors,
        }
        extra = _extra_tool_calls(state_for_llm, llm)
        if not extra:
            break
        iterations += 1
        for call in extra:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                kwargs = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                kwargs = {}
            if not isinstance(kwargs, dict):
                kwargs = {}
            queue.append((name, kwargs, _llm_result_key(name, kwargs)))

    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "errors": errors,
        "iterations": iterations,
    }


# -------------------------------------------------------------- validator ----
def evidence_validator(state: AgentState, *, llm: LLMClient | None = None) -> AgentState:
    """Check that the collected material can actually support an answer.

    `evidence_ok` answers "is this material sufficient to answer?" — not merely
    "did anything go wrong?" — because the UI renders it as a 证据校验 verdict.
    The check is about *subjects*, never about time: a quant baseline date and a
    `publish_time` are allowed to differ (that is a documented project semantic).
    """
    results = state.get("tool_results") or {}
    errors = state.get("errors") or []
    called_tools = {call.get("name") for call in state.get("tool_calls") or []}
    notes: list[str] = []

    # A comparison run stores one entry per stock (`search_financial_knowledge:600519`),
    # so evidence is the union over every search key.
    evidence: list[dict[str, Any]] = []
    for key, result in results.items():
        if policies.base_tool_name(key) == SEARCH_TOOL_NAME and isinstance(result, dict):
            evidence.extend(result.get("results") or [])

    ok = True

    # --- C2: completeness uses the tool layer's single field list -------------
    for item in evidence:
        missing = [field for field in REQUIRED_EVIDENCE_FIELDS if not item.get(field)]
        if missing:
            ok = False
            notes.append(f"证据 {item.get('chunk_id')} 缺少元数据 {missing}")

    has_quant = any(
        policies.base_tool_name(key) in QUANT_RESULT_TOOLS for key in results
    )

    if not results:
        ok = False
        notes.append("没有任何工具返回结果：无法给出结论")
    if SEARCH_TOOL_NAME in called_tools and not evidence:
        # C1: the tool ran and came back empty. "Nothing failed" is not the same
        # as "the material is sufficient", and this flag is user-visible.
        ok = False
        notes.append("知识库未检索到相关证据：结论不应包含文本类事实")
    if state.get("intent") in ("stock_analysis", "stock_comparison", "stock_ranking") and not has_quant:
        ok = False
        notes.append("缺少量化工具结果：不得给出量化结论")

    # --- C3: subject consistency (stock only — dates are deliberately ignored) -
    requested = [str(code) for code in (state.get("stock_codes") or [])]
    if requested:
        requested_set = set(requested)
        for item in evidence:
            code = str(item.get("stock_code") or "")
            if code and code not in requested_set:
                ok = False
                notes.append(
                    f"证据 {item.get('chunk_id')} 的股票 {code} 不属于本次查询股票 {requested}"
                )
        for key, result in results.items():
            if policies.base_tool_name(key) not in policies.CODE_SCOPED_TOOLS:
                continue
            if not isinstance(result, dict):
                continue
            code = str(result.get("code") or "")
            if code and code not in requested_set:
                ok = False
                notes.append(
                    f"量化结果 {key} 的股票 {code} 不属于本次查询股票 {requested}"
                )

    for error in errors:
        notes.append(f"工具错误 {error.get('code')}: {error.get('message')}")

    return {"evidence": evidence, "evidence_ok": ok, "evidence_notes": notes}


# --------------------------------------------------------------- reporter ----
REPORT_SYSTEM_PROMPT = """你是 AI 金融投研助手。

规则：
1. 不得伪造股票数据、新闻、公告或模型结果；只能使用工具返回的内容。
2. 需要量化数据时必须来自量化工具结果；需要文本证据时必须来自检索工具结果，并标注来源。
3. 不要自行计算或修改模型给出的数字。
4. 明确区分「数据事实」「模型结果」「基于证据的推断」。
5. 工具报错或无数据时，必须如实说明，不得用常识填补。
6. 结论不是投资建议。strong_prob 是弱信号辅助字段，不代表可靠上涨概率。

输出结构（Markdown）：
# 股票综合分析
## 1. 量化概览
## 2. 模型解读
## 3. 最新信息（每条附来源）
## 4. 综合判断（区分数据事实 / 模型结果 / 推断）
## 5. 主要风险
## 信息来源
"""


def report_generator(state: AgentState, *, llm: LLMClient | None = None) -> AgentState:
    """Write the final report strictly from collected tool output."""
    if llm is None or not llm.available:
        from app.agents.llm import LLMUnavailable

        raise LLMUnavailable(
            "LLM_API_KEY is not configured; refusing to generate a report without the model"
        )

    payload = {
        "user_query": state.get("user_query"),
        "intent": state.get("intent"),
        "stock_codes": state.get("stock_codes"),
        "stock_names": state.get("stock_names"),
        "tool_results": state.get("tool_results") or {},
        "tool_errors": state.get("errors") or [],
        "validator_notes": state.get("evidence_notes") or [],
        "evidence_ok": state.get("evidence_ok"),
    }
    user_prompt = (
        "以下是本次分析可用的**全部**材料（JSON）。只能基于这些内容作答；"
        "缺失的信息必须明确写出「数据缺失/未检索到」。\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    answer = llm.chat_text(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )
    return {"final_answer": answer, "llm_used": True}
