"""Agent policies: intents, tool plans and hard limits.

Everything an Agent is *allowed* to do lives here, so the boundaries are
auditable in one file (design doc §14/§28): which tools exist, which intent maps
to which tools, how many loops, how long a tool may run.
"""

from __future__ import annotations

from app.config import Config
from app.tools import quant_tools, rag_tools

#: Intents the Agent understands (design doc §16).
INTENTS = (
    "stock_analysis",
    "stock_comparison",
    "stock_ranking",
    "news_search",
    "factor_explanation",
    "general",
)

#: Deterministic intent → tools plan. The LLM may request *additional* tools
#: inside the execution loop, but it can never skip this baseline.
INTENT_TOOL_PLANS: dict[str, list[str]] = {
    "stock_analysis": [
        quant_tools.PROFILE_TOOL_NAME,
        quant_tools.PREDICTION_TOOL_NAME,
        quant_tools.FACTORS_TOOL_NAME,
        rag_tools.SEARCH_TOOL_NAME,
    ],
    "stock_comparison": [
        quant_tools.PREDICTION_TOOL_NAME,
        quant_tools.FACTORS_TOOL_NAME,
        rag_tools.SEARCH_TOOL_NAME,
    ],
    "stock_ranking": [quant_tools.RANKING_TOOL_NAME],
    "news_search": [rag_tools.SEARCH_TOOL_NAME],
    # knowledge-only answer: the LLM explains the concept, no data tool needed
    "factor_explanation": [],
    "general": [],
}

#: Hard limits — an Agent must terminate and must not hammer tools.
#:
#: `MAX_ITERATIONS` counts *loop bodies*, not LLM rounds. Every run always starts
#: with one batch of planned tools (that batch does not increment `iterations`);
#: afterwards the LLM may request extra tool calls at most `MAX_ITERATIONS - 1`
#: times (`nodes.tool_execution_loop`). The default 6 therefore means:
#:
#:     1 batch of planned tools + up to 5 LLM extra-tool rounds
#:
#: and `MAX_ITERATIONS = 1` (or 0) disables LLM extra-tool rounds entirely — the
#: planned batch still runs. Total tool *calls* are additionally capped by
#: `MAX_TOOL_CALLS`.
MAX_ITERATIONS = Config.AGENT_MAX_ITERATIONS

#: Wall-clock budget for one tool call. On expiry the loop records a structured
#: `ToolTimeout` and moves on; the abandoned call keeps running in its own thread
#: (Python cannot kill a running thread — see `nodes._run_tool`).
TOOL_TIMEOUT_SECONDS = Config.AGENT_TOOL_TIMEOUT
MAX_TOOL_CALLS = 12
MAX_STOCKS_PER_QUERY = 5

#: Tools whose arguments are bound to one stock code. A run that covers several
#: stocks (`stock_comparison`) calls these once per stock and keys each result by
#: stock, so a second stock cannot overwrite the first in `tool_results`.
CODE_SCOPED_TOOLS = frozenset(
    [
        quant_tools.PROFILE_TOOL_NAME,
        quant_tools.FACTORS_TOOL_NAME,
        quant_tools.PREDICTION_TOOL_NAME,
    ]
)

#: `tool_results` keys of multi-stock runs carry the stock (`predict_stock:600519`),
#: while single-stock runs keep the historical plain tool name. These two helpers
#: are the only place where that convention is encoded.
RESULT_KEY_SEPARATOR = ":"


def result_key(name: str, code: str | None = None) -> str:
    """Key a tool result is stored under (stock-suffixed when pinned to a stock)."""
    return f"{name}{RESULT_KEY_SEPARATOR}{code}" if code else name


def base_tool_name(key: str) -> str:
    """Tool name behind a `tool_results` key (`predict_stock:600519` → `predict_stock`)."""
    return str(key).split(RESULT_KEY_SEPARATOR, 1)[0]


#: Tool names the loop may call (whitelist; no SQL/shell/file tools exist).
ALLOWED_TOOLS = frozenset(
    [
        rag_tools.SEARCH_TOOL_NAME,
        quant_tools.PROFILE_TOOL_NAME,
        quant_tools.FACTORS_TOOL_NAME,
        quant_tools.PREDICTION_TOOL_NAME,
        quant_tools.RANKING_TOOL_NAME,
    ]
)

#: Rule keywords used by the deterministic query analyzer (before any LLM call).
INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stock_ranking", ("排名", "排行", "评分最高", "选股", "top", "top10", "推荐哪些", "哪些股票")),
    ("stock_comparison", ("比较", "对比", "vs", "哪个更好", "谁更强")),
    ("news_search", ("新闻", "公告", "消息", "报道", "快讯", "最新动态", "舆情")),
    ("factor_explanation", ("什么是", "是什么", "解释", "含义", "怎么算", "如何理解")),
)
