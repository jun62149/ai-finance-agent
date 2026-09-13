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
MAX_ITERATIONS = Config.AGENT_MAX_ITERATIONS
TOOL_TIMEOUT_SECONDS = Config.AGENT_TOOL_TIMEOUT
MAX_TOOL_CALLS = 12
MAX_STOCKS_PER_QUERY = 5

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
