"""Tool registry: one place that describes the tools an Agent may call.

`list_tools()` gives the LangGraph/function-calling layer names, descriptions and
JSON Schemas; `call_tool()` is the agent-facing boundary — it never lets a data
error escape as an exception, it returns a structured error object instead
(design doc §9: no silent fallbacks, no exceptions leaking into the LLM loop).

Tools are the *only* way the Agent may touch quant data or the knowledge base:
there is no SQL, shell or filesystem tool here by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.errors import QuantError
from app.tools import quant_tools, rag_tools
from app.tools.errors import ToolNotFound, error_payload


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """Function-calling schema (OpenAI/DeepSeek-compatible shape)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name=rag_tools.SEARCH_TOOL_NAME,
        description=(
            "检索金融新闻与公告知识库，返回带来源（标题/日期/来源/链接）的证据片段。"
            "用于回答「最近有什么公告」「某事件的报道」等文本类问题；"
            "量化因子、模型预测、选股排名请改用对应的量化工具。"
        ),
        parameters=rag_tools.SEARCH_TOOL_SCHEMA,
        func=rag_tools.search_financial_knowledge,
    ),
    ToolSpec(
        name=quant_tools.PROFILE_TOOL_NAME,
        description="查询股票基础信息（名称、市场、上市日期）以及本地是否有量化数据。",
        parameters=quant_tools.PROFILE_TOOL_SCHEMA,
        func=quant_tools.get_stock_profile,
    ),
    ToolSpec(
        name=quant_tools.FACTORS_TOOL_NAME,
        description="查询股票的技术与流动性因子（动量、波动、量比、回撤等），支持历史时点。",
        parameters=quant_tools.FACTORS_TOOL_SCHEMA,
        func=quant_tools.get_stock_factors,
    ),
    ToolSpec(
        name=quant_tools.PREDICTION_TOOL_NAME,
        description=(
            "调用 XGBoost 模型获取 5 日超额收益预测、风险评分与综合评分（支持历史时点）。"
            "strong_prob 为弱信号辅助字段，不得当作可靠上涨概率。"
        ),
        parameters=quant_tools.PREDICTION_TOOL_SCHEMA,
        func=quant_tools.predict_stock,
    ),
    ToolSpec(
        name=quant_tools.RANKING_TOOL_NAME,
        description="按模型综合评分返回 Top-N 股票（仅含通过流动性/板块过滤的入池股票）。",
        parameters=quant_tools.RANKING_TOOL_SCHEMA,
        func=quant_tools.rank_stocks,
    ),
)


def list_tools() -> list[ToolSpec]:
    return list(TOOL_SPECS)


def tool_names() -> list[str]:
    return [spec.name for spec in TOOL_SPECS]


def get_tool(name: str) -> ToolSpec:
    for spec in TOOL_SPECS:
        if spec.name == name:
            return spec
    raise ToolNotFound(
        f"unknown tool {name!r}; available: {', '.join(tool_names())}"
    )


def tool_schemas() -> list[dict[str, Any]]:
    return [spec.schema() for spec in TOOL_SPECS]


def call_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """Run a tool, converting any structured error into a result object.

    Returns `{"ok": True, "result": ...}` or
    `{"ok": False, "error": {"code": ..., "message": ...}}`.
    """
    try:
        spec = get_tool(name)
        result = spec.func(**kwargs)
    except QuantError as exc:
        return {"ok": False, "tool": name, "error": error_payload(exc)}
    except TypeError as exc:  # wrong keyword arguments
        return {
            "ok": False,
            "tool": name,
            "error": {"code": "InvalidToolArgument", "message": str(exc)},
        }
    return {"ok": True, "tool": name, "result": result}
