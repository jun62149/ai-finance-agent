"""Quant tools (Phase 6): the existing Quant services, exposed to an Agent.

These tools are deliberately *thin*: they validate arguments at the boundary and
then call the same service functions the FastAPI layer uses, so a tool result
and an API response can never diverge in meaning or error code.

Iron rule: the LLM never computes quant numbers — it may only call these tools.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from app.config import Config
from app.quant.ranking import target_factor_date
from app.services import prediction_service, stock_service
from app.tools.errors import ToolArgumentError

PROFILE_TOOL_NAME = "get_stock_profile"
FACTORS_TOOL_NAME = "get_stock_factors"
PREDICTION_TOOL_NAME = "predict_stock"
RANKING_TOOL_NAME = "rank_stocks"

MAX_TOP_N = 50


def _validate_as_of(as_of: str | None) -> str | None:
    if as_of is None:
        return None
    text = str(as_of).strip()
    if not text:
        return None
    try:
        date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ToolArgumentError(
            f"as_of must be an ISO date (YYYY-MM-DD), got {as_of!r}"
        ) from exc
    return text[:10]


def _validate_code(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ToolArgumentError("code must be a 6-digit stock code")
    return stock_service.validate_code(code.strip())


PROFILE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "description": "6 位股票代码，如 000725"}},
    "required": ["code"],
}

FACTORS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "6 位股票代码"},
        "as_of": {
            "type": "string",
            "description": "可选，历史时点（YYYY-MM-DD）；不传则取最新交易日",
        },
    },
    "required": ["code"],
}

PREDICTION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "6 位股票代码"},
        "as_of": {
            "type": "string",
            "description": "可选，历史时点（YYYY-MM-DD）；不传则取最新交易日",
        },
    },
    "required": ["code"],
}

RANKING_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "top_n": {"type": "integer", "description": f"返回条数（1~{MAX_TOP_N}，默认 10）"},
        "as_of": {"type": "string", "description": "可选，历史时点（YYYY-MM-DD）"},
        "board": {
            "type": "string",
            "enum": ["main", "growth", "star", "bj"],
            "description": "可选，板块过滤",
        },
    },
    "required": [],
}


def get_stock_profile(code: str) -> dict[str, Any]:
    """Basic company info (name / market / listing date) for a stock code."""
    code = _validate_code(code)
    basic = stock_service.get_stock_basic(code)
    return {
        "code": code,
        "name": basic.get("stock_name"),
        "market": basic.get("market"),
        "list_date": basic.get("list_date"),
        "is_active": bool(basic.get("is_active", 1)),
        "has_factor_data": stock_service.has_factor_data(code),
    }


def get_stock_factors(code: str, as_of: str | None = None) -> dict[str, Any]:
    """Latest (or as-of) quant factors for a stock."""
    code = _validate_code(code)
    as_of = _validate_as_of(as_of)
    stock_service.get_stock_basic(code)
    row = stock_service.get_factor_row(code, as_of)
    payload = stock_service.factor_row_to_dict(row.iloc[0])
    payload["model_note"] = (
        "因子为技术/流动性指标；strong_prob 为弱信号辅助字段，不代表可靠上涨概率。"
    )
    payload["disclaimer"] = "数据与模型分析结果，非投资建议。"
    return payload


def predict_stock(code: str, as_of: str | None = None) -> dict[str, Any]:
    """XGBoost prediction (5-day excess return, risk, score) for a stock."""
    code = _validate_code(code)
    as_of = _validate_as_of(as_of)
    prediction = prediction_service.predict_stock(code, as_of)
    payload = prediction.model_dump()
    payload["disclaimer"] = "模型分析结果，非投资建议。"
    return payload


def rank_stocks(
    top_n: int = 10,
    as_of: str | None = None,
    board: str | None = None,
) -> dict[str, Any]:
    """Top-N stocks by model score (only pool-eligible names are ranked)."""
    try:
        top_n = int(top_n)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(f"top_n must be an integer, got {top_n!r}") from exc
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolArgumentError(f"top_n must be between 1 and {MAX_TOP_N}, got {top_n}")
    if board is not None and board not in Config.BOARD_VALUES:
        raise ToolArgumentError(
            f"board must be one of {Config.BOARD_VALUES}, got {board!r}"
        )
    as_of = _validate_as_of(as_of)

    # resolve the trade date the ranking is actually computed on (StockRankingItem
    # carries no date of its own)
    resolved_date = target_factor_date(as_of) or ""
    items = prediction_service.rank_stocks(top_n, as_of, board)
    return {
        "as_of": resolved_date,
        "top_n": top_n,
        "board": board,
        "count": len(items),
        "items": [item.model_dump() for item in items],
        "disclaimer": "模型分析结果，非投资建议；strong_prob 为弱信号辅助字段。",
    }
