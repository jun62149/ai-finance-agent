"""Agent API (design doc §20): POST /api/chat.

Returns the final report plus the execution trace (intent, plan, tool calls,
tool errors, validator notes) so a demo or an evaluation can show *why* the
answer looks the way it does. Every failure is a structured `{code, message}`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agents import graph as agent_graph
from app.agents.llm import LLMClient


router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="自然语言问题")


class ChatResponse(BaseModel):
    query: str
    answer: str
    intent: str
    intent_source: str
    stock_codes: list[str]
    plan: list[str]
    tool_calls: list[dict[str, Any]]
    tool_errors: list[dict[str, Any]]
    evidence_count: int
    evidence_ok: bool
    validator_notes: list[str]
    model: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the controlled Agent workflow for one question."""
    state = agent_graph.run_agent(request.query, llm=LLMClient())
    evidence = state.get("evidence") or []
    return ChatResponse(
        query=request.query,
        answer=str(state.get("final_answer") or ""),
        intent=str(state.get("intent") or ""),
        intent_source=str(state.get("intent_source") or ""),
        stock_codes=list(state.get("stock_codes") or []),
        plan=list(state.get("plan") or []),
        tool_calls=list(state.get("tool_calls") or []),
        tool_errors=list(state.get("errors") or []),
        evidence_count=len(evidence),
        evidence_ok=bool(state.get("evidence_ok")),
        validator_notes=list(state.get("evidence_notes") or []),
        model=LLMClient().model,
    )
