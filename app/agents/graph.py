"""LangGraph wiring for the controlled investment-research Agent (Phase 7).

Graph: query_analyzer -> stock_resolver -> task_planner -> tool_execution_loop
       -> evidence_validator -> report_generator

The graph is deterministic; the only LLM-driven step inside it is the optional
extra-tool-call request in `tool_execution_loop` (bounded by MAX_ITERATIONS).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.agents import nodes
from app.agents.llm import LLMClient
from app.agents.nodes import AgentState

NODE_SEQUENCE = (
    "query_analyzer",
    "stock_resolver",
    "task_planner",
    "tool_execution_loop",
    "evidence_validator",
    "report_generator",
)


def build_graph(llm: LLMClient | None = None) -> Any:
    """Compile the agent graph. `llm` is injected so tests can run offline."""
    graph = StateGraph(AgentState)

    graph.add_node("query_analyzer", lambda state: nodes.query_analyzer(state, llm=llm))
    graph.add_node("stock_resolver", lambda state: nodes.stock_resolver(state, llm=llm))
    graph.add_node("task_planner", lambda state: nodes.task_planner(state, llm=llm))
    graph.add_node(
        "tool_execution_loop", lambda state: nodes.tool_execution_loop(state, llm=llm)
    )
    graph.add_node(
        "evidence_validator", lambda state: nodes.evidence_validator(state, llm=llm)
    )
    graph.add_node("report_generator", lambda state: nodes.report_generator(state, llm=llm))

    graph.set_entry_point("query_analyzer")
    for current, following in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        graph.add_edge(current, following)
    graph.add_edge("report_generator", END)
    return graph.compile()


def run_agent(
    query: str,
    *,
    llm: LLMClient | None = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Run one question through the agent and return the final state.

    Raises the underlying structured error (e.g. LLMUnavailable) — the API layer
    converts it into the same `{code, message}` shape as the Quant API.
    """
    llm = llm or LLMClient()
    compiled = graph if graph is not None else build_graph(llm)
    result = compiled.invoke({"user_query": query})
    return dict(result)
