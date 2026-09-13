"""Agent tools (P5/P6).

`rag_tools`  — knowledge-base retrieval with traceable evidence metadata.
`quant_tools` — the Quant Engine services (factors / prediction / ranking).
`registry`   — tool descriptions, JSON schemas and the structured-error boundary.

Design rule: tools are the only bridge between the Agent and the data. There is
deliberately no SQL / shell / filesystem tool.
"""

from app.tools.registry import (
    ToolSpec,
    call_tool,
    get_tool,
    list_tools,
    tool_names,
    tool_schemas,
)

__all__ = [
    "ToolSpec",
    "call_tool",
    "get_tool",
    "list_tools",
    "tool_names",
    "tool_schemas",
]
