"""Tool-layer errors.

Tools reuse the Quant Engine error codes for data problems (StockNotFound,
NoData, ModelUnavailable, ...) so the API and the Agent expose **identical**
semantics — an Agent must never receive a different failure vocabulary than the
HTTP contract. Only genuinely tool-specific failures get new codes here.
"""

from __future__ import annotations

from app.errors import QuantError


class ToolError(QuantError):
    """Base class for tool boundary failures (bad arguments, unknown tool)."""

    code = "ToolError"
    http_status = 400


class ToolArgumentError(ToolError):
    code = "InvalidToolArgument"


class ToolNotFound(ToolError):
    code = "UnknownTool"


class EvidenceMetadataMissing(ToolError):
    """A retrieved item lacked the traceable metadata every citation requires."""

    code = "EvidenceMetadataMissing"
    http_status = 500


class ToolTimeout(ToolError):
    """A tool exceeded its budget (enforced by the Agent layer in P7)."""

    code = "ToolTimeout"
    http_status = 504


def error_payload(exc: BaseException) -> dict[str, str]:
    """Structured error body shared by API responses and tool results."""
    code = getattr(exc, "code", None) or type(exc).__name__
    message = getattr(exc, "message", None) or str(exc)
    return {"code": str(code), "message": str(message)}
