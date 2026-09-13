"""LLM client for the Agent (DeepSeek, OpenAI-compatible chat completions).

Kept deliberately small and dependency-free (httpx only) so the tool-calling
loop is easy to test and easy to swap to another OpenAI-compatible provider by
changing `LLM_BASE_URL` / `LLM_MODEL`.

Failure is explicit: a missing key, a network error or a malformed response
raises `LLMUnavailable`, and the Agent must surface that as a structured error
instead of writing a report from nothing.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import Config
from app.errors import QuantError


class LLMUnavailable(QuantError):
    code = "LLMUnavailable"
    http_status = 503


class LLMClient:
    """Minimal chat-completions client with optional tool calling."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else Config.LLM_API_KEY
        self.base_url = (base_url or Config.LLM_BASE_URL).rstrip("/")
        self.model = model or Config.LLM_MODEL
        self.timeout = timeout or Config.LLM_TIMEOUT
        self.temperature = Config.LLM_TEMPERATURE if temperature is None else temperature

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """One chat completion. Returns the raw `message` object from the API."""
        if not self.available:
            raise LLMUnavailable(
                "LLM_API_KEY is not set — put it in .env (LLM_API_KEY=...); "
                "the Agent never fabricates an answer without the model"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailable(
                f"LLM HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - network / JSON failures
            raise LLMUnavailable(f"LLM request failed: {type(exc).__name__}: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMUnavailable(f"LLM returned no choices: {json.dumps(data)[:200]}")
        message = choices[0].get("message") or {}
        message.setdefault("role", "assistant")
        return message

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Convenience wrapper returning just the text content."""
        message = self.chat(messages, **kwargs)
        return str(message.get("content") or "")
