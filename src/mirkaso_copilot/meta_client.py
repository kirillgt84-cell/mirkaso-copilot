"""Minimal async client for the Meta Model API (OpenAI-compatible chat completions).

Configuration comes only from the environment:

- ``META_API_KEY``  (required) bearer token
- ``META_MODEL``    (required) model id, e.g. the Muse Spark model
- ``META_API_BASE`` (optional) defaults to https://api.meta.ai/v1

No retries yet: any failure surfaces to the caller as ``MetaAPIError``.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_BASE = "https://api.meta.ai/v1"
REASONING_EFFORTS = frozenset({"low", "medium", "high"})
_MAX_ERROR_BODY = 500

logger = logging.getLogger(__name__)


class MetaConfigError(RuntimeError):
    """Required configuration is missing from the environment."""


class MetaAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None


def parse_usage(body: dict[str, Any]) -> Usage:
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")
    return Usage(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=reasoning,
        total_tokens=usage.get("total_tokens"),
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MetaConfigError(f"{name} is not set")
    return value


class MetaClient:
    def __init__(
        self, *, timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        api_key = _require_env("META_API_KEY")
        self.model = _require_env("META_MODEL")
        base_url = os.environ.get("META_API_BASE") or DEFAULT_API_BASE
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "MetaClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """POST /chat/completions and return the decoded response body."""
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self._http.post("chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise MetaAPIError(f"request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise MetaAPIError(
                f"Meta API returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text[:_MAX_ERROR_BODY],
            )

        body = response.json()
        usage = parse_usage(body)
        logger.info(
            "meta chat completion model=%s prompt_tokens=%s completion_tokens=%s "
            "reasoning_tokens=%s total_tokens=%s",
            self.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
        )
        return body
