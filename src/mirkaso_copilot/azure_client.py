"""Minimal async client for Azure OpenAI chat completions (the v1 API).

Configuration comes only from the environment:

- ``AZURE_OPENAI_API_KEY``    (required) sent in the ``api-key`` header
- ``AZURE_OPENAI_ENDPOINT``   (required) e.g. https://<resource>.openai.azure.com
- ``AZURE_OPENAI_DEPLOYMENT`` (required) the deployment name, not the model name

Requests go to ``<endpoint>/openai/v1/chat/completions`` with the deployment name in the
``model`` field. The v1 API needs no ``api-version`` parameter.

No retries yet: any failure surfaces to the caller as ``AzureAPIError``.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

V1_PATH = "/openai/v1"
REASONING_EFFORTS = frozenset({"low", "medium", "high"})
_MAX_ERROR_BODY = 500
_MAX_DETAIL_MESSAGE = 200

logger = logging.getLogger(__name__)


class AzureConfigError(RuntimeError):
    """Required configuration is missing from the environment."""


class AzureAPIError(RuntimeError):
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


def _body_error_message(body: str | None) -> str | None:
    """The ``error.message`` (or ``message``/``error`` string) of an error body, else its text."""
    if not body:
        return None
    try:
        decoded = json.loads(body)
    except ValueError:
        decoded = None
    message: object = None
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
        message = message or decoded.get("message")
    if decoded is None:
        message = body
    if not isinstance(message, str) or not message.strip():
        return None
    return message.strip()[:_MAX_DETAIL_MESSAGE]


def describe_error(exc: BaseException) -> dict[str, Any]:
    """A short, safe description of a failed model request, for opt-in diagnostics.

    ``{"error_type", "status_code", "message"}``. ``status_code`` and ``message`` are set
    only when the API answered with an HTTP error; ``message`` is at most 200 characters of
    the error message in the response body. Request headers (and so the API key) and
    exception text are never included. Without an HTTP response (e.g. a connection error)
    only the type of the underlying exception is given.
    """
    if isinstance(exc, AzureAPIError) and exc.status_code is not None:
        return {
            "error_type": type(exc).__name__,
            "status_code": exc.status_code,
            "message": _body_error_message(exc.body),
        }
    cause = exc.__cause__ if isinstance(exc, AzureAPIError) and exc.__cause__ else exc
    return {"error_type": type(cause).__name__, "status_code": None, "message": None}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AzureConfigError(f"{name} is not set")
    return value


def v1_base_url(endpoint: str) -> str:
    """``<endpoint>/openai/v1/``; an endpoint that already ends in ``/openai/v1`` is kept."""
    base = endpoint.strip().rstrip("/")
    if not base.endswith(V1_PATH):
        base += V1_PATH
    return base + "/"


class AzureClient:
    def __init__(
        self, *, timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        api_key = _require_env("AZURE_OPENAI_API_KEY")
        base_url = v1_base_url(_require_env("AZURE_OPENAI_ENDPOINT"))
        # The v1 API takes the deployment name where OpenAI takes a model id.
        self.model = _require_env("AZURE_OPENAI_DEPLOYMENT")
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "AzureClient":
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
        """POST /chat/completions and return the decoded response body.

        ``max_tokens`` is sent as ``max_completion_tokens``: reasoning models accept only
        that, and the API marks ``max_tokens`` as deprecated. ``reasoning_effort`` is sent
        only when given, since models without reasoning reject it.
        """
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
            payload["max_completion_tokens"] = max_tokens

        try:
            response = await self._http.post("chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise AzureAPIError(f"request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise AzureAPIError(
                f"Azure OpenAI returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text[:_MAX_ERROR_BODY],
            )

        body = response.json()
        usage = parse_usage(body)
        logger.info(
            "azure chat completion deployment=%s prompt_tokens=%s completion_tokens=%s "
            "reasoning_tokens=%s total_tokens=%s",
            self.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
        )
        return body
