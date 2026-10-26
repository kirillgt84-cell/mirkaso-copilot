import json
import logging
from typing import Any

import httpx
import pytest

from mirkaso_copilot.meta_client import (
    MetaAPIError,
    MetaClient,
    MetaConfigError,
    parse_usage,
)

COMPLETION = {
    "id": "c1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 7},
    },
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("META_MODEL", "test-model")
    monkeypatch.delenv("META_API_BASE", raising=False)


def _transport(captured: list[httpx.Request], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=COMPLETION if status < 400 else {"error": "nope"})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("missing", ["META_API_KEY", "META_MODEL"])
def test_missing_env(env: None, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.delenv(missing)
    with pytest.raises(MetaConfigError, match=missing):
        MetaClient()


async def test_chat_request_and_usage_logging(env: None, caplog: pytest.LogCaptureFixture) -> None:
    captured: list[httpx.Request] = []
    tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "t"}}]
    async with MetaClient(transport=_transport(captured)) as client:
        with caplog.at_level(logging.INFO, logger="mirkaso_copilot.meta_client"):
            body = await client.chat(
                [{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice="auto",
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )

    assert body == COMPLETION
    (request,) = captured
    assert str(request.url) == "https://api.meta.ai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    sent = json.loads(request.content)
    assert sent == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
    }
    assert "reasoning_tokens=7" in caplog.text
    assert "test-key" not in caplog.text


async def test_api_base_override(env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_BASE", "https://example.test/v9")
    captured: list[httpx.Request] = []
    async with MetaClient(transport=_transport(captured)) as client:
        await client.chat([])
    assert str(captured[0].url) == "https://example.test/v9/chat/completions"
    assert "tools" not in json.loads(captured[0].content)


@pytest.mark.parametrize("effort", ["minimal", "max", ""])
async def test_rejects_unknown_reasoning_effort(env: None, effort: str) -> None:
    captured: list[httpx.Request] = []
    async with MetaClient(transport=_transport(captured)) as client:
        with pytest.raises(ValueError):
            await client.chat([], reasoning_effort=effort)
    assert captured == []


async def test_http_error_no_retry(env: None) -> None:
    captured: list[httpx.Request] = []
    async with MetaClient(transport=_transport(captured, status=503)) as client:
        with pytest.raises(MetaAPIError) as info:
            await client.chat([])
    assert info.value.status_code == 503
    assert len(captured) == 1


def test_parse_usage_variants() -> None:
    assert parse_usage({}).reasoning_tokens is None
    assert parse_usage({"usage": {"reasoning_tokens": 3}}).reasoning_tokens == 3
    assert parse_usage(COMPLETION).total_tokens == 30
