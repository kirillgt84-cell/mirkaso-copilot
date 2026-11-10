import json
import logging
from typing import Any

import httpx
import pytest

from mirkaso_copilot.azure_client import (
    AzureAPIError,
    AzureClient,
    AzureConfigError,
    describe_error,
    parse_usage,
    v1_base_url,
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
CHAT_URL = "https://test-resource.openai.azure.com/openai/v1/chat/completions"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")


def _transport(captured: list[httpx.Request], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=COMPLETION if status < 400 else {"error": "nope"})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "missing", ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
)
def test_missing_env(env: None, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.delenv(missing)
    with pytest.raises(AzureConfigError, match=missing):
        AzureClient()


async def test_chat_request_and_usage_logging(env: None, caplog: pytest.LogCaptureFixture) -> None:
    captured: list[httpx.Request] = []
    tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "t"}}]
    async with AzureClient(transport=_transport(captured)) as client:
        with caplog.at_level(logging.INFO, logger="mirkaso_copilot.azure_client"):
            body = await client.chat(
                [{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice="auto",
                response_format={"type": "json_object"},
                reasoning_effort="low",
                max_tokens=3000,
            )

    assert body == COMPLETION
    (request,) = captured
    assert str(request.url) == CHAT_URL
    assert request.headers["api-key"] == "test-key"
    assert "Authorization" not in request.headers
    sent = json.loads(request.content)
    assert sent == {
        "model": "test-deployment",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
        "max_completion_tokens": 3000,
    }
    assert "azure chat completion deployment=test-deployment" in caplog.text
    assert "reasoning_tokens=7" in caplog.text
    assert "test-key" not in caplog.text


async def test_reasoning_effort_not_sent_when_none(env: None) -> None:
    captured: list[httpx.Request] = []
    async with AzureClient(transport=_transport(captured)) as client:
        await client.chat([], reasoning_effort=None, max_tokens=100)
    sent = json.loads(captured[0].content)
    assert "reasoning_effort" not in sent
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 100


async def test_minimal_request(env: None) -> None:
    captured: list[httpx.Request] = []
    async with AzureClient(transport=_transport(captured)) as client:
        await client.chat([])
    assert json.loads(captured[0].content) == {"model": "test-deployment", "messages": []}


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://test-resource.openai.azure.com",
        "https://test-resource.openai.azure.com/",
        "https://test-resource.openai.azure.com/openai/v1",
        "https://test-resource.openai.azure.com/openai/v1/",
    ],
)
async def test_endpoint_forms(env: None, monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    captured: list[httpx.Request] = []
    async with AzureClient(transport=_transport(captured)) as client:
        await client.chat([])
    assert str(captured[0].url) == CHAT_URL


def test_v1_base_url() -> None:
    assert v1_base_url("https://r.openai.azure.com") == "https://r.openai.azure.com/openai/v1/"


@pytest.mark.parametrize("effort", ["minimal", "max", ""])
async def test_rejects_unknown_reasoning_effort(env: None, effort: str) -> None:
    captured: list[httpx.Request] = []
    async with AzureClient(transport=_transport(captured)) as client:
        with pytest.raises(ValueError):
            await client.chat([], reasoning_effort=effort)
    assert captured == []


async def test_http_error_no_retry(env: None) -> None:
    captured: list[httpx.Request] = []
    async with AzureClient(transport=_transport(captured, status=503)) as client:
        with pytest.raises(AzureAPIError) as info:
            await client.chat([])
    assert info.value.status_code == 503
    assert len(captured) == 1


def test_parse_usage_variants() -> None:
    assert parse_usage({}).reasoning_tokens is None
    assert parse_usage({"usage": {"reasoning_tokens": 3}}).reasoning_tokens == 3
    assert parse_usage(COMPLETION).total_tokens == 30
    assert parse_usage(COMPLETION).reasoning_tokens == 7


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"error": {"message": "Invalid API key", "type": "auth"}}, "Invalid API key"),
        ({"error": "nope"}, "nope"),
        ({"code": "401", "message": "Access denied"}, "Access denied"),
        ({"detail": []}, None),
    ],
)
async def test_describe_http_error(env: None, body: dict[str, Any], message: str | None) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json=body))
    async with AzureClient(transport=transport) as client:
        with pytest.raises(AzureAPIError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}])
    assert describe_error(exc_info.value) == {
        "error_type": "AzureAPIError",
        "status_code": 401,
        "message": message,
    }


async def test_describe_error_never_includes_the_key(env: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "401", "message": "Access denied"}})

    async with AzureClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AzureAPIError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}])
    assert "test-key" not in json.dumps(describe_error(exc_info.value))
    assert "test-key" not in str(exc_info.value)


def test_describe_error_truncates_and_handles_plain_text() -> None:
    long = AzureAPIError("x", status_code=500, body=json.dumps({"error": {"message": "m" * 400}}))
    assert describe_error(long)["message"] == "m" * 200
    plain = AzureAPIError("x", status_code=502, body="Bad gateway\n")
    assert describe_error(plain)["message"] == "Bad gateway"
    empty = AzureAPIError("x", status_code=503, body="")
    assert describe_error(empty)["message"] is None


async def test_describe_connection_error_gives_type_only(env: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to secret-host", request=request)

    async with AzureClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AzureAPIError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}])
    assert describe_error(exc_info.value) == {
        "error_type": "ConnectError",
        "status_code": None,
        "message": None,
    }


def test_describe_other_exceptions_gives_type_only() -> None:
    assert describe_error(KeyError("choices")) == {
        "error_type": "KeyError",
        "status_code": None,
        "message": None,
    }
