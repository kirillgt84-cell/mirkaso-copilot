"""Agent loop tests with a scripted fake model client. No network."""

import asyncio
import json
import logging
import re
from typing import Any

import pytest

from mirkaso_copilot.agent import FALLBACK_ANSWERS, AgentConfig, run_agent
from mirkaso_copilot.meta_client import MetaAPIError
from mirkaso_copilot.providers import CallerContext, Tier
from tests.conftest import AS_OF, FakeProvider, ScriptedClient, call, reply

CTX = CallerContext(tier=Tier.TRADER)
SECRET = "db password hunter2 at 10.0.0.1"


def tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] == "tool"]


def tool_payload(message: dict[str, Any]) -> dict[str, Any]:
    return json.loads(message["content"])


async def test_parallel_tool_calls_run_concurrently(provider: FakeProvider) -> None:
    macro_started, derivs_started = asyncio.Event(), asyncio.Event()
    macro, derivs = provider.macro_regime, provider.coin_derivatives

    # Each tool waits for the other to start: this only completes if both run at once.
    async def slow_macro() -> dict[str, Any]:
        macro_started.set()
        await derivs_started.wait()
        return await macro()

    async def slow_derivs(**kwargs: Any) -> dict[str, Any] | None:
        derivs_started.set()
        await macro_started.wait()
        return await derivs(**kwargs)

    provider.macro_regime = slow_macro  # type: ignore[method-assign]
    provider.coin_derivatives = slow_derivs  # type: ignore[method-assign]
    client = ScriptedClient(
        reply(
            tool_calls=[
                call("macro_get_regime", {}, "a"),
                call("coin_get_derivatives", {"symbol": "BTC"}, "b"),
            ]
        ),
        reply(f"Regime is Neutral as of {AS_OF}."),
    )
    config = AgentConfig(per_tool_timeout_s=2)

    result = await asyncio.wait_for(
        run_agent("How is BTC?", provider=provider, ctx=CTX, client=client, config=config), 5
    )

    assert result.stop_reason == "final"
    assert result.answer == f"Regime is Neutral as of {AS_OF}."
    assert result.steps == 2
    assert [(r.name, r.ok, r.error_code) for r in result.tool_calls] == [
        ("macro_get_regime", True, None),
        ("coin_get_derivatives", True, None),
    ]
    assert {r.source_id for r in result.tool_calls} == {"mirkaso.macro", "mirkaso.derivatives"}
    assert all(r.as_of == AS_OF for r in result.tool_calls)
    assert result.tool_calls[1].args == {"symbol": "BTC"}

    # Second request carries the assistant tool_calls and one tool message per call, in order.
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "tool"]
    assert [c["id"] for c in sent[2]["tool_calls"]] == ["a", "b"]
    assert [m["tool_call_id"] for m in sent[3:]] == ["a", "b"]
    assert tool_payload(sent[3])["source_id"] == "mirkaso.macro"
    assert result.messages[-1] == {"role": "assistant", "content": result.answer}

    first = client.requests[0]
    assert first["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in first["tools"]} >= {"macro_get_regime"}
    assert first["reasoning_effort"] == "low"
    assert first["max_tokens"] == 3000


async def test_tool_exception_becomes_structured_error(
    provider: FakeProvider, caplog: pytest.LogCaptureFixture
) -> None:
    provider.fail_with = RuntimeError(SECRET)
    client = ScriptedClient(
        reply(tool_calls=[call("macro_get_regime")]),
        reply("Macro data is unavailable right now."),
    )

    with caplog.at_level(logging.INFO, logger="mirkaso_copilot.agent"):
        result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "final"
    assert result.answer == "Macro data is unavailable right now."
    (record,) = result.tool_calls
    assert (record.ok, record.error_code, record.source_id) == (False, "internal_error", None)
    (msg,) = tool_messages(result.messages)
    assert tool_payload(msg) == {
        "error": {"code": "internal_error", "message": "The tool failed unexpectedly."}
    }
    assert SECRET not in json.dumps(result.messages)
    agent_logs = [r.getMessage() for r in caplog.records if r.name == "mirkaso_copilot.agent"]
    assert agent_logs and not any(SECRET in line for line in agent_logs)


async def test_agent_level_exception_is_caught(
    provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any) -> dict[str, Any]:
        raise RuntimeError(SECRET)

    monkeypatch.setattr("mirkaso_copilot.agent.execute_tool", boom)
    client = ScriptedClient(reply(tool_calls=[call("macro_get_regime")]), reply("Sorry."))

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "final"
    assert result.tool_calls[0].error_code == "internal_error"
    assert SECRET not in json.dumps(result.messages)


async def test_tool_timeout(provider: FakeProvider) -> None:
    async def hang() -> dict[str, Any]:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    provider.macro_regime = hang  # type: ignore[method-assign]
    client = ScriptedClient(
        reply(tool_calls=[call("macro_get_regime")]), reply("The data source timed out.")
    )
    config = AgentConfig(per_tool_timeout_s=0.05)

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client, config=config)

    assert result.stop_reason == "final"
    (record,) = result.tool_calls
    assert (record.ok, record.error_code) == (False, "timeout")
    assert record.latency_ms < 1000
    assert tool_payload(tool_messages(result.messages)[0])["error"]["code"] == "timeout"


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"text"'])
async def test_bad_json_arguments(provider: FakeProvider, bad: str) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("coin_get_derivatives", bad)]), reply("I could not look that up.")
    )

    result = await run_agent("BTC?", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "final"
    (record,) = result.tool_calls
    assert (record.args, record.ok, record.error_code) == (None, False, "invalid_arguments")
    assert tool_payload(tool_messages(result.messages)[0]) == {
        "error": {"code": "invalid_arguments", "message": "Arguments must be a JSON object."}
    }
    assert provider.calls == []


async def test_unknown_tool(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("db_run_sql", {"q": "select 1"})]), reply("I cannot do that.")
    )

    result = await run_agent("Run SQL", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "final"
    assert result.tool_calls[0].error_code == "unknown_tool"
    assert tool_payload(tool_messages(result.messages)[0])["error"]["code"] == "unknown_tool"
    assert provider.calls == []


async def test_duplicate_call_guard(provider: FakeProvider) -> None:
    first = '{"symbol": "BTC", "include_history": false}'
    # Same call with keys reordered and different whitespace: still a duplicate.
    again = '{"include_history":false,"symbol":"BTC"}'
    client = ScriptedClient(
        reply(tool_calls=[call("coin_get_derivatives", first)]),
        reply(tool_calls=[call("coin_get_derivatives", again, "c2")]),
        reply("BTC data as above."),
    )

    result = await run_agent("BTC?", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "final"
    assert [(r.ok, r.error_code) for r in result.tool_calls] == [
        (True, None),
        (False, "duplicate_call"),
    ]
    assert [name for name, _ in provider.calls] == ["coin_derivatives"]
    second = tool_messages(result.messages)[1]
    assert second["tool_call_id"] == "c2"
    assert tool_payload(second)["error"]["code"] == "duplicate_call"


async def test_duplicate_within_one_parallel_batch(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("macro_get_regime", {}, "a"), call("macro_get_regime", {}, "b")]),
        reply("Done."),
    )

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    assert [r.error_code for r in result.tool_calls] == [None, "duplicate_call"]
    assert provider.calls == [("macro_regime", {})]


async def test_duplicate_detection_applies_defaults(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("macro_get_regime", {}, "a")]),
        # Explicit defaults request the same data as omitting them.
        reply(tool_calls=[call("macro_get_regime", {"history_days": 7}, "b")]),
        reply(tool_calls=[call("macro_get_regime", {"include_history": True}, "c")]),
        reply("Done."),
    )

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "final"
    assert [r.error_code for r in result.tool_calls] == [None, "duplicate_call", None]
    assert [name for name, _ in provider.calls] == [
        "macro_regime",
        "macro_regime",
        "macro_regime_history",
    ]


async def test_invalid_arguments_are_never_duplicates(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("macro_get_regime", {"history_days": 99}, "a")]),
        reply(tool_calls=[call("macro_get_regime", {"history_days": 99}, "b")]),
        reply("Done."),
    )

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    assert [r.error_code for r in result.tool_calls] == ["invalid_arguments"] * 2


def _no_digits(text: str | None) -> bool:
    return text is not None and not re.search(r"\d", text)


async def test_max_steps_stops_with_fallback(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("coin_get_derivatives", {"symbol": "BTC"}, "c1")]),
        reply(tool_calls=[call("coin_get_derivatives", {"symbol": "ETH"}, "c2")]),
        reply(tool_calls=[call("coin_get_derivatives", {"symbol": "SOL"}, "c3")]),
    )
    config = AgentConfig(max_steps=3)

    result = await run_agent("Coins?", provider=provider, ctx=CTX, client=client, config=config)

    assert result.stop_reason == "max_steps"
    assert result.steps == 3
    assert result.answer == FALLBACK_ANSWERS["max_steps"]
    assert _no_digits(result.answer)
    # The tool calls asked for in the last step are not executed.
    assert [r.args for r in result.tool_calls] == [{"symbol": "BTC"}, {"symbol": "ETH"}]
    assert client.responses == []


async def test_max_tool_calls_stops_with_fallback(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(tool_calls=[call("macro_get_regime", {}, "a"), call("sectors_get_indices", {}, "b")]),
        reply(tool_calls=[call("coin_get_derivatives", {"symbol": "BTC"}, "c")]),
    )
    config = AgentConfig(max_tool_calls_total=2)

    result = await run_agent("All?", provider=provider, ctx=CTX, client=client, config=config)

    assert result.stop_reason == "max_tool_calls"
    assert len(result.tool_calls) == 2
    assert _no_digits(result.answer)


async def test_max_total_tokens_stops_with_fallback(provider: FakeProvider) -> None:
    big = {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
    client = ScriptedClient(reply(tool_calls=[call("macro_get_regime")], usage=big))
    config = AgentConfig(max_total_tokens=1000)

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client, config=config)

    assert result.stop_reason == "max_tokens"
    assert result.tool_calls == []
    assert _no_digits(result.answer)


async def test_finish_reason_length_is_truncated(provider: FakeProvider) -> None:
    client = ScriptedClient(reply("The regime is", finish_reason="length"))

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    assert result.stop_reason == "truncated"
    assert result.answer is None
    assert result.warnings == ["finish_reason_length"]
    assert result.steps == 1


async def test_usage_aggregation_includes_reasoning(provider: FakeProvider) -> None:
    client = ScriptedClient(
        reply(
            tool_calls=[call("macro_get_regime")],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        ),
        # No total_tokens and top-level reasoning_tokens: total falls back to the sum.
        reply(
            "Done.", usage={"prompt_tokens": 200, "completion_tokens": 10, "reasoning_tokens": 4}
        ),
    )

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=client)

    usage = result.usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (300, 50)
    assert (usage.reasoning_tokens, usage.total_tokens) == (34, 350)


async def test_global_timeout(provider: FakeProvider) -> None:
    client = ScriptedClient(reply("never"), delay=10)
    config = AgentConfig(total_timeout_s=0.05)

    result = await asyncio.wait_for(
        run_agent("Regime?", provider=provider, ctx=CTX, client=client, config=config), 2
    )

    assert result.stop_reason == "timeout"
    assert result.answer == FALLBACK_ANSWERS["timeout"]
    assert _no_digits(result.answer)
    assert result.latency_ms < 1000


async def test_model_error_stops_without_raising(provider: FakeProvider) -> None:
    class Failing:
        async def chat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(SECRET)

    result = await run_agent("Regime?", provider=provider, ctx=CTX, client=Failing())

    assert result.stop_reason == "error"
    assert result.warnings == ["model_request_failed"]
    assert _no_digits(result.answer)


@pytest.mark.parametrize("effort", ["minimal", "max", ""])
def test_config_rejects_unknown_reasoning_effort(effort: str) -> None:
    with pytest.raises(ValueError):
        AgentConfig(reasoning_effort=effort)


def test_fallback_answers_have_no_numbers() -> None:
    assert all(_no_digits(text) for text in FALLBACK_ANSWERS.values())


async def test_model_error_detail_stays_out_of_messages_and_logs(
    provider: FakeProvider, caplog: pytest.LogCaptureFixture
) -> None:
    body = json.dumps({"error": {"message": f"Invalid API key ({SECRET})"}})

    class Unauthorized:
        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            raise MetaAPIError("Meta API returned HTTP 401", status_code=401, body=body)

    with caplog.at_level(logging.DEBUG):
        result = await run_agent("q", provider=provider, ctx=CTX, client=Unauthorized())

    assert result.stop_reason == "error"
    assert result.warnings == ["model_request_failed"]
    assert result.warnings_detail == [
        {
            "warning": "model_request_failed",
            "error_type": "MetaAPIError",
            "status_code": 401,
            "message": f"Invalid API key ({SECRET})",
        }
    ]
    assert SECRET not in json.dumps(result.messages)
    assert SECRET not in caplog.text


async def test_successful_run_has_no_warning_details(provider: FakeProvider) -> None:
    result = await run_agent("q", provider=provider, ctx=CTX, client=ScriptedClient(reply("ok")))
    assert result.warnings_detail == []
