"""Two-phase structured runs with a scripted fake model client. No network."""

import json
from typing import Any

from mirkaso_copilot.agent import STRUCTURED_INSTRUCTION, AgentConfig, run_agent
from mirkaso_copilot.answer import answer_json_schema
from mirkaso_copilot.meta_client import MetaAPIError
from mirkaso_copilot.providers import CallerContext, Tier
from tests.conftest import AS_OF, FakeProvider, ScriptedClient, call, reply

CTX = CallerContext(tier=Tier.TRADER)
USAGE_A = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
           "completion_tokens_details": {"reasoning_tokens": 5}}  # fmt: skip
USAGE_B = {"prompt_tokens": 400, "completion_tokens": 90, "total_tokens": 490,
           "completion_tokens_details": {"reasoning_tokens": 30}}  # fmt: skip


def value(vid: str, call_id: str, path: str, kind: str) -> dict[str, Any]:
    return {"id": vid, "tool_call_id": call_id, "path": path, "kind": kind, "decimals": None}


STRUCTURED = {
    "summary": "The regime is {v1} with score {v2} as of {v3}.",
    "claims": [{"text": "Liquidations totalled {v4} over 24h, {v5} of them longs."}],
    "values": [
        value("v1", "m", "/data/regime", "text"),
        value("v2", "m", "/data/score", "number"),
        value("v3", "m", "/as_of", "timestamp"),
        value("v4", "l", "/data/summary/total_usd", "usd"),
        value("v5", "l", "/data/summary/long_pct", "percent"),
    ],
    "caveats": [],
}


def client(phase_b: dict[str, Any]) -> ScriptedClient:
    return ScriptedClient(
        reply(
            tool_calls=[
                call("macro_get_regime", {}, "m"),
                call("market_get_liquidations", {"period": "24h"}, "l"),
            ],
            usage=USAGE_A,
        ),
        reply("The regime is Neutral.", usage=USAGE_A),
        phase_b,
    )


async def run(provider: FakeProvider, c: ScriptedClient, **kwargs: Any):
    return await run_agent("Regime?", provider=provider, ctx=CTX, client=c, **kwargs)


async def test_structured_run_resolves_values(provider: FakeProvider) -> None:
    c = client(reply(json.dumps(STRUCTURED), usage=USAGE_B))

    result = await run(provider, c, structured=True)

    assert result.stop_reason == "final"
    assert result.answer == "The regime is Neutral."  # phase A answer is kept
    assert result.steps == 2  # phase B is not a tool-loop step
    assert result.warnings == []
    assert result.structured is not None and result.structured.summary == STRUCTURED["summary"]
    assert result.resolved is not None and result.resolution_errors == []
    assert (
        result.resolved.summary == "The regime is Neutral with score 57 as of 2026-10-26 12:00 UTC."
    )
    assert result.resolved.claims == ["Liquidations totalled $3.6M over 24h, 69.4% of them longs."]
    assert [(v.id, v.tool, v.source_id, v.as_of) for v in result.resolved.values] == [
        ("v1", "macro_get_regime", "mirkaso.macro", AS_OF),
        ("v2", "macro_get_regime", "mirkaso.macro", AS_OF),
        ("v3", "macro_get_regime", "mirkaso.macro", AS_OF),
        ("v4", "market_get_liquidations", "mirkaso.liquidations", AS_OF),
        ("v5", "market_get_liquidations", "mirkaso.liquidations", AS_OF),
    ]
    assert [r.tool_call_id for r in result.tool_calls] == ["m", "l"]


async def test_phase_b_has_no_tools_and_a_strict_response_format(provider: FakeProvider) -> None:
    c = client(reply(json.dumps(STRUCTURED), usage=USAGE_B))

    result = await run(provider, c, structured=True, config=AgentConfig(reasoning_effort="medium"))

    phase_a, phase_b = c.requests[0], c.requests[2]
    assert "response_format" not in phase_a and phase_a["tools"]
    assert "tools" not in phase_b and "tool_choice" not in phase_b
    assert phase_b["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_answer",
            "strict": True,
            "schema": answer_json_schema(),
        },
    }
    assert (phase_b["reasoning_effort"], phase_b["max_tokens"]) == ("medium", 3000)
    # The transcript (with the tool results the pointers refer to) plus the instruction.
    roles = [m["role"] for m in phase_b["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
    assert phase_b["messages"][-1]["content"] == STRUCTURED_INSTRUCTION
    assert result.messages[-1] == {"role": "assistant", "content": json.dumps(STRUCTURED)}


async def test_usage_and_latency_include_phase_b(provider: FakeProvider) -> None:
    c = client(reply(json.dumps(STRUCTURED), usage=USAGE_B))

    result = await run(provider, c, structured=True)

    u = result.usage
    assert (u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.total_tokens) == (
        100 + 100 + 400, 20 + 20 + 90, 5 + 5 + 30, 120 + 120 + 490,
    )  # fmt: skip
    assert u.total_tokens == u.prompt_tokens + u.completion_tokens


async def test_latency_includes_phase_b(provider: FakeProvider) -> None:
    c = client(reply(json.dumps(STRUCTURED), usage=USAGE_B))
    c.delay = 0.05  # every request; phase B is the third
    result = await run(provider, c, structured=True)
    assert result.latency_ms >= 3 * 50


async def test_invalid_json_is_structured_failed(provider: FakeProvider) -> None:
    c = client(reply("The regime is Neutral (57).", usage=USAGE_B))

    result = await run(provider, c, structured=True)

    assert result.stop_reason == "final"
    assert result.answer == "The regime is Neutral."
    assert result.structured is None and result.resolved is None
    assert result.resolution_errors == []
    assert result.warnings == ["structured_failed"]
    assert result.usage.total_tokens == 120 + 120 + 490  # the failed reply still cost tokens


async def test_schema_violation_is_structured_failed(provider: FakeProvider) -> None:
    bad = {**STRUCTURED, "values": [{**STRUCTURED["values"][0], "kind": "euro"}]}
    result = await run(provider, client(reply(json.dumps(bad))), structured=True)
    assert result.structured is None
    assert result.warnings == ["structured_failed"]


async def test_truncated_phase_b_is_structured_failed(provider: FakeProvider) -> None:
    c = client(reply('{"summary": "The', finish_reason="length"))
    result = await run(provider, c, structured=True)
    assert (result.stop_reason, result.structured) == ("final", None)
    assert result.warnings == ["structured_failed"]


async def test_failed_phase_b_request_is_structured_failed(provider: FakeProvider) -> None:
    class Client(ScriptedClient):
        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            if "response_format" in kwargs:
                raise MetaAPIError("Meta API returned HTTP 400", status_code=400, body="{}")
            return await super().chat(messages, **kwargs)

    c = Client(
        reply(tool_calls=[call("macro_get_regime", {}, "m")]), reply("The regime is Neutral.")
    )
    result = await run(provider, c, structured=True)
    assert (result.stop_reason, result.answer) == ("final", "The regime is Neutral.")
    assert result.warnings == ["structured_failed"]
    assert result.warnings_detail == [
        {"warning": "structured_failed", "error_type": "MetaAPIError", "status_code": 400,
         "message": None}
    ]  # fmt: skip


async def test_resolution_errors_are_surfaced(provider: FakeProvider) -> None:
    bad = {
        **STRUCTURED,
        "summary": "Score {v2}, missing {v9}.",
        "claims": [],
        "values": [{**STRUCTURED["values"][1], "tool_call_id": "zzz"}],
    }
    result = await run(provider, client(reply(json.dumps(bad))), structured=True)
    assert result.stop_reason == "final" and result.structured is not None
    assert [(e.code, e.value_id) for e in result.resolution_errors] == [
        ("unknown_tool_call", "v2"),
        ("unknown_placeholder", "v9"),
    ]
    assert result.resolved is not None
    assert result.resolution_errors == result.resolved.errors
    assert result.warnings == []


async def test_default_run_makes_no_phase_b_request(provider: FakeProvider) -> None:
    c = client(reply(json.dumps(STRUCTURED)))
    result = await run(provider, c)
    assert len(c.requests) == 2
    assert result.structured is None and result.resolved is None


async def test_no_phase_b_after_a_non_final_stop(provider: FakeProvider) -> None:
    c = ScriptedClient(reply(None))  # empty answer -> stop_reason "error"
    result = await run(provider, c, structured=True)
    assert result.stop_reason == "error"
    assert len(c.requests) == 1
    assert result.structured is None
