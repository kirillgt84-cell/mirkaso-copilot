"""Structured runs: phase B, grounding guard, regeneration and safe refusal (card 10).

Scripted fake model client, no network.
"""

import asyncio
import json
import re
from typing import Any

import pytest

from mirkaso_copilot.agent import (
    CORRECTION_INTRO,
    FALLBACK_ANSWERS,
    LENGTH_PROBLEM,
    REFUSAL_ANSWER,
    SINGLE_PHASE_RULES,
    STRUCTURED_INSTRUCTION,
    AgentConfig,
    run_agent,
)
from mirkaso_copilot.answer import answer_json_schema
from mirkaso_copilot.meta_client import MetaAPIError
from mirkaso_copilot.providers import CallerContext, Tier
from tests.conftest import AS_OF, FakeProvider, ScriptedClient, call, reply

CTX = CallerContext(tier=Tier.TRADER)
USAGE_A = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
           "completion_tokens_details": {"reasoning_tokens": 5}}  # fmt: skip
USAGE_B = {"prompt_tokens": 400, "completion_tokens": 90, "total_tokens": 490,
           "completion_tokens_details": {"reasoning_tokens": 30}}  # fmt: skip
USAGE_C = {"prompt_tokens": 700, "completion_tokens": 60, "total_tokens": 760,
           "completion_tokens_details": {"reasoning_tokens": 10}}  # fmt: skip
PLAIN_ANSWER = "The regime is Neutral (57) and BTC open interest is 12B."


def value(vid: str, call_ref: str, path: str, kind: str) -> dict[str, Any]:
    return {"id": vid, "call_ref": call_ref, "path": path, "kind": kind, "decimals": None}


GOOD = {
    "summary": "The regime is {v1} with score {v2} as of {v3}.",
    "claims": [{"text": "BTC open interest is {v4}, {v5} over 24h."}],
    "values": [
        value("v1", "t1", "/data/regime", "text"),
        value("v2", "t1", "/data/score", "number"),
        value("v3", "t1", "/as_of", "timestamp"),
        value("v4", "t2", "/data/open_interest_usd", "usd"),
        value("v5", "t2", "/data/oi_change_24h_pct", "percent"),
    ],
    "caveats": [],
}
GOOD_TEXT = (
    "The regime is Neutral with score 57 as of 2026-10-26 12:00 UTC.\n"
    "- BTC open interest is $12B, +4.2% over 24h."
)
# Literal number in a claim and a reference to a call that does not exist.
BAD = {
    **GOOD,
    "claims": [{"text": "Funding is 8.5e-05 over 24h."}],
    "values": [*GOOD["values"][:3], value("v4", "t9", "/data/open_interest_usd", "usd")],
}


def answer_reply(body: dict[str, Any] | str, usage: dict[str, Any] = USAGE_B, **kw: Any):
    return reply(body if isinstance(body, str) else json.dumps(body), usage=usage, **kw)


def client(*structured_replies: dict[str, Any]) -> ScriptedClient:
    return ScriptedClient(
        reply(
            tool_calls=[
                call("macro_get_regime", {}, "call_m"),
                call("coin_get_derivatives", {"symbol": "BTC"}, "call_d"),
            ],
            usage=USAGE_A,
        ),
        reply(PLAIN_ANSWER, usage=USAGE_A),
        *structured_replies,
    )


async def run(provider: FakeProvider, c: ScriptedClient, **kwargs: Any):
    kwargs.setdefault("structured", True)
    return await run_agent("Regime?", provider=provider, ctx=CTX, client=c, **kwargs)


def assert_refusal(result: Any, reason: str) -> None:
    assert result.stop_reason == "final"
    assert (result.verified, result.refusal_reason) == (False, reason)
    assert result.answer == REFUSAL_ANSWER
    assert not re.search(r"\d", result.answer)  # a refusal never carries a number
    assert result.structured is None and result.resolved is None


# --- phase B request -----------------------------------------------------------------


async def test_verified_answer(provider: FakeProvider) -> None:
    result = await run(provider, client(answer_reply(GOOD)))

    assert result.stop_reason == "final"
    assert (result.verified, result.refusal_reason) == (True, None)
    assert result.answer == GOOD_TEXT  # the resolved text, not the phase A answer
    assert result.resolved is not None and result.resolved.text == GOOD_TEXT
    assert result.structured is not None and result.structured.summary == GOOD["summary"]
    assert (result.guard_attempts, result.guard_rejections, result.rejected_reasons) == (1, 0, [])
    assert result.resolution_errors == [] and result.warnings == []
    assert result.steps == 2  # phase B is not a tool-loop step
    assert [(v.id, v.call_ref, v.tool_call_id, v.source_id, v.as_of)
            for v in result.resolved.values][::3] == [
        ("v1", "t1", "call_m", "mirkaso.macro", AS_OF),
        ("v4", "t2", "call_d", "mirkaso.derivatives", AS_OF),
    ]  # fmt: skip


async def test_phase_b_request_shape(provider: FakeProvider) -> None:
    c = client(answer_reply(GOOD))

    await run(provider, c, config=AgentConfig(reasoning_effort="medium"))

    phase_a, phase_b = c.requests[0], c.requests[2]
    assert "response_format" not in phase_a and phase_a["tools"]
    assert phase_a["max_tokens"] == 3000
    assert "tools" not in phase_b and "tool_choice" not in phase_b
    assert phase_b["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_answer",
            "strict": True,
            "schema": answer_json_schema(),
        },
    }
    # Its own token limit, separate from request_max_tokens.
    assert (phase_b["reasoning_effort"], phase_b["max_tokens"]) == ("medium", 6000)
    roles = [m["role"] for m in phase_b["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]


async def test_instruction_lists_every_call_with_its_alias(provider: FakeProvider) -> None:
    provider.derivatives = None  # coin_get_derivatives -> not_found
    c = ScriptedClient(
        reply(
            tool_calls=[
                call("macro_get_regime", {}, "a"),
                call("coin_get_derivatives", {"symbol": "NOPE"}, "b"),
            ]
        ),
        reply(PLAIN_ANSWER),
        answer_reply({**GOOD, "claims": [], "values": GOOD["values"][:3]}),
    )

    await run(provider, c)

    instruction = c.requests[2]["messages"][-1]["content"]
    assert instruction.startswith(STRUCTURED_INSTRUCTION)
    assert instruction.endswith(
        "Tool calls, by alias (use the alias as call_ref):\n"
        f"t1 = macro_get_regime({{}}) ok, source mirkaso.macro, as_of {AS_OF}\n"
        't2 = coin_get_derivatives({"symbol": "NOPE"}) error not_found'
    )
    assert "at most 6 claims and at most 24 values" in instruction


async def test_tool_name_as_call_ref_still_resolves(provider: FakeProvider) -> None:
    """The live failure: the model wrote the tool name instead of the id."""
    by_name = {
        **GOOD,
        "values": [
            {**v, "call_ref": "macro_get_regime" if v["call_ref"] == "t1" else "call_d"}
            for v in GOOD["values"]
        ],
    }
    result = await run(provider, client(answer_reply(by_name)))
    assert result.verified is True
    assert result.answer == GOOD_TEXT


# --- regeneration --------------------------------------------------------------------


async def test_regeneration_succeeds_on_the_second_try(provider: FakeProvider) -> None:
    c = client(answer_reply(BAD), answer_reply(GOOD, usage=USAGE_C))

    result = await run(provider, c)

    assert (result.verified, result.answer) == (True, GOOD_TEXT)
    assert (result.guard_attempts, result.guard_rejections) == (2, 1)
    assert result.rejected_reasons == [
        [
            "v4: unknown_call_ref (No call 't9'.)",
            "literal number '8.5e-05' in claim 1; use a placeholder",
        ]
    ]
    # The regeneration request: transcript + rejected reply + a correction naming the problems.
    regen = c.requests[3]
    assert "tools" not in regen and regen["response_format"]["json_schema"]["strict"] is True
    roles = [m["role"] for m in regen["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user",
                     "assistant", "user"]  # fmt: skip
    assert regen["messages"][-2]["content"] == json.dumps(BAD)
    correction = regen["messages"][-1]["content"]
    assert correction.startswith(CORRECTION_INTRO)
    assert "- v4: unknown_call_ref (No call 't9'.)" in correction
    assert "- literal number '8.5e-05' in claim 1; use a placeholder" in correction
    assert [p.name for p in result.phases] == ["tool", "structured_1", "structured_2"]


async def test_invalid_json_then_valid(provider: FakeProvider) -> None:
    c = client(answer_reply("The regime is Neutral (57)."), answer_reply(GOOD))
    result = await run(provider, c)
    assert result.verified is True
    (reasons,) = result.rejected_reasons
    assert reasons[0].startswith("invalid answer: Invalid JSON")
    assert "Neutral (57)" not in c.requests[3]["messages"][-1]["content"]  # no reply text echoed


async def test_too_many_claims_is_an_invalid_answer(provider: FakeProvider) -> None:
    long = {**GOOD, "claims": [{"text": "Regime {v1}."}] * 7}
    result = await run(provider, client(answer_reply(long), answer_reply(GOOD)))
    assert result.verified is True
    assert result.rejected_reasons == [
        ["invalid answer: Value error, at most 6 claims allowed, got 7"]
    ]


async def test_advice_wording_is_rejected(provider: FakeProvider) -> None:
    advice = {**GOOD, "summary": "The regime is {v1}; you should buy."}
    result = await run(provider, client(answer_reply(advice), answer_reply(GOOD)))
    assert result.verified is True
    assert result.rejected_reasons == [
        ["advice wording 'you should buy' in summary; describe the data only"]
    ]


async def test_injected_output_guard(provider: FakeProvider) -> None:
    def product_guard(text: str) -> list[str]:
        return ["Neutral"] if "Neutral" in text else []

    result = await run(provider, client(answer_reply(GOOD), answer_reply(GOOD)),
                       output_guard=product_guard)  # fmt: skip
    assert_refusal(result, "guard_failed")
    assert result.rejected_reasons[0] == [
        "advice wording 'Neutral' in summary; describe the data only"
    ]


# --- safe refusal --------------------------------------------------------------------


async def test_refusal_after_two_guard_failures(provider: FakeProvider) -> None:
    c = client(answer_reply(BAD), answer_reply(BAD))

    result = await run(provider, c)

    assert_refusal(result, "guard_failed")
    assert (result.guard_attempts, result.guard_rejections) == (2, 2)
    assert len(result.rejected_reasons) == 2
    assert [e.code for e in result.resolution_errors] == ["unknown_call_ref"]  # last attempt
    assert len(c.requests) == 4  # exactly one regeneration


async def test_never_falls_back_to_the_plain_answer(provider: FakeProvider) -> None:
    result = await run(provider, client(answer_reply("not json"), answer_reply("still not")))
    assert_refusal(result, "structured_invalid")
    assert PLAIN_ANSWER not in result.answer
    assert result.messages[5]["content"] == PLAIN_ANSWER  # it is only in the transcript


async def test_length_cut_twice_is_a_refusal(provider: FakeProvider) -> None:
    cut = answer_reply('{"summary": "The', finish_reason="length")
    c = client(cut, answer_reply('{"summary": "The', finish_reason="length"))

    result = await run(provider, c)

    assert_refusal(result, "structured_length")
    assert result.rejected_reasons == [[LENGTH_PROBLEM], [LENGTH_PROBLEM]]
    assert LENGTH_PROBLEM in c.requests[3]["messages"][-1]["content"]


async def test_failed_request_is_a_refusal_without_retry(provider: FakeProvider) -> None:
    class Client(ScriptedClient):
        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            if "response_format" in kwargs:
                self.requests.append(kwargs)
                raise MetaAPIError("Meta API returned HTTP 400", status_code=400, body="{}")
            return await super().chat(messages, **kwargs)

    c = Client(reply(tool_calls=[call("macro_get_regime", {}, "m")]), reply(PLAIN_ANSWER))

    result = await run(provider, c)

    assert_refusal(result, "structured_request_failed")
    assert len(c.requests) == 3  # an HTTP error is not retried
    assert result.guard_attempts == 0
    assert result.warnings == ["structured_failed"]
    assert result.warnings_detail == [
        {"warning": "structured_failed", "error_type": "MetaAPIError", "status_code": 400,
         "message": None}
    ]  # fmt: skip


async def test_structured_phase_stays_within_the_total_timeout(provider: FakeProvider) -> None:
    class Slow(ScriptedClient):
        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            if "response_format" in kwargs:
                await asyncio.sleep(5)
            return await super().chat(messages, **kwargs)

    c = Slow(reply(tool_calls=[call("macro_get_regime", {}, "m")]), reply(PLAIN_ANSWER))
    result = await asyncio.wait_for(run(provider, c, config=AgentConfig(total_timeout_s=0.3)), 2)
    assert_refusal(result, "timeout")
    assert result.latency_ms < 1000


async def test_no_structured_phase_after_a_non_final_stop(provider: FakeProvider) -> None:
    c = ScriptedClient(reply(None))  # empty answer -> stop_reason "error"
    result = await run(provider, c)
    assert result.stop_reason == "error"
    assert result.answer == FALLBACK_ANSWERS["error"]
    assert (result.verified, result.refusal_reason, result.guard_attempts) == (False, None, 0)
    assert len(c.requests) == 1


async def test_default_run_is_unchanged(provider: FakeProvider) -> None:
    c = client(answer_reply(GOOD))
    result = await run(provider, c, structured=False)
    assert len(c.requests) == 2
    assert result.answer == PLAIN_ANSWER
    assert (result.verified, result.structured, result.guard_attempts) == (None, None, 0)
    assert [p.name for p in result.phases] == ["tool"]


# --- usage and latency ---------------------------------------------------------------


async def test_usage_is_aggregated_across_all_requests(provider: FakeProvider) -> None:
    c = client(answer_reply(BAD), answer_reply(GOOD, usage=USAGE_C))

    result = await run(provider, c)

    u = result.usage
    assert (u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.total_tokens) == (
        100 + 100 + 400 + 700, 20 + 20 + 90 + 60, 5 + 5 + 30 + 10, 120 + 120 + 490 + 760,
    )  # fmt: skip
    tool, first, second = result.phases
    assert (tool.usage.prompt_tokens, tool.usage.total_tokens) == (200, 240)
    assert (first.usage.completion_tokens, first.usage.reasoning_tokens) == (90, 30)
    assert second.usage.total_tokens == 760
    assert sum(p.usage.total_tokens for p in result.phases) == u.total_tokens


async def test_latency_per_phase(provider: FakeProvider) -> None:
    c = client(answer_reply(GOOD))
    c.delay = 0.05
    result = await run(provider, c)
    tool, structured = result.phases
    assert tool.latency_ms >= 2 * 50 and structured.latency_ms >= 50
    assert result.latency_ms >= tool.latency_ms + structured.latency_ms - 1


# --- single-phase experiment ---------------------------------------------------------


def single_client(*final: dict[str, Any]) -> ScriptedClient:
    return ScriptedClient(
        reply(
            tool_calls=[
                call("macro_get_regime", {}, "call_m"),
                call("coin_get_derivatives", {"symbol": "BTC"}, "call_d"),
            ],
            usage=USAGE_A,
        ),
        *final,
    )


async def test_single_phase_parses_the_final_loop_message(provider: FakeProvider) -> None:
    c = single_client(answer_reply(GOOD))

    result = await run(provider, c, structured_mode="single_phase")

    assert (result.verified, result.answer) == (True, GOOD_TEXT)
    assert len(c.requests) == 2  # no extra phase B request
    for request in c.requests:  # schema and tools together in every loop request
        assert request["tools"] and request["tool_choice"] == "auto"
        assert request["response_format"]["json_schema"]["strict"] is True
        assert request["max_tokens"] == 6000
    assert c.requests[0]["messages"][0]["content"].endswith(SINGLE_PHASE_RULES)
    assert [p.name for p in result.phases] == ["tool"]
    assert result.guard_attempts == 1


async def test_single_phase_regeneration_shows_the_call_list(provider: FakeProvider) -> None:
    c = single_client(answer_reply("not json"), answer_reply(GOOD))

    result = await run(provider, c, structured_mode="single_phase")

    assert result.verified is True
    regen = c.requests[2]
    assert "tools" not in regen and regen["response_format"]
    prompt = regen["messages"][-1]["content"]
    assert prompt.startswith(STRUCTURED_INSTRUCTION) and CORRECTION_INTRO in prompt
    assert "t2 = coin_get_derivatives" in prompt
    assert [p.name for p in result.phases] == ["tool", "structured_2"]


async def test_single_phase_rejected_by_the_api_is_reported_cleanly(
    provider: FakeProvider,
) -> None:
    class Rejecting(ScriptedClient):
        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            if kwargs.get("tools") and "response_format" in kwargs:
                raise MetaAPIError(
                    "Meta API returned HTTP 400",
                    status_code=400,
                    body='{"error": {"message": "tools with response_format"}}',
                )
            return await super().chat(messages, **kwargs)  # fmt: skip

    result = await run(provider, Rejecting(), structured_mode="single_phase")

    assert result.stop_reason == "error"
    assert result.answer == FALLBACK_ANSWERS["error"]
    assert result.verified is False
    assert result.warnings == ["model_request_failed"]
    assert result.warnings_detail[0]["status_code"] == 400
    assert result.warnings_detail[0]["message"] == "tools with response_format"


async def test_unknown_structured_mode(provider: FakeProvider) -> None:
    with pytest.raises(ValueError, match="structured_mode"):
        await run(provider, client(), structured_mode="three_phase")
