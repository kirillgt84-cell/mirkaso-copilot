"""resolve_answer: pointer lookup, formatting per kind and typed errors. Pure, no network."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from mirkaso_copilot.agent import ToolCallRecord
from mirkaso_copilot.answer import StructuredAnswer, answer_json_schema
from mirkaso_copilot.fixture_provider import FixtureProvider
from mirkaso_copilot.providers import CallerContext, Tier
from mirkaso_copilot.resolve import ResolutionErrorCode, format_value, resolve_answer
from mirkaso_copilot.tools.registry import execute_tool

AS_OF = "2026-10-26T12:00:00Z"
LIQUIDATIONS = {
    "as_of": AS_OF,
    "source_id": "mirkaso.liquidations",
    "gated": False,
    "gated_fields": [],
    "newest_event_at": "2026-10-26T11:59:18Z",
    "data": {
        "period": "24h",
        "summary": {"total_usd": 56867100.0, "long_usd": 37085600.0, "short_usd": 19781500.0,
                    "long_pct": 65.2, "short_pct": 34.8, "count": 6114},
        "top_coins": [{"symbol": "BTCUSDT", "total_usd": 12429900.0, "count": 1650}],
    },
}  # fmt: skip
SECTORS = {
    "as_of": AS_OF,
    "source_id": "mirkaso.sectors",
    "gated": False,
    "gated_fields": [],
    "data": {
        "sectors": [
            {"slug": "ai", "sector_score": 74.2, "sector_score_delta_24h": 6.8,
             "price_change_24h_pct": 4.9, "relative_strength_btc": 1.41, "breadth_ratio": 0.82},
            {"slug": "meme", "sector_score": 47.3, "sector_score_delta_24h": -5.9,
             "price_change_24h_pct": -2.8, "relative_strength_btc": 0.74, "breadth_ratio": 0.31},
        ]
    },
}  # fmt: skip
NOT_FOUND = {"error": {"code": "not_found", "message": "No data found for the requested item."}}


def record(call_id: str, name: str, result: dict[str, Any]) -> ToolCallRecord:
    return ToolCallRecord(
        step=1,
        tool_call_id=call_id,
        name=name,
        args={},
        ok="error" not in result,
        error_code=result["error"]["code"] if "error" in result else None,
        latency_ms=1.0,
        source_id=result.get("source_id"),
        as_of=result.get("as_of"),
    )


CALLS = [
    ("c1", "market_get_liquidations", LIQUIDATIONS),
    ("c2", "sectors_get_indices", SECTORS),
    ("c3", "coin_get_derivatives", NOT_FOUND),
]
RECORDS = [record(*c) for c in CALLS]
MESSAGES = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": None, "tool_calls": []},
    *({"role": "tool", "tool_call_id": cid, "content": json.dumps(r)} for cid, _, r in CALLS),
]


def ref(vid: str, call_ref: str, path: str, kind: str, decimals: int | None = None) -> dict:
    return {"id": vid, "call_ref": call_ref, "path": path, "kind": kind, "decimals": decimals}


def answer(summary: str, values: list[dict], claims=(), caveats=()) -> StructuredAnswer:
    return StructuredAnswer.model_validate(
        {
            "summary": summary,
            "claims": [{"text": c} for c in claims],
            "values": values,
            "caveats": list(caveats),
        }
    )


def resolve(a: StructuredAnswer):
    return resolve_answer(a, RECORDS, MESSAGES)


def test_valid_answer_resolves_every_kind() -> None:
    a = answer(
        "{v1} was liquidated in the last 24h as of {v2}.",
        [
            ref("v1", "c1", "/data/summary/total_usd", "usd"),
            ref("v2", "c1", "/as_of", "timestamp"),
            ref("v3", "c1", "/data/summary/long_pct", "percent"),
            ref("v4", "c1", "/data/summary/count", "count"),
            ref("v5", "c2", "/data/sectors/0/relative_strength_btc", "ratio"),
            ref("v6", "c2", "/data/sectors/0/slug", "text"),
            ref("v7", "c2", "/data/sectors/0/price_change_24h_pct", "percent"),
            ref("v8", "c2", "/data/sectors/0/sector_score", "number"),
        ],
        claims=[
            "Longs were {v3} of {v4} liquidations.",
            "{v6} leads with score {v8}, {v7} over 24h and {v5} strength vs BTC.",
        ],
    )

    out = resolve(a)

    assert out.ok and out.errors == [] and out.unused_values == []
    assert out.summary == "$56.87M was liquidated in the last 24h as of 2026-10-26 12:00 UTC."
    assert out.claims == [
        "Longs were 65.2% of 6,114 liquidations.",
        "ai leads with score 74.2, +4.9% over 24h and 1.41 strength vs BTC.",
    ]
    assert out.text == "\n".join([out.summary, *(f"- {c}" for c in out.claims)])
    v1 = out.values[0]
    assert (v1.id, v1.value, v1.formatted, v1.tool, v1.source_id, v1.as_of, v1.path) == (
        "v1", 56867100.0, "$56.87M", "market_get_liquidations", "mirkaso.liquidations",
        AS_OF, "/data/summary/total_usd",
    )  # fmt: skip
    assert [v.formatted for v in out.values] == [
        "$56.87M", "2026-10-26 12:00 UTC", "65.2%", "6,114", "1.41", "ai", "+4.9%", "74.2",
    ]  # fmt: skip


def test_caveats_are_resolved_too() -> None:
    a = answer("x", [ref("v1", "c1", "/newest_event_at", "timestamp")], caveats=["Newest {v1}."])
    assert resolve(a).caveats == ["Newest 2026-10-26 11:59 UTC."]


@pytest.mark.parametrize(
    ("raw", "kind", "decimals", "path", "expected"),
    [
        (56867100.0, "usd", None, "/total_usd", "$56.87M"),
        (56867100.0, "usd", 1, "/total_usd", "$56.9M"),
        (12100000000.0, "usd", None, "/open_interest_usd", "$12.1B"),
        (1.5e12, "usd", None, "/x", "$1.5T"),
        (999_999_999.0, "usd", None, "/x", "$1B"),  # 999.999999M rounds up into billions
        (420000.0, "usd", None, "/x", "$420,000"),
        (67250.5, "usd", None, "/price", "$67,250.50"),
        (152.31, "usd", None, "/price", "$152.31"),
        (0.0000123, "usd", None, "/price", "$0.000012"),
        (-2500000.0, "usd", None, "/x", "-$2.5M"),
        (-2.7, "percent", None, "/price_change_1h_pct", "-2.7%"),
        (6.8, "percent", None, "/oi_change_1h_pct", "+6.8%"),
        (65.2, "percent", None, "/long_pct", "65.2%"),  # a share, not a change: no sign
        (0.0, "percent", None, "/price_change_24h_pct", "0%"),
        (4.91, "percent", 1, "/price_change_24h_pct", "+4.9%"),
        (6.8, "number", None, "/sector_score_delta_24h", "+6.8"),
        (-0.00021, "number", None, "/funding_rate", "-0.00021"),
        (8.5e-05, "number", None, "/funding_rate", "0.000085"),
        (12100000000.0, "number", None, "/open_interest_usd", "12,100,000,000"),
        (-0.0001, "number", 2, "/x", "0.00"),  # no "-0.00"
        (0.5, "ratio", None, "/breadth_ratio", "0.5"),
        (1.41, "ratio", 1, "/relative_strength_btc", "1.4"),
        (6114, "count", None, "/count", "6,114"),
        (6114.0, "count", 2, "/count", "6,114"),  # decimals do not apply to counts
        ("2026-10-26T13:46:14+02:00", "timestamp", None, "/as_of", "2026-10-26 11:46 UTC"),
        ("Neutral", "text", None, "/regime", "Neutral"),
    ],
)
def test_format_value(raw: Any, kind: str, decimals: int | None, path: str, expected: str) -> None:
    assert format_value(raw, kind, decimals, path) == expected


def errors(a: StructuredAnswer) -> list[tuple[str, str | None]]:
    return [(e.code.value, e.value_id) for e in resolve(a).errors]


def test_placeholder_without_value() -> None:
    a = answer("Total {v1}, longs {v2}.", [ref("v1", "c1", "/data/summary/total_usd", "usd")])
    out = resolve(a)
    assert errors(a) == [("unknown_placeholder", "v2")]
    assert out.summary == "Total $56.87M, longs {v2}."  # unresolved placeholders stay visible


def test_pointer_into_missing_path() -> None:
    a = answer(
        "{v1} {v2} {v3}",
        [
            ref("v1", "c1", "/data/summary/median_usd", "usd"),
            ref("v2", "c2", "/data/sectors/7/sector_score", "number"),
            ref("v3", "c2", "/data/sectors/01/sector_score", "number"),  # no leading zeros
        ],
    )
    assert errors(a) == [("missing_path", "v1"), ("missing_path", "v2"), ("missing_path", "v3")]
    assert resolve(a).summary == "{v1} {v2} {v3}"


async def test_open_interest_usd_resolves_as_usd_and_old_path_is_gone() -> None:
    provider, ctx = FixtureProvider(fixed_time=True), CallerContext(tier=Tier.TRADER)
    args = {"symbol": "BTC", "include_history": True}
    result = await execute_tool("coin_get_derivatives", args, provider, ctx)
    calls = [("c1", "coin_get_derivatives", result)]
    a = answer(
        "OI {v1}, last point {v2}.",
        [
            ref("v1", "t1", "/data/open_interest_usd", "usd"),
            ref("v2", "t1", "/data/history/23/open_interest_usd", "usd"),
        ],
    )
    out = resolve_with(calls, a)
    assert out.ok and out.summary == "OI $12.1B, last point $12.1B."

    for path in ("/data/open_interest", "/data/history/23/open_interest"):
        stale = resolve_with(calls, answer("{v1}", [ref("v1", "t1", path, "usd")]))
        assert [(e.code.value, e.value_id) for e in stale.errors] == [("missing_path", "v1")]


def test_invalid_pointer_syntax() -> None:
    a = answer("{v1}", [ref("v1", "c1", "data/summary/total_usd", "usd")])
    assert errors(a) == [("invalid_pointer", "v1")]


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("/data/period", "number"),  # string for a numeric kind
        ("/data/summary", "usd"),  # object
        ("/gated", "count"),  # bool is not a number
        ("/data/summary/total_usd", "text"),  # number for text
        ("/data/summary/total_usd", "timestamp"),
        ("/data/period", "timestamp"),  # "24h" is not a time
        ("/data/sectors/0/sector_score", "count"),  # not a whole number
    ],
)
def test_pointer_to_wrong_type(path: str, kind: str) -> None:
    call_id = "c2" if "sectors" in path else "c1"
    a = answer("{v1}", [ref("v1", call_id, path, kind)])
    out = resolve(a)
    assert [(e.code, e.value_id) for e in out.errors] == [(ResolutionErrorCode.WRONG_TYPE, "v1")]
    assert kind in out.errors[0].message


def test_unknown_tool_call_id() -> None:
    a = answer("{v1}", [ref("v1", "c9", "/data/summary/total_usd", "usd")])
    assert errors(a) == [("unknown_call_ref", "v1")]


def test_failed_tool_call_is_not_a_source() -> None:
    a = answer("{v1}", [ref("v1", "c3", "/error/message", "text")])
    assert errors(a) == [("tool_call_failed", "v1")]


def test_duplicate_value_id() -> None:
    a = answer(
        "{v1}",
        [
            ref("v1", "c1", "/data/summary/total_usd", "usd"),
            ref("v1", "c1", "/data/summary/long_usd", "usd"),
        ],
    )
    out = resolve(a)
    assert errors(a) == [("duplicate_value_id", "v1")]
    assert out.summary == "$56.87M"  # the first definition wins


def test_unused_values_are_reported_not_errors() -> None:
    a = answer(
        "Total {v1}.",
        [
            ref("v1", "c1", "/data/summary/total_usd", "usd"),
            ref("v2", "c1", "/data/summary/count", "count"),
        ],
    )
    out = resolve(a)
    assert out.ok
    assert out.unused_values == ["v2"]
    assert [v.id for v in out.values] == ["v1", "v2"]


def test_literal_numbers_in_text_are_left_as_is() -> None:
    """Golden: only placeholders are replaced. A number the model typed itself is kept
    verbatim (and not flagged here); the answer validator (card 10) rejects it."""
    a = answer(
        "Liquidations hit $57M ({v1}) with 6114 events in the last 24h.",
        [ref("v1", "c1", "/data/summary/total_usd", "usd")],
    )
    out = resolve(a)
    assert out.ok
    assert out.summary == "Liquidations hit $57M ($56.87M) with 6114 events in the last 24h."


@pytest.mark.parametrize("vid", ["v1", "v0", "v42", "v999"])
def test_valid_value_ids(vid: str) -> None:
    assert answer(f"{{{vid}}}", [ref(vid, "c1", "/as_of", "timestamp")]).values[0].id == vid


@pytest.mark.parametrize("vid", ["x1", "v", "v1000", "V1", "v1a", " v1", "v1\n", "v-1"])
def test_value_id_pattern_is_enforced_by_pydantic(vid: str) -> None:
    with pytest.raises(ValidationError, match="id must be v followed by 1 to 3 digits"):
        answer("x", [ref(vid, "c1", "/as_of", "timestamp")])


@pytest.mark.parametrize("decimals", [None, 0, 3, 6])
def test_valid_decimals(decimals: int | None) -> None:
    a = answer("{v1}", [ref("v1", "c1", "/as_of", "timestamp", decimals=decimals)])
    assert a.values[0].decimals == decimals


@pytest.mark.parametrize("decimals", [-1, 7, 100])
def test_decimals_range_is_enforced_by_pydantic(decimals: int) -> None:
    with pytest.raises(ValidationError, match="decimals must be between 0 and 6"):
        answer("{v1}", [ref("v1", "c1", "/as_of", "timestamp", decimals=decimals)])


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        answer("{v1}", [{**ref("v1", "c1", "/as_of", "timestamp"), "value": 42}])


def _objects(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        found = [node] if node.get("type") == "object" else []
        return found + [o for v in node.values() for o in _objects(v)]
    if isinstance(node, list):
        return [o for v in node for o in _objects(v)]
    return []


FORBIDDEN_KEYWORDS = {"pattern", "minimum", "maximum", "exclusiveMinimum",
                      "exclusiveMaximum", "anyOf", "oneOf", "allOf", "$ref", "$defs",
                      "title", "default"}  # fmt: skip


def _keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}
    return set()


def _enums(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        found = [path] if "enum" in node else []
        return found + [e for k, v in node.items() for e in _enums(v, f"{path}/{k}")]
    if isinstance(node, list):
        return [e for i, v in enumerate(node) for e in _enums(v, f"{path}/{i}")]
    return []


def test_schema_uses_a_conservative_subset() -> None:
    schema = answer_json_schema()
    assert not _keys(schema) & FORBIDDEN_KEYWORDS
    assert _enums(schema) == ["/properties/values/items/properties/kind"]
    value = schema["properties"]["values"]["items"]["properties"]
    assert value["decimals"]["type"] == ["integer", "null"]
    assert value["id"] == {"type": "string", "description": value["id"]["description"]}


def test_schema_is_strict() -> None:
    schema = answer_json_schema()
    objects = _objects(schema)
    assert len(objects) == 3  # answer, claim, value
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    value = schema["properties"]["values"]["items"]["properties"]
    assert value["kind"]["enum"] == [
        "number", "percent", "usd", "ratio", "count", "timestamp", "text",
    ]  # fmt: skip
    assert "{v1}" in schema["description"]


# --- call references -----------------------------------------------------------------

SCANNER = {"as_of": AS_OF, "source_id": "mirkaso.scanner", "gated": False, "gated_fields": [],
           "data": {"signals": [{"symbol": "BTCUSDT", "score": 11}], "score_max": 13}}  # fmt: skip
SCANNER_SOL = {
    **SCANNER,
    "data": {"signals": [{"symbol": "SOLUSDT", "score": 10}], "score_max": 13},
}
TWO_SCANS = [
    ("call_a", "scanner_get_signals", SCANNER),
    ("call_b", "scanner_get_signals", SCANNER_SOL),
    ("call_c", "macro_get_regime", NOT_FOUND),
]


def resolve_with(calls: list[tuple[str, str, dict[str, Any]]], a: StructuredAnswer):
    records = [record(*c) for c in calls]
    messages = [
        {"role": "tool", "tool_call_id": cid, "content": json.dumps(r)} for cid, _, r in calls
    ]
    return resolve_answer(a, records, messages)


@pytest.mark.parametrize(
    ("call_ref", "tool_call_id"),
    [
        ("t1", "c1"),  # alias: first executed call
        ("t2", "c2"),
        ("c2", "c2"),  # the real tool call id still works
        ("market_get_liquidations", "c1"),  # a tool name used by exactly one successful call
    ],
)
def test_call_ref_forms(call_ref: str, tool_call_id: str) -> None:
    path = "/as_of" if tool_call_id == "c1" else "/data/sectors/0/slug"
    kind = "timestamp" if tool_call_id == "c1" else "text"
    out = resolve(answer("{v1}", [ref("v1", call_ref, path, kind)]))
    assert out.ok
    (value,) = out.values
    assert (value.tool_call_id, value.call_ref) == (
        tool_call_id,
        "t1" if tool_call_id == "c1" else "t2",
    )


def test_alias_wins_over_a_real_id_of_the_same_spelling() -> None:
    calls = [
        ("t2", "market_get_liquidations", LIQUIDATIONS),
        ("t1", "sectors_get_indices", SECTORS),
    ]
    out = resolve_with(calls, answer("{v1}", [ref("v1", "t1", "/source_id", "text")]))
    assert out.values[0].formatted == "mirkaso.liquidations"  # t1 = first call, not id "t1"


def test_tool_name_used_by_two_successful_calls_is_ambiguous() -> None:
    a = answer("{v1}", [ref("v1", "scanner_get_signals", "/data/signals/0/score", "number")])
    out = resolve_with(TWO_SCANS, a)
    assert [(e.code.value, e.value_id) for e in out.errors] == [("ambiguous_call_ref", "v1")]
    assert "t1" in out.errors[0].message


def test_aliases_disambiguate_calls_of_the_same_tool() -> None:
    a = answer(
        "{v1} {v2}",
        [
            ref("v1", "t1", "/data/signals/0/symbol", "text"),
            ref("v2", "t2", "/data/signals/0/symbol", "text"),
        ],
    )
    assert resolve_with(TWO_SCANS, a).summary == "BTCUSDT SOLUSDT"


def test_tool_name_of_a_failed_call_only() -> None:
    a = answer("{v1}", [ref("v1", "macro_get_regime", "/data/score", "number")])
    out = resolve_with(TWO_SCANS, a)
    assert [(e.code.value, e.value_id) for e in out.errors] == [("tool_call_failed", "v1")]


def test_name_fallback_ignores_failed_calls_of_the_same_tool() -> None:
    calls = [("x", "coin_get_derivatives", NOT_FOUND), *CALLS[:1]]
    calls = [
        *calls,
        ("y", "coin_get_derivatives", {**LIQUIDATIONS, "source_id": "mirkaso.derivatives"}),
    ]
    a = answer("{v1}", [ref("v1", "coin_get_derivatives", "/source_id", "text")])
    out = resolve_with(calls, a)
    assert out.ok and out.values[0].call_ref == "t3"


@pytest.mark.parametrize("call_ref", ["t9", "t0", "nope", "T1", ""])
def test_unknown_call_ref(call_ref: str) -> None:
    out = resolve(answer("{v1}", [ref("v1", call_ref, "/as_of", "timestamp")]))
    assert [(e.code.value, e.value_id) for e in out.errors] == [("unknown_call_ref", "v1")]


# --- no scientific notation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        (8.5e-05, "number", "0.000085"),
        (-0.00021, "number", "-0.00021"),
        (1.23e-08, "number", "0.0000000123"),
        (1.23e-08, "usd", "$0.0000000123"),
        (2.5e-07, "percent", "0.00000025%"),
        (1e-07, "ratio", "0.0000001"),
        (12100000000.0, "number", "12,100,000,000"),
        (1.5e20, "number", "150,000,000,000,000,000,000"),
        (1.5e20, "usd", "$150,000,000T"),
    ],
)
def test_never_scientific_notation(raw: float, kind: str, expected: str) -> None:
    text = format_value(raw, kind, None, "/x")
    assert text == expected
    assert "e" not in text.lower().replace("$", "")


# --- limits --------------------------------------------------------------------------


def test_claim_and_value_limits_are_enforced_by_pydantic() -> None:
    values = [ref(f"v{i}", "t1", "/as_of", "timestamp") for i in range(1, 25)]
    answer("x", values, claims=["c"] * 6)  # at the limits: fine
    with pytest.raises(ValidationError, match="at most 6 claims allowed, got 7"):
        answer("x", values, claims=["c"] * 7)
    with pytest.raises(ValidationError, match="at most 24 values allowed, got 25"):
        answer("x", [*values, ref("v25", "t1", "/as_of", "timestamp")])
