import json
import re
from typing import Any

import pytest

from mirkaso_copilot.providers import CallerContext, ProviderUnavailable, Tier
from mirkaso_copilot.tools.macro_get_regime import regime_bands
from mirkaso_copilot.tools.registry import execute_tool, tool_definitions, tool_names
from tests.conftest import FakeProvider

GUEST = CallerContext(tier=Tier.GUEST)
TRADER = CallerContext(tier=Tier.TRADER)

EXPECTED_TOOLS = {
    "scanner_get_signals",
    "sectors_get_indices",
    "macro_get_regime",
    "coin_get_derivatives",
    "market_get_liquidations",
}


def _params(name: str) -> dict[str, Any]:
    for d in tool_definitions():
        if d["function"]["name"] == name:
            return d["function"]["parameters"]
    raise KeyError(name)


def test_definitions_are_openai_format() -> None:
    defs = tool_definitions()
    assert {d["function"]["name"] for d in defs} == EXPECTED_TOOLS == set(tool_names())
    for d in defs:
        assert d["type"] == "function"
        assert d["function"]["description"]
        params = d["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert "title" not in params
        json.dumps(d)  # serialisable


def test_schema_caps() -> None:
    scanner = _params("scanner_get_signals")["properties"]
    assert scanner["min_score"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 13,
        "default": 8,
        "description": "Minimum signal score (0-13).",
    }
    assert scanner["limit"]["minimum"] == 1
    assert scanner["limit"]["maximum"] == 20
    assert scanner["limit"]["default"] == 10
    assert scanner["direction"]["enum"] == ["LONG", "SHORT"]
    assert scanner["confidence"]["enum"] == ["high", "medium", "low"]

    assert _params("sectors_get_indices")["properties"]["asset_class"]["enum"] == [
        "crypto",
        "tradfi",
    ]

    macro = _params("macro_get_regime")["properties"]
    assert macro["include_history"]["default"] is False
    assert (macro["history_days"]["minimum"], macro["history_days"]["maximum"]) == (1, 30)
    assert macro["history_days"]["default"] == 7

    coin = _params("coin_get_derivatives")
    assert coin["required"] == ["symbol"]
    hp = coin["properties"]["history_points"]
    assert (hp["minimum"], hp["maximum"], hp["default"]) == (1, 48, 24)

    liq = _params("market_get_liquidations")["properties"]
    assert liq["period"]["enum"] == ["1h", "4h", "24h", "7d"]
    assert liq["period"]["default"] == "24h"
    assert (liq["limit"]["minimum"], liq["limit"]["maximum"]) == (1, 20)


async def test_unknown_tool(provider: FakeProvider) -> None:
    out = await execute_tool("drop_tables", {}, provider, GUEST)
    assert out == {"error": {"code": "unknown_tool", "message": "Unknown tool."}}


@pytest.mark.parametrize(
    "args",
    [
        {"limit": 21},
        {"min_score": 14},
        {"direction": "UP"},
        {"surprise": True},
        "{not json",
        "[1, 2]",
    ],
)
async def test_invalid_arguments(provider: FakeProvider, args: Any) -> None:
    out = await execute_tool("scanner_get_signals", args, provider, TRADER)
    assert out["error"]["code"] == "invalid_arguments"
    assert provider.calls == []


async def test_invalid_arguments_message_names_fields_only(provider: FakeProvider) -> None:
    out = await execute_tool("scanner_get_signals", {"limit": 99}, provider, TRADER)
    assert out["error"]["message"] == "Invalid arguments: limit."


async def test_json_string_args_and_defaults(provider: FakeProvider) -> None:
    out = await execute_tool("scanner_get_signals", '{"direction": "LONG"}', provider, TRADER)
    assert out["gated"] is False
    assert provider.calls == [
        (
            "scanner_signals",
            {"min_score": 8, "direction": "LONG", "confidence": None, "base_asset": None,
             "limit": 10},
        )
    ]  # fmt: skip


async def test_none_args_mean_defaults(provider: FakeProvider) -> None:
    out = await execute_tool("macro_get_regime", None, provider, GUEST)
    assert out["data"] == {
        "score": 57.0,
        "regime": "Neutral",
        "panic_override": False,
        "bands": regime_bands(),
    }
    assert out["as_of"] == "2026-10-26T12:00:00Z"
    assert out["source_id"] == "mirkaso.macro"


async def test_exceptions_never_leak(provider: FakeProvider) -> None:
    provider.fail_with = RuntimeError("SELECT * FROM private_table WHERE x = 1")
    out = await execute_tool("macro_get_regime", {}, provider, TRADER)
    assert out == {"error": {"code": "internal_error", "message": "The tool failed unexpectedly."}}
    assert "private_table" not in json.dumps(out)


async def test_provider_unavailable(provider: FakeProvider) -> None:
    provider.fail_with = ProviderUnavailable("db host 10.0.0.5 down")
    out = await execute_tool("sectors_get_indices", {}, provider, TRADER)
    assert out["error"]["code"] == "upstream_unavailable"
    assert "10.0.0.5" not in json.dumps(out)


async def test_scanner_guest_masked_end_to_end(provider: FakeProvider) -> None:
    out = await execute_tool("scanner_get_signals", {}, provider, GUEST)
    assert out["gated"] is True
    assert out["data"]["signals"][1]["symbol"] == "***"


async def test_scanner_guest_masked_coin_filter_returns_nothing(provider: FakeProvider) -> None:
    out = await execute_tool("scanner_get_signals", {"base_asset": "sol"}, provider, GUEST)
    assert out["gated"] is True
    assert "base_asset_filter" in out["gated_fields"]
    assert out["data"] == {"signals": [], "score_max": 13}
    assert provider.calls == []


async def test_scanner_guest_btc_filter_allowed(provider: FakeProvider) -> None:
    await execute_tool("scanner_get_signals", {"base_asset": "btc"}, provider, GUEST)
    assert provider.calls[0][1]["base_asset"] == "BTC"


async def test_scanner_respects_limit(provider: FakeProvider) -> None:
    out = await execute_tool("scanner_get_signals", {"limit": 1}, provider, TRADER)
    assert len(out["data"]["signals"]) == 1


async def test_sectors_guest(provider: FakeProvider) -> None:
    out = await execute_tool("sectors_get_indices", {"asset_class": "crypto"}, provider, GUEST)
    assert out["gated_fields"] == ["sector_score_delta_24h"]
    assert provider.calls == [("sector_indices", {"asset_class": "crypto"})]


async def test_macro_with_history(provider: FakeProvider) -> None:
    out = await execute_tool(
        "macro_get_regime", {"include_history": True, "history_days": 30}, provider, GUEST
    )
    assert len(out["data"]["history"]) == 48
    assert provider.calls[-1] == ("macro_regime_history", {"days": 30})


async def test_macro_panic_override(provider: FakeProvider) -> None:
    provider.macro = {"as_of": "2026-10-26T12:00:00Z", "score": 75, "panic_override": True}
    out = await execute_tool("macro_get_regime", {}, provider, GUEST)
    assert out["data"]["regime"] == "Panic"


async def test_derivatives(provider: FakeProvider) -> None:
    out = await execute_tool(
        "coin_get_derivatives",
        {"symbol": "btcusdt", "include_history": True, "history_points": 12},
        provider,
        GUEST,
    )
    assert out["data"]["symbol"] == "BTCUSDT"
    assert out["data"]["funding_rate"] == 0.0001
    assert len(out["data"]["history"]) == 12
    assert set(out["data"]["history"][0]) == {"ts", "open_interest", "price"}
    assert provider.calls[0] == ("coin_derivatives", {"symbol": "BTCUSDT"})


async def test_derivatives_without_history(provider: FakeProvider) -> None:
    out = await execute_tool("coin_get_derivatives", {"symbol": "BTC"}, provider, GUEST)
    assert "history" not in out["data"]
    assert [c[0] for c in provider.calls] == ["coin_derivatives"]


async def test_derivatives_symbol_required(provider: FakeProvider) -> None:
    out = await execute_tool("coin_get_derivatives", {}, provider, GUEST)
    assert out["error"] == {"code": "invalid_arguments", "message": "Invalid arguments: symbol."}


async def test_derivatives_not_found(provider: FakeProvider) -> None:
    provider.derivatives = None
    out = await execute_tool("coin_get_derivatives", {"symbol": "NOPE"}, provider, GUEST)
    assert out["error"]["code"] == "not_found"


async def test_liquidations(provider: FakeProvider) -> None:
    out = await execute_tool("market_get_liquidations", {"period": "4h"}, provider, GUEST)
    assert out["as_of"] == "2026-10-26T12:00:00Z"
    assert out["newest_event_at"] == "2026-10-26T11:58:30Z"
    data = out["data"]
    assert set(data) == {"period", "summary", "top_coins", "time_series"}
    assert data["period"] == "4h"
    assert data["summary"] == {
        "total_usd": 3.6e6,
        "long_usd": 2.5e6,
        "short_usd": 1.1e6,
        "long_pct": 69.4,
        "short_pct": 30.6,
        "count": 42,
    }
    assert len(data["top_coins"]) == 10
    assert set(data["top_coins"][0]) == {"symbol", "total_usd", "long_usd", "short_usd", "count"}
    assert len(data["time_series"]) == 24
    assert data["time_series"][0] == {
        "bucket": "2026-10-20T00:00:00Z",
        "long_usd": 1000.0,
        "short_usd": 500.0,
    }
    assert "events" not in json.dumps(out)
    assert provider.calls == [("liquidations", {"period": "4h", "limit": 10})]


async def test_liquidations_top_coins_limit(provider: FakeProvider) -> None:
    out = await execute_tool("market_get_liquidations", {"limit": 20}, provider, GUEST)
    assert len(out["data"]["top_coins"]) == 20


async def test_liquidations_empty_payload(provider: FakeProvider) -> None:
    provider.liquidations_payload = {"as_of": "2026-10-26T12:00:00Z"}
    out = await execute_tool("market_get_liquidations", {}, provider, GUEST)
    assert out["newest_event_at"] is None
    assert out["data"]["top_coins"] == []
    assert out["data"]["time_series"] == []
    assert out["data"]["summary"]["count"] is None


@pytest.mark.parametrize(
    ("score", "panic", "regime"),
    [
        (95, True, "Panic"),
        (50, True, "Panic"),
        (0, True, "Panic"),
        (19.9, False, "Panic"),
        (20, False, "Risk OFF"),
        (65, False, "Risk ON"),
        (100, False, "Strong Risk ON"),
    ],
)
async def test_macro_current_regime(
    provider: FakeProvider, score: float, panic: bool, regime: str
) -> None:
    provider.macro = {"as_of": "2026-10-26T12:00:00Z", "score": score, "panic_override": panic}
    out = await execute_tool("macro_get_regime", {}, provider, GUEST)
    assert out["data"]["regime"] == regime
    assert out["data"]["panic_override"] is panic


# Numbers the model may see in a description: timeframes (1h, 24-hour, 7d), field and
# sector names (price_change_1h_pct, layer-1s) and caps on what a call may return
# ("up to 48 points", "max 48"). Data thresholds and scales belong in the result, where
# an answer can point at them.
_ALLOWED_NUMBERS = re.compile(
    r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b"  # field names
    r"|\blayer-\d+s?\b"
    r"|\b\d+(?:[hd]|[- ]hours?|[- ]days?)\b"
    r"|\b(?:up to|max) \d+\b"
)


def test_tool_descriptions_state_no_data_numbers() -> None:
    for d in tool_definitions():
        description = _ALLOWED_NUMBERS.sub("", d["function"]["description"])
        assert not re.search(r"\d", description), d["function"]["name"]


async def test_macro_bands_are_in_the_result(provider: FakeProvider) -> None:
    out = await execute_tool("macro_get_regime", {}, provider, TRADER)
    assert out["data"]["bands"] == [
        {"label": "Panic", "min": 0, "max": 20},
        {"label": "Risk OFF", "min": 20, "max": 35},
        {"label": "Cautious", "min": 35, "max": 50},
        {"label": "Neutral", "min": 50, "max": 65},
        {"label": "Risk ON", "min": 65, "max": 80},
        {"label": "Strong Risk ON", "min": 80, "max": 100},
    ]


async def test_scanner_result_carries_the_score_scale(provider: FakeProvider) -> None:
    out = await execute_tool("scanner_get_signals", {}, provider, TRADER)
    assert out["data"]["score_max"] == 13
