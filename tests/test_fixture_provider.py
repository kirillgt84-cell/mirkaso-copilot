"""FixtureProvider: bundled synthetic data that every tool accepts. No network."""

import json
from datetime import UTC, datetime, timedelta
from importlib import resources

import pytest

from mirkaso_copilot.envelope import ErrorResult, LiquidationsResult, ToolResult, utc_now
from mirkaso_copilot.fixture_provider import (
    BASE_TIME,
    FIXTURE_PACKAGE,
    SYNTHETIC_NOTICE,
    FixtureProvider,
    load_fixture,
)
from mirkaso_copilot.providers import CallerContext, Tier
from mirkaso_copilot.tools import (
    coin_get_derivatives,
    market_get_liquidations,
    scanner_get_signals,
    sectors_get_indices,
)
from mirkaso_copilot.tools.registry import execute_tool, tool_names

FIXTURE_NAMES = ("scanner", "sectors", "macro", "derivatives", "liquidations")
TRADER = CallerContext(tier=Tier.TRADER)
GUEST = CallerContext(tier=Tier.GUEST)

# One representative call per tool, with every optional feature switched on.
TOOL_CALLS = {
    "scanner_get_signals": {"min_score": 0, "limit": 20},
    "sectors_get_indices": {},
    "macro_get_regime": {"include_history": True, "history_days": 7},
    "coin_get_derivatives": {"symbol": "BTC", "include_history": True, "history_points": 48},
    "market_get_liquidations": {"period": "24h", "limit": 20},
}


@pytest.fixture
def fixtures() -> FixtureProvider:
    return FixtureProvider(fixed_time=True)


async def run(name: str, args: dict, ctx: CallerContext = TRADER) -> dict:
    return await execute_tool(name, args, FixtureProvider(fixed_time=True), ctx)


def raw(name: str) -> dict:
    text = resources.files(FIXTURE_PACKAGE).joinpath(f"{name}.json").read_text("utf-8")
    return json.loads(text)


def test_every_tool_has_a_fixture_call() -> None:
    assert set(TOOL_CALLS) == set(tool_names())


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_files_are_marked_synthetic_with_utc_times(name: str) -> None:
    payload = raw(name)
    assert payload["notice"] == SYNTHETIC_NOTICE
    assert "synthetic demo data" in payload["notice"]
    assert payload["base_time"] == BASE_TIME == "2026-10-26T12:00:00Z"
    assert payload["as_of"].endswith("Z")


@pytest.mark.parametrize("tier", list(Tier))
@pytest.mark.parametrize("name", list(TOOL_CALLS))
async def test_every_tool_returns_a_valid_result(name: str, tier: Tier) -> None:
    result = await run(name, TOOL_CALLS[name], CallerContext(tier=tier))
    assert "error" not in result, result
    model = LiquidationsResult if name == "market_get_liquidations" else ToolResult
    parsed = model.model_validate(result)
    assert parsed.as_of == BASE_TIME


async def test_full_fidelity_rows_have_every_field() -> None:
    signals = (await run("scanner_get_signals", TOOL_CALLS["scanner_get_signals"]))["data"]
    assert len(signals["signals"]) == 12
    for row in signals["signals"]:
        assert set(row) == set(scanner_get_signals.SIGNAL_FIELDS)
        assert None not in row.values()
        assert 6 <= row["score"] <= 12
        assert row["detected_at"].endswith("Z")

    sectors = (await run("sectors_get_indices", {}))["data"]["sectors"]
    for row in sectors:
        assert set(row) == set(sectors_get_indices.SECTOR_FIELDS)
        assert None not in row.values()

    coin = (await run("coin_get_derivatives", TOOL_CALLS["coin_get_derivatives"]))["data"]
    assert None not in (coin[f] for f in coin_get_derivatives.DERIVATIVES_FIELDS)

    liq = (await run("market_get_liquidations", TOOL_CALLS["market_get_liquidations"]))["data"]
    assert set(liq["summary"]) == set(market_get_liquidations.SUMMARY_FIELDS)
    assert None not in liq["summary"].values()
    for row in liq["top_coins"]:
        assert None not in row.values()


async def test_scanner_filters_and_limit(fixtures: FixtureProvider) -> None:
    base = {"min_score": 0, "direction": None, "confidence": None, "base_asset": None}
    rows = (await fixtures.scanner_signals(**{**base, "limit": 20}))["signals"]
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert {"BTC", "ETH", "SOL"} <= {r["base_asset"] for r in rows}

    strong = (await fixtures.scanner_signals(**{**base, "min_score": 9, "limit": 20}))["signals"]
    assert strong and all(r["score"] >= 9 for r in strong)
    assert len(strong) < len(rows)

    shorts = (await fixtures.scanner_signals(**{**base, "direction": "SHORT", "limit": 20}))[
        "signals"
    ]
    assert shorts and {r["direction"] for r in shorts} == {"SHORT"}

    high = (await fixtures.scanner_signals(**{**base, "confidence": "high", "limit": 20}))[
        "signals"
    ]
    assert high and {r["confidence"] for r in high} == {"high"}

    eth = (await fixtures.scanner_signals(**{**base, "base_asset": "eth", "limit": 20}))["signals"]
    assert len(eth) == 2 and {r["base_asset"] for r in eth} == {"ETH"}

    assert len((await fixtures.scanner_signals(**{**base, "limit": 3}))["signals"]) == 3


async def test_scanner_tool_default_min_score() -> None:
    rows = (await run("scanner_get_signals", {}))["data"]["signals"]
    assert rows and all(r["score"] >= 8 for r in rows)


async def test_scanner_is_masked_for_guest() -> None:
    result = await run("scanner_get_signals", {"min_score": 0}, GUEST)
    assert result["gated"] is True
    assert {r["symbol"] for r in result["data"]["signals"]} >= {"BTCUSDT", "***"}


async def test_sectors_asset_class_filter(fixtures: FixtureProvider) -> None:
    every = (await fixtures.sector_indices(asset_class=None))["sectors"]
    crypto = (await fixtures.sector_indices(asset_class="crypto"))["sectors"]
    tradfi = (await fixtures.sector_indices(asset_class="tradfi"))["sectors"]
    assert len(crypto) >= 10 and len(tradfi) >= 3
    assert len(every) == len(crypto) + len(tradfi)
    assert {s["asset_class"] for s in crypto} == {"crypto"}
    assert {s["asset_class"] for s in tradfi} == {"tradfi"}


async def test_macro_current_and_rebucketed_history(fixtures: FixtureProvider) -> None:
    week = (await fixtures.macro_regime_history(days=7))["points"]
    assert len(week) == 7 * 24 * 4
    assert week[-1]["ts"] == BASE_TIME
    assert len((await fixtures.macro_regime_history(days=1))["points"]) == 96
    # The fixtures hold 7 days; longer windows return what there is.
    assert len((await fixtures.macro_regime_history(days=30))["points"]) == len(week)

    result = await run("macro_get_regime", TOOL_CALLS["macro_get_regime"])
    assert result["data"]["score"] == 54.3
    assert result["data"]["regime"] == "Neutral"
    history = result["data"]["history"]
    assert len(history) == 48
    assert history[-1]["ts"] == BASE_TIME


@pytest.mark.parametrize("symbol", ["BTC", "btc", "BTCUSDT", "ETH", "SOL"])
async def test_derivatives_symbols_and_open_interest_history(symbol: str) -> None:
    result = await run("coin_get_derivatives", {"symbol": symbol, "include_history": True})
    data = result["data"]
    assert data["symbol"].startswith(symbol.upper().removesuffix("USDT"))
    assert len(data["history"]) == 24
    assert data["history"][-1]["open_interest_usd"] == data["open_interest_usd"]
    assert data["history"][-1]["ts"] == BASE_TIME


async def test_open_interest_history_is_48_hourly_points(fixtures: FixtureProvider) -> None:
    points = (await fixtures.coin_open_interest_history(symbol="SOL"))["points"]
    assert len(points) == 48
    assert points[-1]["ts"] == BASE_TIME


async def test_unknown_symbol_is_not_found(fixtures: FixtureProvider) -> None:
    assert await fixtures.coin_derivatives(symbol="NOPE") is None
    assert (await fixtures.coin_open_interest_history(symbol="NOPE")) == {"points": []}
    result = await run("coin_get_derivatives", {"symbol": "NOPE"})
    ErrorResult.model_validate(result)
    assert result["error"]["code"] == "not_found"


@pytest.mark.parametrize("period", ["1h", "4h", "24h", "7d"])
async def test_liquidations_periods(period: str, fixtures: FixtureProvider) -> None:
    payload = await fixtures.liquidations(period=period, limit=5)
    assert len(payload["top_coins"]) == 5
    summary = payload["summary"]
    assert summary["long_usd"] + summary["short_usd"] == summary["total_usd"]
    assert sum(p["long_usd"] for p in payload["time_series"]) == summary["long_usd"]
    assert sum(c["total_usd"] for c in payload["top_coins"]) <= summary["total_usd"]

    result = await run("market_get_liquidations", {"period": period, "limit": 20})
    assert result["data"]["period"] == period
    assert result["newest_event_at"].endswith("Z")
    assert len(result["data"]["time_series"]) <= market_get_liquidations.MAX_TIME_SERIES_POINTS


async def test_returned_rows_are_copies(fixtures: FixtureProvider) -> None:
    first = await fixtures.sector_indices(asset_class=None)
    first["sectors"][0]["sector_score"] = -1
    second = await fixtures.sector_indices(asset_class=None)
    assert second["sectors"][0]["sector_score"] != -1


# --- time shifting -----------------------------------------------------------------

SHIFT = timedelta(days=3, minutes=7, seconds=5)
SHIFTED_NOW = datetime(2026, 10, 26, 12, tzinfo=UTC) + SHIFT


async def snapshot(provider: FixtureProvider) -> dict[str, list[str]]:
    """Every timestamp each method returns, keyed by method."""
    macro_history = await provider.macro_regime_history(days=7)
    liq = await provider.liquidations(period="24h", limit=20)
    oi_points = await provider.coin_open_interest_history(symbol="SOL")
    signals = await provider.scanner_signals(
        min_score=0, direction=None, confidence=None, base_asset=None, limit=20
    )
    return {
        "as_of": [
            (await provider.macro_regime())["as_of"],
            (await provider.sector_indices(asset_class=None))["as_of"],
            (await provider.coin_derivatives(symbol="BTC"))["as_of"],
            signals["as_of"],
            liq["as_of"],
        ],
        "detected_at": [s["detected_at"] for s in signals["signals"]],
        "macro_ts": [p["ts"] for p in macro_history["points"]],
        "oi_ts": [p["ts"] for p in oi_points["points"]],
        "liq": [liq["newest_event_at"], *(p["bucket"] for p in liq["time_series"])],
    }


def as_dt(values: list[str]) -> list[datetime]:
    return [datetime.fromisoformat(v) for v in values]


async def test_fixed_time_keeps_stored_timestamps() -> None:
    provider = FixtureProvider(fixed_time=True, now=SHIFTED_NOW)
    assert provider.offset == timedelta(0)
    assert provider.base_time == BASE_TIME
    times = await snapshot(provider)
    assert set(times["as_of"]) == {BASE_TIME}
    assert times["macro_ts"] == [p["ts"] for p in raw("macro")["history"]]


async def test_shift_moves_every_timestamp_by_the_same_offset() -> None:
    fixed = await snapshot(FixtureProvider(fixed_time=True))
    provider = FixtureProvider(now=SHIFTED_NOW)
    assert provider.offset == SHIFT
    shifted = await snapshot(provider)

    assert set(shifted["as_of"]) == {provider.base_time} == {"2026-10-29T12:07:05Z"}
    for key in fixed:
        assert all(v.endswith("Z") for v in shifted[key]), key
        assert [s - f for s, f in zip(as_dt(shifted[key]), as_dt(fixed[key]), strict=True)] == [
            SHIFT
        ] * len(fixed[key]), key
    # Differences between timestamps, including across files, are preserved.
    anchor = as_dt(shifted["as_of"])[0]
    fixed_anchor = as_dt(fixed["as_of"])[0]
    for key in fixed:
        assert [t - anchor for t in as_dt(shifted[key])] == [
            t - fixed_anchor for t in as_dt(fixed[key])
        ]


async def test_shift_does_not_touch_the_cached_fixture() -> None:
    await snapshot(FixtureProvider(now=SHIFTED_NOW))
    assert raw("macro")["as_of"] == load_fixture("macro")["as_of"] == BASE_TIME


async def test_default_shifts_to_the_current_time() -> None:
    before = utc_now().replace(microsecond=0)
    provider = FixtureProvider()
    as_of = datetime.fromisoformat((await provider.macro_regime())["as_of"])
    assert before - timedelta(seconds=1) <= as_of <= utc_now()
    assert (await provider.macro_regime())["as_of"].endswith("Z")

    # Shifted history still ends at as_of and windows are measured from it.
    history = (await provider.macro_regime_history(days=1))["points"]
    assert len(history) == 96
    assert datetime.fromisoformat(history[-1]["ts"]) == as_of

    result = await execute_tool("market_get_liquidations", {"period": "1h"}, provider, TRADER)
    LiquidationsResult.model_validate(result)
    assert datetime.fromisoformat(result["newest_event_at"]) == as_of - timedelta(seconds=42)
