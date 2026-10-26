"""scanner_get_signals: current market-scanner signals, gated by caller tier."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mirkaso_copilot.envelope import ToolResult, utc_now
from mirkaso_copilot.providers import CallerContext, DataProvider, Tier
from mirkaso_copilot.tools._common import as_of_from, parameters_schema, pick

NAME = "scanner_get_signals"
SOURCE_ID = "mirkaso.scanner"
DESCRIPTION = (
    "List the strongest current market-scanner signals for crypto perpetual futures. "
    "Each signal has a direction (LONG or SHORT), a score from 0 to 13 (higher is stronger), "
    "a confidence level and price_change_1h_pct, the price move over the last 1 hour "
    "(never a 24-hour move). If the result has gated=true, the fields named in gated_fields "
    "are hidden on the user's plan: masked symbols appear as '***', price is a range label, "
    "and null means 'not available on this plan', not zero. Never guess hidden values."
)

FULL_FIDELITY_TIER = Tier.TRADER
UNMASKED_BASES = frozenset({"BTC", "ETH"})
MASK = "***"

SIGNAL_FIELDS = (
    "symbol",
    "base_asset",
    "direction",
    "score",
    "confidence",
    "price",
    "price_change_1h_pct",
    "volume_ratio",
    "oi_change_pct",
    "quote_volume_24h",
    "detected_at",
)
NULLED_FIELDS = ("volume_ratio", "oi_change_pct", "price_change_1h_pct", "quote_volume_24h")
GATED_FIELDS = ("symbol", "base_asset", "price", *NULLED_FIELDS)
BASE_ASSET_FILTER_FIELD = "base_asset_filter"

_PRICE_BUCKETS = (
    (0.01, "<0.01"),
    (0.1, "0.01-0.1"),
    (1, "0.1-1"),
    (10, "1-10"),
    (100, "10-100"),
    (1_000, "100-1k"),
    (10_000, "1k-10k"),
    (100_000, "10k-100k"),
)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_score: int = Field(8, ge=0, le=13, description="Minimum signal score (0-13).")
    direction: Literal["LONG", "SHORT"] | None = Field(None, description="Only this direction.")
    confidence: Literal["high", "medium", "low"] | None = Field(
        None, description="Only this confidence level."
    )
    base_asset: str | None = Field(
        None, pattern=r"^[A-Za-z0-9]{1,15}$", description="Only this base asset, e.g. BTC."
    )
    limit: int = Field(10, ge=1, le=20, description="Maximum number of signals (1-20).")

    @field_validator("base_asset")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value


def bucket_price(price: float | None) -> str | None:
    """Order-of-magnitude price range label, e.g. ``"10-100"``."""
    if price is None:
        return None
    p = float(price)
    for upper, label in _PRICE_BUCKETS:
        if p < upper:
            return label
    return ">=100k"


def mask_signal(row: Mapping[str, Any]) -> dict[str, Any]:
    masked = dict(row)
    if str(row.get("base_asset") or "").upper() not in UNMASKED_BASES:
        masked["symbol"] = MASK
        masked["base_asset"] = MASK
    masked["price"] = bucket_price(row.get("price"))
    for field in NULLED_FIELDS:
        masked[field] = None
    return masked


def gate_signals(
    signals: Iterable[Mapping[str, Any]], ctx: CallerContext
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Return ``(rows, gated, gated_fields)`` for the caller's tier."""
    rows = [pick(s, SIGNAL_FIELDS) for s in signals]
    if ctx.at_least(FULL_FIDELITY_TIER):
        return rows, False, []
    return [mask_signal(r) for r in rows], True, list(GATED_FIELDS)


def parameters() -> dict[str, Any]:
    return parameters_schema(Params)


async def execute(provider: DataProvider, ctx: CallerContext, params: Params) -> ToolResult:
    if (
        not ctx.at_least(FULL_FIDELITY_TIER)
        and params.base_asset
        and params.base_asset not in UNMASKED_BASES
    ):
        # Filtering by a masked coin would reveal whether that coin has signals.
        return ToolResult(
            as_of=utc_now(),
            source_id=SOURCE_ID,
            gated=True,
            gated_fields=[BASE_ASSET_FILTER_FIELD, *GATED_FIELDS],
            data={"signals": []},
        )

    payload = await provider.scanner_signals(
        min_score=params.min_score,
        direction=params.direction,
        confidence=params.confidence,
        base_asset=params.base_asset,
        limit=params.limit,
    )
    signals, gated, gated_fields = gate_signals(payload.get("signals", [])[: params.limit], ctx)
    return ToolResult(
        as_of=as_of_from(payload),
        source_id=SOURCE_ID,
        gated=gated,
        gated_fields=gated_fields,
        data={"signals": signals},
    )
