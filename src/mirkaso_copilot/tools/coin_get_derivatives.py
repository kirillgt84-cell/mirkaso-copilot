"""coin_get_derivatives: derivatives snapshot for one coin, optionally with OI history."""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mirkaso_copilot.envelope import ToolResult, to_utc_iso
from mirkaso_copilot.providers import CallerContext, DataProvider, NotFound
from mirkaso_copilot.tools._common import (
    MAX_HISTORY_POINTS,
    as_of_from,
    chunk_evenly,
    parameters_schema,
    parse_ts,
    pick,
)

NAME = "coin_get_derivatives"
SOURCE_ID = "mirkaso.derivatives"
DESCRIPTION = (
    "Get the derivatives snapshot for one coin: price, price changes over 1h/4h/24h, "
    "open interest and its 1h/24h change, funding rate, trend and market score. "
    "Set include_history=true for an open-interest and price history of up to "
    "history_points points (max 48)."
)

DERIVATIVES_FIELDS = (
    "symbol",
    "price",
    "price_change_1h_pct",
    "price_change_4h_pct",
    "price_change_24h_pct",
    "open_interest",
    "oi_change_1h_pct",
    "oi_change_24h_pct",
    "funding_rate",
    "trend",
    "market_score",
)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        pattern=r"^[A-Za-z0-9]{1,20}$", description="Coin symbol, e.g. BTC or BTCUSDT."
    )
    include_history: bool = Field(False, description="Also return open-interest history.")
    history_points: int = Field(
        24, ge=1, le=MAX_HISTORY_POINTS, description="Number of history points (1-48)."
    )

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


def rebucket_open_interest(
    points: Iterable[Mapping[str, Any]], max_points: int = MAX_HISTORY_POINTS
) -> list[dict[str, Any]]:
    """Downsample to at most ``max_points`` time-ordered buckets.

    Open interest and price are levels, so each bucket keeps its latest row.
    """
    max_points = min(max_points, MAX_HISTORY_POINTS)
    ordered = sorted(points, key=lambda p: parse_ts(p["ts"]))
    out = []
    for bucket in chunk_evenly(ordered, max_points):
        latest = bucket[-1]
        out.append(
            {
                "ts": to_utc_iso(latest["ts"]),
                "open_interest": latest.get("open_interest"),
                "price": latest.get("price"),
            }
        )
    return out


def parameters() -> dict[str, Any]:
    return parameters_schema(Params)


async def execute(provider: DataProvider, ctx: CallerContext, params: Params) -> ToolResult:
    payload = await provider.coin_derivatives(symbol=params.symbol)
    if not payload:
        raise NotFound(params.symbol)
    data = pick(payload, DERIVATIVES_FIELDS)
    if params.include_history:
        history = await provider.coin_open_interest_history(symbol=params.symbol)
        data["history"] = rebucket_open_interest(history.get("points", []), params.history_points)
    return ToolResult(as_of=as_of_from(payload), source_id=SOURCE_ID, data=data)
