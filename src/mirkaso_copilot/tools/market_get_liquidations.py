"""market_get_liquidations: aggregated forced liquidations over a period."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mirkaso_copilot.envelope import LiquidationsResult, to_utc_iso
from mirkaso_copilot.providers import CallerContext, DataProvider
from mirkaso_copilot.tools._common import (
    as_of_from,
    chunk_evenly,
    parameters_schema,
    parse_ts,
    pick,
)

NAME = "market_get_liquidations"
SOURCE_ID = "mirkaso.liquidations"
DESCRIPTION = (
    "Get aggregated forced liquidations of leveraged positions over a period "
    "(1h, 4h, 24h or 7d): a summary of total, long and short USD value with shares and "
    "count, the top coins by liquidated value, and a long/short USD time series. "
    "newest_event_at is the time of the newest liquidation seen; if it is old, say the "
    "data may be stale."
)

MAX_TIME_SERIES_POINTS = 168
SUMMARY_FIELDS = ("total_usd", "long_usd", "short_usd", "long_pct", "short_pct", "count")
TOP_COIN_FIELDS = ("symbol", "total_usd", "long_usd", "short_usd", "count")


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: Literal["1h", "4h", "24h", "7d"] = Field("24h", description="Look-back period.")
    limit: int = Field(10, ge=1, le=20, description="Maximum number of top coins (1-20).")


def rebucket_time_series(
    points: Iterable[Mapping[str, Any]], max_points: int = MAX_TIME_SERIES_POINTS
) -> list[dict[str, Any]]:
    """Downsample to at most ``max_points`` time-ordered buckets.

    Liquidated USD values are flows, so each bucket sums its rows and is labelled
    with the start of its first row.
    """
    ordered = sorted(points, key=lambda p: parse_ts(p["bucket"]))
    out = []
    for bucket in chunk_evenly(ordered, max_points):
        out.append(
            {
                "bucket": to_utc_iso(bucket[0]["bucket"]),
                "long_usd": sum(p.get("long_usd") or 0 for p in bucket),
                "short_usd": sum(p.get("short_usd") or 0 for p in bucket),
            }
        )
    return out


def parameters() -> dict[str, Any]:
    return parameters_schema(Params)


async def execute(provider: DataProvider, ctx: CallerContext, params: Params) -> LiquidationsResult:
    payload = await provider.liquidations(period=params.period, limit=params.limit)
    data = {
        "period": params.period,
        "summary": pick(payload.get("summary") or {}, SUMMARY_FIELDS),
        "top_coins": [pick(c, TOP_COIN_FIELDS) for c in payload.get("top_coins", [])][
            : params.limit
        ],
        "time_series": rebucket_time_series(payload.get("time_series", [])),
    }
    return LiquidationsResult(
        as_of=as_of_from(payload),
        source_id=SOURCE_ID,
        newest_event_at=payload.get("newest_event_at"),
        data=data,
    )
