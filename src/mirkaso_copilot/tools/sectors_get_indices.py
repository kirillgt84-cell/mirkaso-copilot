"""sectors_get_indices: sector index snapshot, gated by caller tier."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mirkaso_copilot.envelope import ToolResult
from mirkaso_copilot.providers import CallerContext, DataProvider, Tier
from mirkaso_copilot.tools._common import as_of_from, parameters_schema, pick

NAME = "sectors_get_indices"
SOURCE_ID = "mirkaso.sectors"
DESCRIPTION = (
    "Get the current sector indices (e.g. layer-1s, DeFi, AI coins) with a sector score, "
    "its 24-hour change, 24-hour price and open-interest changes, breadth (share of members "
    "rising) and strength relative to BTC. Filter by asset_class: crypto or tradfi. "
    "If gated=true, fields in gated_fields are null because they are not available on the "
    "user's plan; do not treat them as zero."
)

DELTA_MIN_TIER = Tier.STARTER
SECTOR_FIELDS = (
    "slug",
    "asset_class",
    "sector_score",
    "sector_score_delta_24h",
    "price_change_24h_pct",
    "oi_change_24h_pct",
    "breadth_ratio",
    "relative_strength_btc",
)
GATED_FIELDS = ("sector_score_delta_24h",)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_class: Literal["crypto", "tradfi"] | None = Field(
        None, description="Only sectors of this asset class; omit for all."
    )


def gate_sectors(
    sectors: Iterable[Mapping[str, Any]], ctx: CallerContext
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Return ``(rows, gated, gated_fields)`` for the caller's tier."""
    rows = [pick(s, SECTOR_FIELDS) for s in sectors]
    if ctx.at_least(DELTA_MIN_TIER):
        return rows, False, []
    for row in rows:
        for field in GATED_FIELDS:
            row[field] = None
    return rows, True, list(GATED_FIELDS)


def parameters() -> dict[str, Any]:
    return parameters_schema(Params)


async def execute(provider: DataProvider, ctx: CallerContext, params: Params) -> ToolResult:
    payload = await provider.sector_indices(asset_class=params.asset_class)
    sectors, gated, gated_fields = gate_sectors(payload.get("sectors", []), ctx)
    return ToolResult(
        as_of=as_of_from(payload),
        source_id=SOURCE_ID,
        gated=gated,
        gated_fields=gated_fields,
        data={"sectors": sectors},
    )
