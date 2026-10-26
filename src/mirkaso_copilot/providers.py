"""The boundary between this repository and whatever serves the market data.

The private Mirkaso product implements ``DataProvider`` in its own codebase; tests
use a fake. Every method returns a plain dict. Tools only read the documented keys,
so extra keys a provider returns never reach the model.

Timestamps (``as_of``, ``ts``, ``bucket``, ``newest_event_at``) are timezone-aware ISO-8601
strings or datetimes. ``as_of`` is optional; tools fall back to the current time.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class Tier(StrEnum):
    GUEST = "guest"
    STARTER = "starter"
    TRADER = "trader"
    PRO = "pro"


_TIER_RANK: dict[str, int] = {
    Tier.GUEST: 0,
    Tier.STARTER: 1,
    Tier.TRADER: 2,
    Tier.PRO: 3,
}


@dataclass(frozen=True, slots=True)
class CallerContext:
    """Who a tool call is made for.

    Built by the HOST application from its own session/auth state. It is never
    derived from model output or tool arguments. Unknown tiers rank as guest.
    """

    tier: str = Tier.GUEST

    def at_least(self, minimum: Tier) -> bool:
        return _TIER_RANK.get(self.tier, 0) >= _TIER_RANK[minimum]


class ProviderUnavailable(Exception):
    """Raised by a provider when its upstream data is temporarily unavailable."""


class NotFound(Exception):
    """Raised when the requested item (e.g. a symbol) does not exist."""


class DataProvider(Protocol):
    async def scanner_signals(
        self,
        *,
        min_score: int,
        direction: str | None,
        confidence: str | None,
        base_asset: str | None,
        limit: int,
    ) -> dict[str, Any]:
        """``{"as_of", "signals": [{symbol, base_asset, direction, score, confidence, price,
        price_change_1h_pct, volume_ratio, oi_change_pct, quote_volume_24h, detected_at}]}``
        ordered strongest first."""
        ...

    async def sector_indices(self, *, asset_class: str | None) -> dict[str, Any]:
        """``{"as_of", "sectors": [{slug, asset_class, sector_score, sector_score_delta_24h,
        price_change_24h_pct, oi_change_24h_pct, breadth_ratio, relative_strength_btc}]}``"""
        ...

    async def macro_regime(self) -> dict[str, Any]:
        """``{"as_of", "score": 0-100, "panic_override": bool}``"""
        ...

    async def macro_regime_history(self, *, days: int) -> dict[str, Any]:
        """``{"points": [{ts, score, regime, panic_override}]}`` covering ``days`` days."""
        ...

    async def coin_derivatives(self, *, symbol: str) -> dict[str, Any] | None:
        """``{"as_of", symbol, price, price_change_1h_pct, price_change_4h_pct,
        price_change_24h_pct, open_interest, oi_change_1h_pct, oi_change_24h_pct,
        funding_rate, trend, market_score}`` or ``None`` if the symbol is unknown."""
        ...

    async def coin_open_interest_history(self, *, symbol: str) -> dict[str, Any]:
        """``{"points": [{ts, open_interest, price}]}`` at the provider's native resolution."""
        ...

    async def liquidations(self, *, period: str, limit: int) -> dict[str, Any]:
        """Aggregates only, no individual events::

        {"as_of", "newest_event_at",
         "summary": {total_usd, long_usd, short_usd, long_pct, short_pct, count},
         "top_coins": [{symbol, total_usd, long_usd, short_usd, count}],
         "time_series": [{bucket, long_usd, short_usd}]}
        """
        ...
