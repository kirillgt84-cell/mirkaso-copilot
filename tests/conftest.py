"""Shared fixtures: a fake DataProvider returning fixed dicts and a scripted model client.

No network.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

AS_OF = "2026-10-26T12:00:00Z"
T0 = datetime(2026, 10, 20, tzinfo=UTC)

SIGNALS = [
    {
        "symbol": "BTCUSDT",
        "base_asset": "BTC",
        "direction": "LONG",
        "score": 11,
        "confidence": "high",
        "price": 67250.5,
        "price_change_1h_pct": 1.2,
        "volume_ratio": 2.4,
        "oi_change_pct": 3.1,
        "quote_volume_24h": 1.5e9,
        "detected_at": "2026-10-26T11:45:00Z",
        "internal_note": "must never reach the model",
    },
    {
        "symbol": "SOLUSDT",
        "base_asset": "SOL",
        "direction": "SHORT",
        "score": 9,
        "confidence": "medium",
        "price": 152.3,
        "price_change_1h_pct": -0.8,
        "volume_ratio": 1.7,
        "oi_change_pct": -2.0,
        "quote_volume_24h": 3.2e8,
        "detected_at": "2026-10-26T11:50:00Z",
    },
]

SECTORS = [
    {
        "slug": "layer-1",
        "asset_class": "crypto",
        "sector_score": 62.5,
        "sector_score_delta_24h": 4.1,
        "price_change_24h_pct": 2.3,
        "oi_change_24h_pct": 1.1,
        "breadth_ratio": 0.7,
        "relative_strength_btc": 1.05,
        "sparkline": [1, 2, 3],
    }
]

DERIVATIVES = {
    "as_of": AS_OF,
    "symbol": "BTCUSDT",
    "price": 67250.5,
    "price_change_1h_pct": 0.4,
    "price_change_4h_pct": 1.0,
    "price_change_24h_pct": 2.5,
    "open_interest_usd": 1.2e10,
    "oi_change_1h_pct": 0.3,
    "oi_change_24h_pct": 4.2,
    "funding_rate": 0.0001,
    "trend": "up",
    "market_score": 72,
}


def hourly_points(n: int, **values: Any) -> list[dict[str, Any]]:
    return [
        {"ts": (T0 + timedelta(hours=i)).isoformat(), **{k: v(i) for k, v in values.items()}}
        for i in range(n)
    ]


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_with: BaseException | None = None
        self.macro: dict[str, Any] = {"as_of": AS_OF, "score": 57.0, "panic_override": False}
        self.macro_points = hourly_points(
            7 * 24, score=lambda i: 50.0, regime=lambda i: "Neutral", panic_override=lambda i: False
        )
        self.oi_points = hourly_points(
            96, open_interest_usd=lambda i: 1e10 + i, price=lambda i: 67000.0 + i
        )
        self.derivatives: dict[str, Any] | None = DERIVATIVES

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        if self.fail_with is not None:
            raise self.fail_with

    async def scanner_signals(self, **kwargs: Any) -> dict[str, Any]:
        self._record("scanner_signals", **kwargs)
        return {"as_of": AS_OF, "signals": SIGNALS}

    async def sector_indices(self, **kwargs: Any) -> dict[str, Any]:
        self._record("sector_indices", **kwargs)
        return {"as_of": AS_OF, "sectors": SECTORS}

    async def macro_regime(self) -> dict[str, Any]:
        self._record("macro_regime")
        return self.macro

    async def macro_regime_history(self, **kwargs: Any) -> dict[str, Any]:
        self._record("macro_regime_history", **kwargs)
        return {"points": self.macro_points}

    async def coin_derivatives(self, **kwargs: Any) -> dict[str, Any] | None:
        self._record("coin_derivatives", **kwargs)
        return self.derivatives

    async def coin_open_interest_history(self, **kwargs: Any) -> dict[str, Any]:
        self._record("coin_open_interest_history", **kwargs)
        return {"points": self.oi_points}


def call(name: str, args: Any = None, call_id: str = "c1") -> dict[str, Any]:
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }


class ScriptedClient:
    """Returns scripted responses in order and records every request's messages."""

    def __init__(self, *responses: dict[str, Any], delay: float = 0.0) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.delay = delay

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.requests.append({"messages": json.loads(json.dumps(messages)), **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.responses:
            raise AssertionError("client called more often than scripted")
        return self.responses.pop(0)


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()
