import pytest

from mirkaso_copilot.providers import CallerContext, Tier
from mirkaso_copilot.tools.scanner_get_signals import (
    GATED_FIELDS,
    MASK,
    bucket_price,
    gate_signals,
)
from mirkaso_copilot.tools.sectors_get_indices import gate_sectors
from tests.conftest import SECTORS, SIGNALS


@pytest.mark.parametrize("tier", [Tier.TRADER, Tier.PRO])
def test_scanner_full_fidelity_for_trader_and_above(tier: Tier) -> None:
    rows, gated, gated_fields = gate_signals(SIGNALS, CallerContext(tier=tier))
    assert gated is False
    assert gated_fields == []
    assert rows[1]["symbol"] == "SOLUSDT"
    assert rows[1]["price"] == 152.3
    assert rows[1]["volume_ratio"] == 1.7


@pytest.mark.parametrize("tier", [Tier.GUEST, Tier.STARTER, "unknown-tier"])
def test_scanner_masked_below_trader(tier: str) -> None:
    rows, gated, gated_fields = gate_signals(SIGNALS, CallerContext(tier=tier))
    assert gated is True
    assert gated_fields == list(GATED_FIELDS)

    btc, sol = rows
    assert btc["symbol"] == "BTCUSDT"
    assert btc["base_asset"] == "BTC"
    assert sol["symbol"] == MASK
    assert sol["base_asset"] == MASK

    assert btc["price"] == "10k-100k"
    assert sol["price"] == "100-1k"
    for row in rows:
        for field in ("volume_ratio", "oi_change_pct", "price_change_1h_pct", "quote_volume_24h"):
            assert row[field] is None
        # Non-sensitive fields stay visible.
        assert row["direction"] in {"LONG", "SHORT"}
        assert isinstance(row["score"], int)


def test_scanner_drops_unknown_provider_fields() -> None:
    rows, _, _ = gate_signals(SIGNALS, CallerContext(tier=Tier.PRO))
    assert "internal_note" not in rows[0]


def test_scanner_does_not_mutate_input() -> None:
    gate_signals(SIGNALS, CallerContext(tier=Tier.GUEST))
    assert SIGNALS[1]["symbol"] == "SOLUSDT"


@pytest.mark.parametrize(
    ("price", "label"),
    [
        (None, None),
        (0.005, "<0.01"),
        (0.01, "0.01-0.1"),
        (0.5, "0.1-1"),
        (9.99, "1-10"),
        (10, "10-100"),
        (999, "100-1k"),
        (5_000, "1k-10k"),
        (67_250.5, "10k-100k"),
        (150_000, ">=100k"),
    ],
)
def test_bucket_price(price: float | None, label: str | None) -> None:
    assert bucket_price(price) == label


def test_sectors_guest_hides_delta() -> None:
    rows, gated, gated_fields = gate_sectors(SECTORS, CallerContext(tier=Tier.GUEST))
    assert gated is True
    assert gated_fields == ["sector_score_delta_24h"]
    assert rows[0]["sector_score_delta_24h"] is None
    assert rows[0]["sector_score"] == 62.5
    assert "sparkline" not in rows[0]


@pytest.mark.parametrize("tier", [Tier.STARTER, Tier.TRADER, Tier.PRO])
def test_sectors_starter_and_above_see_delta(tier: Tier) -> None:
    rows, gated, gated_fields = gate_sectors(SECTORS, CallerContext(tier=tier))
    assert gated is False
    assert gated_fields == []
    assert rows[0]["sector_score_delta_24h"] == 4.1
