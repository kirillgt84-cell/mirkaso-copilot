import pytest

from mirkaso_copilot.tools._common import chunk_evenly
from mirkaso_copilot.tools.coin_get_derivatives import rebucket_open_interest
from mirkaso_copilot.tools.macro_get_regime import (
    rebucket_macro_history,
    regime_for_score,
)
from tests.conftest import hourly_points


@pytest.mark.parametrize(
    ("score", "regime"),
    [
        (0, "Panic"),
        (19.99, "Panic"),
        (20, "Risk OFF"),
        (34.99, "Risk OFF"),
        (35, "Cautious"),
        (49.99, "Cautious"),
        (50, "Neutral"),
        (64.99, "Neutral"),
        (65, "Risk ON"),
        (79.99, "Risk ON"),
        (80, "Strong Risk ON"),
        (100, "Strong Risk ON"),
    ],
)
def test_regime_bands(score: float, regime: str) -> None:
    # Upper bounds are exclusive: a score on a boundary belongs to the higher band.
    assert regime_for_score(score) == regime


@pytest.mark.parametrize("score", [0, 19.99, 20, 50, 65, 80, 100])
def test_panic_override_forces_panic(score: float) -> None:
    assert regime_for_score(score, panic_override=True) == "Panic"


@pytest.mark.parametrize(("n", "k"), [(0, 48), (1, 48), (48, 48), (49, 48), (720, 48), (5, 2)])
def test_chunk_evenly(n: int, k: int) -> None:
    items = list(range(n))
    chunks = chunk_evenly(items, k)
    assert len(chunks) == min(n, k)
    assert all(chunks)
    assert [x for c in chunks for x in c] == items


def test_macro_short_history_passes_through() -> None:
    points = hourly_points(
        3, score=lambda i: 40.0 + i, regime=lambda i: "Cautious", panic_override=lambda i: False
    )
    out = rebucket_macro_history(points)
    assert [p["score"] for p in out] == [40.0, 41.0, 42.0]
    assert out[0]["ts"] == "2026-10-20T00:00:00Z"


def test_macro_rebucket_caps_at_48_and_averages() -> None:
    # 30 days hourly = 720 rows -> 48 buckets of 15 rows.
    points = hourly_points(
        720,
        score=lambda i: float(i % 15),
        regime=lambda i: "Neutral" if i % 15 < 14 else "Risk ON",
        panic_override=lambda i: False,
    )
    out = rebucket_macro_history(points)
    assert len(out) == 48
    assert out[0]["score"] == 7.0  # mean of 0..14
    assert out[0]["regime"] == "Risk ON"  # latest row in bucket
    assert out[0]["ts"] == "2026-10-20T14:00:00Z"
    assert out[-1]["ts"] == "2026-11-18T23:00:00Z"


def test_macro_rebucket_panic_in_any_row_wins() -> None:
    points = hourly_points(
        96,
        score=lambda i: 70.0,
        regime=lambda i: "Risk ON",
        panic_override=lambda i: i == 0,
    )
    out = rebucket_macro_history(points)
    assert out[0]["panic_override"] is True
    assert out[0]["regime"] == "Panic"
    assert out[1]["panic_override"] is False
    assert out[1]["regime"] == "Risk ON"


def test_macro_rebucket_panic_mid_bucket_overrides_latest_regime() -> None:
    # 720 rows -> 48 buckets of 15; the panic row is in the middle of bucket 3 and the
    # latest row of that bucket says Strong Risk ON.
    points = hourly_points(
        720,
        score=lambda i: 90.0,
        regime=lambda i: "Strong Risk ON",
        panic_override=lambda i: i == 3 * 15 + 7,
    )
    out = rebucket_macro_history(points)
    assert len(out) == 48
    assert [i for i, p in enumerate(out) if p["panic_override"]] == [3]
    assert out[3]["regime"] == "Panic"
    assert out[3]["score"] == 90.0
    assert {p["regime"] for i, p in enumerate(out) if i != 3} == {"Strong Risk ON"}


def test_macro_rebucket_missing_regime_uses_bands() -> None:
    points = hourly_points(2, score=lambda i: 20.0 + 15 * i, panic_override=lambda i: False)
    assert [p["regime"] for p in rebucket_macro_history(points)] == ["Risk OFF", "Cautious"]


def test_macro_rebucket_sorts_by_time() -> None:
    points = hourly_points(
        3, score=lambda i: float(i), regime=lambda i: "Panic", panic_override=lambda i: False
    )
    out = rebucket_macro_history(list(reversed(points)))
    assert [p["score"] for p in out] == [0.0, 1.0, 2.0]


def test_oi_rebucket_keeps_latest_level_per_bucket() -> None:
    points = hourly_points(96, open_interest_usd=lambda i: 1000 + i, price=lambda i: 10.0 + i)
    out = rebucket_open_interest(points, 24)
    assert len(out) == 24
    assert out[0] == {"ts": "2026-10-20T03:00:00Z", "open_interest_usd": 1003, "price": 13.0}
    assert out[-1]["open_interest_usd"] == 1095


def test_oi_rebucket_never_exceeds_48() -> None:
    points = hourly_points(500, open_interest_usd=lambda i: i, price=lambda i: 1.0)
    assert len(rebucket_open_interest(points, 1000)) == 48
