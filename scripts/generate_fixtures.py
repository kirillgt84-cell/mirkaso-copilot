"""Regenerate the synthetic demo fixtures served by ``FixtureProvider``.

Usage (from the repository root)::

    python scripts/generate_fixtures.py [OUTPUT_DIR]

OUTPUT_DIR defaults to ``src/mirkaso_copilot/fixtures``. The random generator uses a
fixed seed, so the output is byte-for-byte identical on every run; a test checks that
the committed files match. All values are synthetic demo data, not real market data.
"""

import json
import math
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mirkaso_copilot.tools.macro_get_regime import regime_for_score

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "mirkaso_copilot" / "fixtures"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
BASE = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
NOTICE = "synthetic demo data - not real market data"
rng = random.Random(20261026)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def header() -> dict:
    return {"notice": NOTICE, "base_time": iso(BASE), "as_of": iso(BASE)}


def dump(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=1) + "\n")


# --- derivatives snapshot (also feeds scanner prices) -------------------------------
DERIVS = {
    "BTCUSDT": dict(
        price=67250.5,
        h1=1.2,
        h4=1.9,
        h24=3.4,
        oi=1.21e10,
        oi1=0.9,
        oi24=4.6,
        funding=0.000085,
        trend="up",
        market_score=71,
    ),
    "ETHUSDT": dict(
        price=2618.4,
        h1=0.4,
        h4=-0.6,
        h24=1.1,
        oi=6.35e9,
        oi1=0.2,
        oi24=1.8,
        funding=0.000052,
        trend="sideways",
        market_score=58,
    ),
    "SOLUSDT": dict(
        price=152.31,
        h1=-2.7,
        h4=-4.1,
        h24=-1.9,
        oi=2.48e9,
        oi1=6.8,
        oi24=14.2,
        funding=-0.000210,
        trend="down",
        market_score=39,
    ),
}


def series_with_anchors(n: int, last: float, anchors: dict[int, float], vol: float) -> list[float]:
    """n hourly values ending at ``last``; anchors map hours-before-end to exact values."""
    pts = {0: last, **anchors}
    pts.setdefault(n - 1, anchors[max(anchors)] * (1 + rng.uniform(-0.01, 0.01)))
    keys = sorted(pts)
    out = []
    for back in range(n - 1, -1, -1):
        lo = max(k for k in keys if k <= back)
        hi = min(k for k in keys if k >= back)
        if lo == hi:
            out.append(pts[lo])
            continue
        t = (back - lo) / (hi - lo)
        base = pts[lo] + (pts[hi] - pts[lo]) * t
        out.append(base * (1 + rng.gauss(0, vol)))
    return out


derivatives = {}
oi_series = {}
for sym, d in DERIVS.items():
    derivatives[sym] = {
        "symbol": sym,
        "price": d["price"],
        "price_change_1h_pct": d["h1"],
        "price_change_4h_pct": d["h4"],
        "price_change_24h_pct": d["h24"],
        "open_interest_usd": d["oi"],
        "oi_change_1h_pct": d["oi1"],
        "oi_change_24h_pct": d["oi24"],
        "funding_rate": d["funding"],
        "trend": d["trend"],
        "market_score": d["market_score"],
    }
    p = d["price"]
    prices = series_with_anchors(
        48, p, {1: p / (1 + d["h1"] / 100), 4: p / (1 + d["h4"] / 100),
                24: p / (1 + d["h24"] / 100)}, 0.002)  # fmt: skip
    o = d["oi"]
    ois = series_with_anchors(
        48, o, {1: o / (1 + d["oi1"] / 100), 24: o / (1 + d["oi24"] / 100)}, 0.003
    )
    decimals = 2 if p > 100 else 4
    oi_series[sym] = [
        {"ts": iso(BASE - timedelta(hours=47 - i)), "open_interest_usd": round(ois[i], -3),
         "price": round(prices[i], decimals)}
        for i in range(48)
    ]  # fmt: skip
    oi_series[sym][-1]["open_interest_usd"] = o
    oi_series[sym][-1]["price"] = p

dump("derivatives.json", {**header(), "coins": derivatives, "open_interest_history": oi_series})

# --- scanner signals ---------------------------------------------------------------
# (symbol, base, direction, score, confidence, price, 1h %, volume_ratio, oi %, qv24, minutes ago)
SIGNALS = [
    ("BTCUSDT", "BTC", "LONG", 11, "high", 67250.5, 1.2, 2.4, 3.1, 1.52e10, 14),
    ("SOLUSDT", "SOL", "SHORT", 10, "high", 152.31, -2.7, 3.1, 6.8, 2.9e9, 9),
    ("ETHUSDT", "ETH", "LONG", 9, "medium", 2618.4, 0.4, 1.6, 0.9, 7.8e9, 22),
    ("SUIUSDT", "SUI", "LONG", 12, "high", 1.8423, 3.9, 4.2, 8.4, 6.1e8, 6),
    ("DOGEUSDT", "DOGE", "SHORT", 8, "medium", 0.11842, -1.4, 1.9, 2.2, 1.1e9, 31),
    ("LINKUSDT", "LINK", "LONG", 8, "medium", 11.274, 1.1, 1.5, 1.7, 4.2e8, 18),
    ("AVAXUSDT", "AVAX", "SHORT", 7, "low", 26.81, -0.9, 1.3, -1.2, 3.3e8, 44),
    ("XRPUSDT", "XRP", "LONG", 7, "medium", 0.5312, 0.6, 1.4, 1.0, 1.4e9, 37),
    ("ARBUSDT", "ARB", "SHORT", 6, "low", 0.5521, -0.7, 1.2, -0.5, 1.9e8, 52),
    ("FETUSDT", "FET", "LONG", 9, "high", 1.394, 2.8, 2.7, 5.3, 2.6e8, 11),
    ("PEPEUSDT", "PEPE", "LONG", 6, "low", 0.00000921, 1.9, 1.8, 0.7, 5.4e8, 58),
    ("ETHUSDT", "ETH", "SHORT", 6, "low", 2618.4, 0.4, 1.1, -0.4, 7.8e9, 49),
]
signals = [
    {"symbol": s, "base_asset": b, "direction": d, "score": sc, "confidence": c, "price": p,
     "price_change_1h_pct": h1, "volume_ratio": vr, "oi_change_pct": oi,
     "quote_volume_24h": qv, "detected_at": iso(BASE - timedelta(minutes=m))}
    for s, b, d, sc, c, p, h1, vr, oi, qv, m in SIGNALS
]  # fmt: skip
signals.sort(key=lambda r: (-r["score"], r["detected_at"]))
dump("scanner.json", {**header(), "signals": signals})

# --- sectors -----------------------------------------------------------------------
SECTORS = [
    ("ai", "crypto", 74.2, 6.8, 4.9, 7.2, 0.82, 1.41),
    ("layer-1", "crypto", 63.5, 2.1, 2.6, 3.0, 0.68, 1.08),
    ("layer-2", "crypto", 41.8, -3.4, -1.7, -0.9, 0.35, 0.82),
    ("defi", "crypto", 55.9, 1.2, 1.8, 2.2, 0.57, 0.97),
    ("meme", "crypto", 47.3, -5.9, -2.8, 4.1, 0.31, 0.74),
    ("gaming", "crypto", 36.4, -1.8, -1.2, -2.5, 0.29, 0.69),
    ("rwa", "crypto", 68.7, 4.4, 3.5, 5.6, 0.75, 1.22),
    ("depin", "crypto", 58.1, 3.0, 2.2, 1.4, 0.61, 1.02),
    ("exchange-tokens", "crypto", 52.6, 0.4, 0.9, 0.3, 0.50, 0.91),
    ("privacy", "crypto", 44.0, -0.7, -0.4, -1.1, 0.42, 0.85),
    ("us-indices", "tradfi", 61.2, 1.5, 0.8, 0.6, 0.71, 0.88),
    ("precious-metals", "tradfi", 66.9, 2.7, 1.3, 2.0, 0.80, 0.93),
    ("energy", "tradfi", 39.5, -2.2, -1.6, -0.8, 0.33, 0.62),
]
sectors = [
    {"slug": s, "asset_class": ac, "sector_score": sc, "sector_score_delta_24h": dsc,
     "price_change_24h_pct": pc, "oi_change_24h_pct": oc, "breadth_ratio": br,
     "relative_strength_btc": rs}
    for s, ac, sc, dsc, pc, oc, br, rs in SECTORS
]  # fmt: skip
dump("sectors.json", {**header(), "sectors": sectors})


# --- macro regime: 7 days at 15-minute resolution ------------------------------------
n = 7 * 24 * 4
macro_points = []
for i in range(n):
    back = n - 1 - i  # quarter-hours before BASE
    days_back = back / 96
    # Dip into Cautious ~3 days ago, recovering to 54.3 now.
    trend = 54.3 - 12.0 * math.exp(-((days_back - 3.0) ** 2) / 0.6) + 3.0 * (days_back / 7)
    score = 54.3 if back == 0 else round(trend + rng.gauss(0, 0.8), 1)
    macro_points.append({"ts": iso(BASE - timedelta(minutes=15 * back)), "score": score,
                         "regime": regime_for_score(score), "panic_override": False})  # fmt: skip
dump(
    "macro.json",
    {**header(), "current": {"score": 54.3, "panic_override": False}, "history": macro_points},
)

print(f"wrote fixtures to {OUT}")
