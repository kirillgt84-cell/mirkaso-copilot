"""macro_get_regime: market-wide risk regime, optionally with history."""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mirkaso_copilot.envelope import ToolResult, to_utc_iso
from mirkaso_copilot.providers import CallerContext, DataProvider
from mirkaso_copilot.tools._common import (
    MAX_HISTORY_POINTS,
    as_of_from,
    chunk_evenly,
    parameters_schema,
    parse_ts,
)

NAME = "macro_get_regime"
SOURCE_ID = "mirkaso.macro"
DESCRIPTION = (
    "Get the current market risk regime: a score and its regime band. The result lists "
    "every band in bands with its label and score range (min inclusive, max exclusive "
    "except for the top band). panic_override=true forces the regime to Panic regardless "
    "of the score. Set include_history=true for up to 48 points over history_days days."
)

PANIC = "Panic"
SCORE_MAX = 100
# (lower bound inclusive, label), checked from the top down.
REGIME_BANDS = (
    (80, "Strong Risk ON"),
    (65, "Risk ON"),
    (50, "Neutral"),
    (35, "Cautious"),
    (20, "Risk OFF"),
    (0, PANIC),
)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_history: bool = Field(False, description="Also return the regime history.")
    history_days: int = Field(7, ge=1, le=30, description="History window in days (1-30).")


def regime_bands() -> list[dict[str, Any]]:
    """The bands in ascending order: ``{label, min, max}``; ``max`` is the next band's min."""
    ascending = sorted(REGIME_BANDS)
    uppers = [lower for lower, _ in ascending[1:]] + [SCORE_MAX]
    return [
        {"label": label, "min": lower, "max": upper}
        for (lower, label), upper in zip(ascending, uppers, strict=True)
    ]


def regime_for_score(score: float, panic_override: bool = False) -> str:
    if panic_override:
        return PANIC
    for lower, label in REGIME_BANDS:
        if score >= lower:
            return label
    return PANIC


def rebucket_macro_history(
    points: Iterable[Mapping[str, Any]], max_points: int = MAX_HISTORY_POINTS
) -> list[dict[str, Any]]:
    """Downsample to at most ``max_points`` time-ordered buckets.

    Per bucket: ts and regime of the latest row, the average score, and
    panic_override if any row had it (which forces the regime to Panic).
    """
    ordered = sorted(points, key=lambda p: parse_ts(p["ts"]))
    out = []
    for bucket in chunk_evenly(ordered, max_points):
        latest = bucket[-1]
        panic = any(bool(p.get("panic_override")) for p in bucket)
        score = sum(float(p["score"]) for p in bucket) / len(bucket)
        regime = PANIC if panic else latest.get("regime") or regime_for_score(latest["score"])
        out.append(
            {
                "ts": to_utc_iso(latest["ts"]),
                "score": round(score, 2),
                "regime": regime,
                "panic_override": panic,
            }
        )
    return out


def parameters() -> dict[str, Any]:
    return parameters_schema(Params)


async def execute(provider: DataProvider, ctx: CallerContext, params: Params) -> ToolResult:
    payload = await provider.macro_regime()
    score = float(payload["score"])
    panic = bool(payload.get("panic_override"))
    data: dict[str, Any] = {
        "score": score,
        "regime": regime_for_score(score, panic),
        "panic_override": panic,
        "bands": regime_bands(),
    }
    if params.include_history:
        history = await provider.macro_regime_history(days=params.history_days)
        data["history"] = rebucket_macro_history(history.get("points", []))
    return ToolResult(as_of=as_of_from(payload), source_id=SOURCE_ID, data=data)
