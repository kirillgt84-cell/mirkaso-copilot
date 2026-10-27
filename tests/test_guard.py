"""Numeric grounding guard: literal numbers, advice wording, GuardResult. Pure, no network."""

import pytest

from mirkaso_copilot.answer import StructuredAnswer
from mirkaso_copilot.guard import (
    ALLOWED_TOKEN,
    advice_phrases,
    check_answer,
    literal_numbers,
)
from mirkaso_copilot.resolve import ResolutionError, ResolutionErrorCode, ResolvedAnswer


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("$420,000", ["$420,000"]),
        ("8.5e-05", ["8.5e-05"]),
        ("11/13", ["11/13"]),
        ("12100000000.0", ["12100000000.0"]),
        ("2.5%", ["2.5%"]),
        ("Funding is 8.5e-05 and OI is 12100000000.0.", ["8.5e-05", "12100000000.0"]),
        ("price fell -2.7% in 1h", ["-2.7%"]),
        ("OI rose +6.8%", ["+6.8%"]),
        ("score 11 / 13", ["11", "13"]),
        ("about $57M", ["$57M"]),
        ("15M in volume", ["15M"]),  # an upper-case unit is an amount, not a timeframe
        ("$15m", ["$15m"]),
        ("as of 2026-10-27T14:13:34Z.", ["2026-10-27T14:13:34Z"]),
        ("at 14:13 UTC", ["14:13"]),
        ("ranked 1st and 2nd", ["1st", "2nd"]),  # ordinals are not allowed
        ("over the 24-hour period", ["24-hour"]),  # only compact timeframes
        ("breadth 0,5", ["0,5"]),
        ("{v1} and 42", ["42"]),
    ],
)
def test_literal_numbers_are_flagged(text: str, found: list[str]) -> None:
    assert literal_numbers(text) == found


@pytest.mark.parametrize(
    "text",
    [
        "24h",
        "7d",
        "layer-1",
        "Over 1h, 4h, 24h, 7d and 30d, and the 15m chart.",
        "1h/4h/24h changes",
        "the 24h-high",
        "layer-1s and layer-2 sectors lead",
        "x402-style tokens",
        "BTC, ETH, 1INCH and 1000PEPEUSDT",
        "web3 and L2 names",
        "price_change_1h_pct is a 1h move",
        "ai leads with {v1}, {v2} over 24h and {v3} vs BTC.",
        "{v1}{v2}%",  # numbers inside placeholders are ignored
        "No digits at all.",
    ],
)
def test_allowed_tokens_and_placeholders_pass(text: str) -> None:
    assert literal_numbers(text) == []


def test_allow_list_is_a_full_token_match() -> None:
    assert ALLOWED_TOKEN.fullmatch("24h")
    assert not ALLOWED_TOKEN.fullmatch("24h5")
    assert not ALLOWED_TOKEN.fullmatch("1h.5")


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        ("You should buy BTC now.", ["You should buy"]),
        ("you might want to take profit", ["you might want to take profit"]),
        ("I recommend reducing exposure.", ["I recommend"]),
        ("We strongly suggest waiting.", ["We strongly suggest"]),
        ("Go long SOL here.", ["Go long"]),
        ("It is a good time to enter.", ["good time to"]),
        ("buy the dip", ["buy the dip"]),
        ("a strong sell", ["strong sell"]),
        ("Gains are guaranteed.", ["guaranteed"]),
        ("BTC will go up.", ["will go up"]),
        ("SOL will likely drop.", ["will likely drop"]),
        ("Our price target is high.", ["price target"]),
        ("This is financial advice.", ["financial advice"]),
    ],
)
def test_advice_phrases_are_flagged(text: str, hits: list[str]) -> None:
    assert advice_phrases(text) == hits


@pytest.mark.parametrize(
    "text",
    [
        "This is not financial advice.",
        "Not investment advice.",
        "No financial advice is given here.",
        "SOL has a SHORT signal and BTC a LONG signal.",
        "Long liquidations were larger than short liquidations.",
        "Open interest rose while price fell over 1h.",
        "The regime is Neutral.",
    ],
)
def test_descriptive_text_and_disclaimers_pass(text: str) -> None:
    assert advice_phrases(text) == []


def structured(summary: str, claims: list[str], caveats: list[str] = ()) -> StructuredAnswer:
    return StructuredAnswer.model_validate(
        {
            "summary": summary,
            "claims": [{"text": c} for c in claims],
            "values": [],
            "caveats": list(caveats),
        }  # fmt: skip
    )


def rendered(a: StructuredAnswer, errors=()) -> ResolvedAnswer:
    claims = [c.text for c in a.claims]
    return ResolvedAnswer(a.summary, claims, list(a.caveats), "", errors=list(errors))


def test_guard_result_collects_every_problem_with_its_location() -> None:
    a = structured("Total {v1}.", ["Funding 8.5e-05.", "BTC will go up."], ["As of 2026."])
    error = ResolutionError(ResolutionErrorCode.UNKNOWN_CALL_REF, "v3", "No call 'x'.")

    result = check_answer(a, rendered(a, [error]))

    assert not result.ok
    assert [(f.location, f.text) for f in result.literal_numbers] == [
        ("claim 1", "8.5e-05"),
        ("caveat 1", "2026"),
    ]
    assert [(f.location, f.text) for f in result.advice_hits] == [("claim 2", "will go up")]
    assert result.problems() == [
        "v3: unknown_call_ref (No call 'x'.)",
        "literal number '8.5e-05' in claim 1; use a placeholder",
        "literal number '2026' in caveat 1; use a placeholder",
        "advice wording 'will go up' in claim 2; describe the data only",
    ]


def test_clean_answer_passes() -> None:
    a = structured("Total {v1} over 24h.", ["Longs {v2}."])
    result = check_answer(a, rendered(a))
    assert result.ok and result.problems() == []


def test_literal_numbers_are_checked_before_substitution() -> None:
    a = structured("Total {v1}.", [])
    after = ResolvedAnswer("Total $56.87M.", [], [], "Total $56.87M.")
    assert check_answer(a, after).ok  # the server's formatted value is not a literal


def test_output_guard_is_injectable() -> None:
    seen: list[str] = []

    def product_guard(text: str) -> list[str]:
        seen.append(text)
        return ["moon"] if "moon" in text else []

    a = structured("BTC to the moon.", ["You should buy."])
    result = check_answer(a, rendered(a), output_guard=product_guard)
    assert [(f.location, f.text) for f in result.advice_hits] == [("summary", "moon")]
    assert seen == ["BTC to the moon.", "You should buy."]  # replaces the default check
