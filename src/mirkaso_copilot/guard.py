"""Numeric grounding guard for structured answers. Pure functions, no I/O.

``check_answer`` runs after ``resolve_answer`` and collects everything that makes an
answer unfit to show:

- **resolution errors** from the resolver (a value that does not point at tool data);
- **literal numbers**: digits the model wrote into summary, claims or caveats itself
  instead of a ``{vN}`` placeholder (``literal_numbers``), checked on the model's text
  *before* substitution, so the server's formatted values are never flagged;
- **advice wording**: investment-advice phrases (``advice_phrases``), checked on the
  rendered text.

The advice check is injectable: ``check_answer(..., output_guard=...)`` takes any
callable from text to a list of offending phrases. The private Mirkaso product passes
its existing output guard there; ``advice_phrases`` is the conservative default for
this repository.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mirkaso_copilot.answer import StructuredAnswer
from mirkaso_copilot.resolve import ResolutionError, ResolvedAnswer

OutputGuard = Callable[[str], Sequence[str]]

PLACEHOLDER = re.compile(r"\{v[0-9]{1,3}\}")
# A run of characters that can make up a written number or a name: "$420,000", "8.5e-05",
# "11/13", "-2.7%", "layer-1s", "price_change_1h_pct", "1h/4h/24h".
_TOKEN = re.compile(r"[\w$€£%.,/:+\-]+")
_EDGE_PUNCTUATION = ".,:;!?+-/"

# Parts of a token that may contain digits:
#   timeframe        15m, 1h, 4h, 24h, 7d, 30d, 2w     (lower-case unit only, so 15M is a number)
#   letter-led word  x402, L2, web3, BTC2, layer        (names and tickers)
#   digit-led ticker 1INCH, 1000PEPEUSDT                 (at least 3 capitals after the digits)
_TIMEFRAME = r"[0-9]+[mhdw]"
_WORD = r"[A-Za-z][A-Za-z0-9]*"
_TICKER = r"[0-9]+[A-Z]{3}[A-Z0-9]*"
_PART = rf"(?:{_TIMEFRAME}|{_WORD}|{_TICKER})"
# A token is allowed if it is a snake_case field name (price_change_1h_pct), or parts
# joined by "/" (1h/4h/24h) or "-" (x402-style, 24h-high), where a bare number may only
# follow a word after a hyphen (layer-1, layer-1s). Anything else with a digit is a
# literal number: amounts, decimals, scientific notation, fractions (11/13), percentages,
# times and dates, ordinals (1st, 2nd) and spelled-with-digits timeframes (24-hour).
ALLOWED_TOKEN = re.compile(
    rf"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"
    rf"|{_PART}(?:/{_PART})*"
    rf"|{_PART}(?:-(?:{_PART}|[0-9]+s?))+"
)

ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byou (?:should|could|might want to|may want to|need to|must) (?:consider )?"
        r"(?:buy|sell|short|long|go long|go short|hold|accumulate|enter|exit|take profits?)\b",
        r"\b(?:i|we) (?:would |strongly )?(?:recommend|suggest|advise)\b",
        r"\bmy recommendation\b",
        r"\bgo (?:long|short)\b",
        r"\b(?:good|great|right|best|perfect|ideal) (?:time|moment|opportunity|entry) to\b",
        r"\b(?:buy|sell) (?:now|the dip|the rip)\b",
        r"\bstrong (?:buy|sell)\b",
        r"\bguarantee[ds]?\b",
        r"\bwill (?:likely |probably |definitely |surely |certainly )?"
        r"(?:go (?:up|down)|rise|fall|drop|climb|pump|dump|rally|crash|rebound|recover|moon)\b",
        r"\b(?:price|upside|downside) target\b|\btarget price\b",
    )
)
# "financial advice" is allowed only as a disclaimer: "this is not financial advice".
_ADVICE_DISCLAIMER = re.compile(r"\b(?:financial|investment) advice\b", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:not|no|never)\W+(?:\w+\W+){0,2}$", re.IGNORECASE)


def literal_numbers(text: str) -> list[str]:
    """Numbers written into ``text`` other than ``{vN}`` placeholders and allowed tokens."""
    found = []
    for match in _TOKEN.finditer(PLACEHOLDER.sub(" ", text)):
        token = match.group()
        core = token.strip(_EDGE_PUNCTUATION)
        if any(c.isdigit() for c in core) and not ALLOWED_TOKEN.fullmatch(core):
            found.append(token.rstrip(_EDGE_PUNCTUATION))  # keep a sign: "-2.7%"
    return found


def advice_phrases(text: str) -> list[str]:
    """Investment-advice wording in ``text``. Conservative: English, explicit phrases only."""
    hits = [m.group() for p in ADVICE_PATTERNS for m in p.finditer(text)]
    for m in _ADVICE_DISCLAIMER.finditer(text):
        if not _NEGATED.search(text[max(0, m.start() - 40) : m.start()]):
            hits.append(m.group())
    return hits


@dataclass(frozen=True, slots=True)
class Finding:
    location: str  # "summary", "claim 2", "caveat 1"
    text: str


@dataclass(slots=True)
class GuardResult:
    literal_numbers: list[Finding] = field(default_factory=list)
    advice_hits: list[Finding] = field(default_factory=list)
    resolution_errors: list[ResolutionError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.literal_numbers or self.advice_hits or self.resolution_errors)

    def problems(self) -> list[str]:
        """One short line per problem, for the correction message and the run metrics."""
        lines = [f"{e.value_id}: {e.code} ({e.message})" for e in self.resolution_errors]
        lines += [
            f"literal number '{f.text}' in {f.location}; use a placeholder"
            for f in self.literal_numbers
        ]
        lines += [
            f"advice wording '{f.text}' in {f.location}; describe the data only"
            for f in self.advice_hits
        ]
        return lines


def _parts(summary: str, claims: Sequence[str], caveats: Sequence[str]) -> list[tuple[str, str]]:
    return [
        ("summary", summary),
        *((f"claim {i}", c) for i, c in enumerate(claims, 1)),
        *((f"caveat {i}", c) for i, c in enumerate(caveats, 1)),
    ]


def check_answer(
    answer: StructuredAnswer,
    resolved: ResolvedAnswer,
    *,
    output_guard: OutputGuard | None = None,
) -> GuardResult:
    """Guard a structured answer and its resolution. ``output_guard`` replaces the advice
    check (default ``advice_phrases``) and sees each rendered part separately."""
    guard = output_guard or advice_phrases
    raw = _parts(answer.summary, [c.text for c in answer.claims], answer.caveats)
    rendered = _parts(resolved.summary, resolved.claims, resolved.caveats)
    return GuardResult(
        literal_numbers=[Finding(loc, n) for loc, text in raw for n in literal_numbers(text)],
        advice_hits=[Finding(loc, h) for loc, text in rendered for h in guard(text)],
        resolution_errors=list(resolved.errors),
    )
