"""Resolve a ``StructuredAnswer`` against the tool results of the run that produced it.

``resolve_answer`` is pure: it finds the call each ``ValueRef`` refers to, follows its
pointer into the tool result the model saw (the ``tool`` messages of the transcript),
formats the value and substitutes it for its ``{id}`` placeholders. Problems never raise;
they come back as a list of typed ``ResolutionError``s so the caller can reject or
regenerate the answer.

A ``call_ref`` is matched, in this order, as the call's alias (``t1`` for the first
executed call, ``t2`` for the second, ...; see ``call_alias``), as its real tool call id,
or as a tool name that exactly one successful call used.

Only placeholders are replaced. A number the model wrote as a literal in the text is
left as it is; flagging such numbers is the job of ``guard.py``.
"""

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from mirkaso_copilot.answer import StructuredAnswer, ValueRef

PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)\}")
_ARRAY_INDEX = re.compile(r"^(0|[1-9][0-9]*)$")
# Last pointer token naming a change: such values get an explicit sign.
_CHANGE = re.compile(r"change|delta")
_COMPACT_USD = ((1e6, "M"), (1e9, "B"), (1e12, "T"))


class ResolutionErrorCode(StrEnum):
    UNKNOWN_PLACEHOLDER = "unknown_placeholder"
    DUPLICATE_VALUE_ID = "duplicate_value_id"
    UNKNOWN_CALL_REF = "unknown_call_ref"
    AMBIGUOUS_CALL_REF = "ambiguous_call_ref"
    TOOL_CALL_FAILED = "tool_call_failed"
    INVALID_POINTER = "invalid_pointer"
    MISSING_PATH = "missing_path"
    WRONG_TYPE = "wrong_type"


class ToolRecord(Protocol):
    """The parts of ``agent.ToolCallRecord`` the resolver needs."""

    @property
    def tool_call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def ok(self) -> bool: ...
    @property
    def source_id(self) -> str | None: ...
    @property
    def as_of(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ResolutionError:
    code: ResolutionErrorCode
    value_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class ResolvedValue:
    id: str
    kind: str
    value: Any
    formatted: str
    tool: str
    call_ref: str  # the call's alias, e.g. t1, whatever form the answer used
    tool_call_id: str
    source_id: str | None
    as_of: str | None
    path: str


@dataclass(slots=True)
class ResolvedAnswer:
    summary: str
    claims: list[str]
    caveats: list[str]
    text: str
    values: list[ResolvedValue] = field(default_factory=list)
    errors: list[ResolutionError] = field(default_factory=list)
    unused_values: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class _Missing(Exception):
    pass


class _BadPointer(Exception):
    pass


class _WrongType(Exception):
    pass


def call_alias(index: int) -> str:
    """Alias of the ``index``-th executed tool call (0-based): ``t1``, ``t2``, ..."""
    return f"t{index + 1}"


def tool_results(messages: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The decoded result of every ``tool`` message, by tool call id."""
    out: dict[str, Any] = {}
    for m in messages:
        if m.get("role") != "tool" or "tool_call_id" not in m:
            continue
        try:
            out[m["tool_call_id"]] = json.loads(m.get("content") or "null")
        except (TypeError, ValueError):
            continue
    return out


def follow_pointer(doc: Any, pointer: str) -> Any:
    """RFC 6901 JSON Pointer lookup. Raises ``_BadPointer`` or ``_Missing``."""
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise _BadPointer
    node = doc
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                raise _Missing
            node = node[token]
        elif isinstance(node, list):
            if not _ARRAY_INDEX.match(token) or int(token) >= len(node):
                raise _Missing
            node = node[int(token)]
        else:
            raise _Missing
    return node


def _trim(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fixed(value: float, decimals: int | None, default_max: int) -> str:
    """Thousands separators; ``decimals`` places, else up to ``default_max`` trimmed.

    Never scientific notation. Without ``decimals``, a value too small to show in
    ``default_max`` places is written out in full (``0.0000000123``), not as ``0``.
    """
    if decimals is not None:
        text = f"{value:,.{decimals}f}"
    else:
        text = _trim(f"{value:,.{default_max}f}")
        if value != 0 and not any(c in "123456789" for c in text):
            text = _trim(format(Decimal(repr(value)), "f"))
    if text.startswith("-") and not any(c in "123456789" for c in text):
        text = text[1:]  # "-0.00" after rounding
    return text


def _signed(text: str, value: float, signed: bool) -> str:
    if signed and value > 0 and any(c in "123456789" for c in text):
        return "+" + text
    return text


def _usd(value: float, decimals: int | None) -> str:
    """``$152.31``, ``$420,000``; from one million on ``$56.87M``, ``$12.1B``, ``$1.5T``."""
    sign, amount = ("-" if value < 0 else ""), abs(value)
    units = [unit for unit in _COMPACT_USD if amount >= unit[0]]
    if not units:
        if decimals is None and amount >= 1:
            decimals = 0 if amount.is_integer() else 2
        return f"{sign}${_fixed(amount, decimals, 6)}"
    i = len(units) - 1
    mantissa = _fixed(amount / _COMPACT_USD[i][0], decimals, 2)
    if i + 1 < len(_COMPACT_USD) and float(mantissa.replace(",", "")) >= 1000:
        i += 1  # 999.999M rounds to 1,000M: show 1B instead
        mantissa = _fixed(amount / _COMPACT_USD[i][0], decimals, 2)
    return f"{sign}${mantissa}{_COMPACT_USD[i][1]}"


def _number(raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise _WrongType
    return float(raw)


def format_value(raw: Any, kind: str, decimals: int | None, path: str = "") -> str:
    """Format a resolved value for its kind. Raises ``_WrongType`` for a mismatch."""
    if kind == "text":
        if not isinstance(raw, str):
            raise _WrongType
        return raw
    if kind == "timestamp":
        if not isinstance(raw, str):
            raise _WrongType
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError:
            raise _WrongType from None
        if ts.tzinfo is None:
            raise _WrongType
        return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")

    value = _number(raw)
    last = path.rsplit("/", 1)[-1]
    signed = bool(_CHANGE.search(last))
    if kind == "count":
        if not value.is_integer():
            raise _WrongType
        return f"{int(value):,}"
    if kind == "usd":
        return _usd(value, decimals)
    if kind == "percent":
        return _signed(_fixed(value, decimals, 2), value, signed) + "%"
    if kind == "ratio":
        return _fixed(value, decimals, 2)
    return _signed(_fixed(value, decimals, 6), value, signed)


_EXPECTED = {
    "text": "a string",
    "timestamp": "an ISO-8601 time with a time zone",
    "count": "a whole number",
}


def find_call(
    call_ref: str, records: Sequence[ToolRecord]
) -> tuple[int | None, ResolutionErrorCode | None]:
    """Index of the record ``call_ref`` names, or the error code if there is none.

    Tried in order: alias (``t1``), real tool call id, then a tool name that exactly one
    successful call used. A name used by several successful calls is ambiguous; a name
    used only by failed calls is ``tool_call_failed``.
    """
    for i in range(len(records)):
        if call_alias(i) == call_ref:
            return i, None
    for i, r in enumerate(records):
        if r.tool_call_id == call_ref:
            return i, None
    named = [i for i, r in enumerate(records) if r.name == call_ref and r.ok]
    if len(named) == 1:
        return named[0], None
    if len(named) > 1:
        return None, ResolutionErrorCode.AMBIGUOUS_CALL_REF
    if any(r.name == call_ref for r in records):
        return None, ResolutionErrorCode.TOOL_CALL_FAILED
    return None, ResolutionErrorCode.UNKNOWN_CALL_REF


def _resolve_one(
    ref: ValueRef, records: Sequence[ToolRecord], results: Mapping[str, Any]
) -> ResolvedValue | ResolutionError:
    def error(code: ResolutionErrorCode, message: str) -> ResolutionError:
        return ResolutionError(code=code, value_id=ref.id, message=message)

    index, code = find_call(ref.call_ref, records)
    if code is ResolutionErrorCode.AMBIGUOUS_CALL_REF:
        return error(code, f"{ref.call_ref!r} matches several calls; use the alias, e.g. t1.")
    if code is ResolutionErrorCode.TOOL_CALL_FAILED:
        return error(code, f"Every {ref.call_ref!r} call returned an error, not data.")
    if index is None or records[index].tool_call_id not in results:
        return error(ResolutionErrorCode.UNKNOWN_CALL_REF, f"No call {ref.call_ref!r}.")
    record, alias = records[index], call_alias(index)
    if not record.ok:
        return error(
            ResolutionErrorCode.TOOL_CALL_FAILED, f"Call {alias} returned an error, not data."
        )
    try:
        raw = follow_pointer(results[record.tool_call_id], ref.path)
    except _BadPointer:
        return error(ResolutionErrorCode.INVALID_POINTER, f"{ref.path!r} is not a JSON Pointer.")
    except _Missing:
        return error(
            ResolutionErrorCode.MISSING_PATH,
            f"{ref.path!r} does not exist in the result of {alias}.",
        )
    try:
        formatted = format_value(raw, ref.kind, ref.decimals, ref.path)
    except _WrongType:
        expected = _EXPECTED.get(ref.kind, "a number")
        return error(
            ResolutionErrorCode.WRONG_TYPE,
            f"{ref.path!r} holds {type(raw).__name__}, but kind {ref.kind!r} needs {expected}.",
        )
    return ResolvedValue(
        id=ref.id,
        kind=ref.kind,
        value=raw,
        formatted=formatted,
        tool=record.name,
        call_ref=alias,
        tool_call_id=record.tool_call_id,
        source_id=record.source_id,
        as_of=record.as_of,
        path=ref.path,
    )


def _render(text: str, formatted: Mapping[str, str]) -> str:
    """Substitute resolved placeholders; unresolved ones stay as ``{id}``."""
    return PLACEHOLDER.sub(lambda m: formatted.get(m.group(1), m.group(0)), text)


def render_text(summary: str, claims: Sequence[str], caveats: Sequence[str]) -> str:
    lines = [summary]
    lines += [f"- {c}" for c in claims]
    lines += [f"Caveat: {c}" for c in caveats]
    return "\n".join(lines)


def resolve_answer(
    answer: StructuredAnswer,
    tool_records: Iterable[ToolRecord],
    messages: Iterable[Mapping[str, Any]],
) -> ResolvedAnswer:
    """Resolve every value of ``answer`` against the run's tool records and transcript."""
    records = list(tool_records)
    results = tool_results(messages)
    errors: list[ResolutionError] = []
    resolved: list[ResolvedValue] = []
    declared_order: list[str] = []

    for ref in answer.values:
        if ref.id in declared_order:
            errors.append(
                ResolutionError(
                    code=ResolutionErrorCode.DUPLICATE_VALUE_ID,
                    value_id=ref.id,
                    message=f"Value id {ref.id!r} is defined more than once.",
                )
            )
            continue
        declared_order.append(ref.id)
        outcome = _resolve_one(ref, records, results)
        (errors if isinstance(outcome, ResolutionError) else resolved).append(outcome)

    texts = [answer.summary, *(c.text for c in answer.claims), *answer.caveats]
    used: list[str] = []
    for text in texts:
        for placeholder in PLACEHOLDER.findall(text):
            if placeholder not in used:
                used.append(placeholder)
    for placeholder in used:
        if placeholder not in declared_order:
            errors.append(
                ResolutionError(
                    code=ResolutionErrorCode.UNKNOWN_PLACEHOLDER,
                    value_id=placeholder,
                    message=f"Placeholder {{{placeholder}}} has no entry in values.",
                )
            )

    formatted = {v.id: v.formatted for v in resolved}
    summary = _render(answer.summary, formatted)
    claims = [_render(c.text, formatted) for c in answer.claims]
    caveats = [_render(c, formatted) for c in answer.caveats]
    return ResolvedAnswer(
        summary=summary,
        claims=claims,
        caveats=caveats,
        text=render_text(summary, claims, caveats),
        values=resolved,
        errors=errors,
        unused_values=[ref_id for ref_id in declared_order if ref_id not in used],
    )
