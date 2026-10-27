"""Structured answer the model writes in the second phase of a structured run.

The model never writes a number itself. Its prose carries placeholders such as ``{v1}``,
and each placeholder has a ``ValueRef`` that points into one tool result by call
reference (an alias such as ``t1`` from the list of calls the model is shown) and JSON
Pointer. The server resolves the pointers and substitutes the formatted values (see
``resolve.py``), so a number in the rendered answer always comes from a tool result.
"""

import copy
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VALUE_ID_PATTERN = re.compile(r"v[0-9]{1,3}")  # ^v[0-9]{1,3}$, via fullmatch
MAX_DECIMALS = 6
MAX_CLAIMS = 6
MAX_VALUES = 24
SCHEMA_NAME = "structured_answer"

ANSWER_RULES = (
    "An answer built only from the tool results above. Never write a number yourself: "
    "put a placeholder such as {v1} wherever a number, price, percentage, ratio, count or "
    "time goes, in summary, claims and caveats alike, and add one entry to values for each "
    "placeholder. Digits may appear in the text only as part of timeframes such as 1h, 4h, "
    "24h, 7d and in asset or sector names. Each value points into one tool result above: "
    "call_ref is the alias of that call from the list of calls (e.g. t1) and path is a "
    "JSON Pointer into its result, e.g. /data/sectors/0/sector_score or /as_of. At most "
    f"{MAX_CLAIMS} claims and {MAX_VALUES} values."
)

ValueKind = Literal["number", "percent", "usd", "ratio", "count", "timestamp", "text"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="One statement; {id} placeholders for every value.")


class ValueRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Placeholder id: v followed by 1 to 3 digits, e.g. v1.")
    call_ref: str = Field(
        description="Alias of the tool call whose result holds the value, e.g. t1."
    )
    path: str = Field(
        description="JSON Pointer into that call's result, e.g. /data/summary/total_usd."
    )
    kind: ValueKind = Field(
        description="usd for US dollar amounts; percent for fields that already are "
        "percentages (names ending in _pct); ratio for ratios; count for numbers of items; "
        "timestamp for times; number for any other number; text for strings such as a "
        "regime label."
    )
    decimals: int | None = Field(
        description="Decimal places to show, from 0 to 6, or null for the default."
    )

    # Checked here, not in the schema: the schema keeps to a strict-mode-friendly subset.
    @field_validator("id")
    @classmethod
    def _id_pattern(cls, value: str) -> str:
        if not VALUE_ID_PATTERN.fullmatch(value):
            raise ValueError("id must be v followed by 1 to 3 digits")
        return value

    @field_validator("decimals")
    @classmethod
    def _decimals_range(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= MAX_DECIMALS:
            raise ValueError(f"decimals must be between 0 and {MAX_DECIMALS}")
        return value


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"description": ANSWER_RULES})

    summary: str = Field(description="A short direct answer to the question.")
    claims: list[Claim] = Field(description="The supporting statements, one per item.")
    values: list[ValueRef] = Field(description="One entry per placeholder used in the text.")
    caveats: list[str] = Field(
        description="Limits of the data, e.g. gated fields or stale data; may be empty."
    )

    # Count limits are checked here, not in the schema, like the ValueRef rules above.
    @model_validator(mode="after")
    def _limits(self) -> "StructuredAnswer":
        if len(self.claims) > MAX_CLAIMS:
            raise ValueError(f"at most {MAX_CLAIMS} claims allowed, got {len(self.claims)}")
        if len(self.values) > MAX_VALUES:
            raise ValueError(f"at most {MAX_VALUES} values allowed, got {len(self.values)}")
        return self


def _strip(node: Any) -> Any:
    """Drop pydantic titles and null defaults; turn ``anyOf`` of bare types into a type list.

    ``Optional[int]`` becomes ``{"type": ["integer", "null"]}``.
    """
    if isinstance(node, dict):
        any_of = node.get("anyOf")
        if any_of is not None and all(set(branch) == {"type"} for branch in any_of):
            node = {k: v for k, v in node.items() if k != "anyOf"}
            node["type"] = [branch["type"] for branch in any_of]
        return {
            k: _strip(v)
            for k, v in node.items()
            if k != "title" and not (k == "default" and v is None)
        }
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if ref is not None:
            return _inline(copy.deepcopy(defs[ref.rsplit("/", 1)[-1]]), defs)
        return {k: _inline(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def answer_json_schema() -> dict[str, Any]:
    """JSON schema of ``StructuredAnswer`` for strict structured outputs.

    Deliberately a conservative subset: every object has ``additionalProperties: false``
    and lists all of its properties as required, ``$ref``s are inlined, ``kind`` is the
    only enum, and there is no ``pattern``, ``minimum``/``maximum`` or ``anyOf``/``oneOf``.
    The id pattern and the decimals range are enforced by the pydantic validators.
    """
    schema = _strip(StructuredAnswer.model_json_schema())
    defs = schema.pop("$defs", {})
    return _inline(schema, defs)


def response_format() -> dict[str, Any]:
    """The ``response_format`` request field for a strict ``StructuredAnswer``."""
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": answer_json_schema()},
    }
