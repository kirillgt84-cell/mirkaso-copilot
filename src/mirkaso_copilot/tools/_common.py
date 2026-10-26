"""Helpers shared by the tool modules."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from mirkaso_copilot.envelope import utc_now

T = TypeVar("T")

MAX_HISTORY_POINTS = 48


def parameters_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a parameter model, trimmed for function calling.

    Drops pydantic titles and collapses ``Optional[X]`` (``anyOf [X, null]``) to ``X``
    so the model sees a plain schema. ``additionalProperties: false`` comes from
    ``extra="forbid"`` on the model.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema["properties"] = {
        name: _clean_property(prop) for name, prop in schema.get("properties", {}).items()
    }
    return schema


def _clean_property(prop: dict[str, Any]) -> dict[str, Any]:
    prop = {k: v for k, v in prop.items() if k != "title"}
    any_of = prop.pop("anyOf", None)
    if any_of is not None:
        non_null = [s for s in any_of if s.get("type") != "null"]
        if len(non_null) == 1:
            prop = {**non_null[0], **prop}
        else:
            prop["anyOf"] = any_of
    if "default" in prop and prop["default"] is None:
        del prop["default"]
    return prop


def pick(row: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """Whitelist ``fields`` from ``row``; missing keys become ``None``."""
    return {f: row.get(f) for f in fields}


def as_of_from(payload: Mapping[str, Any]) -> datetime | str:
    value = payload.get("as_of")
    return value if value is not None else utc_now()


def parse_ts(value: datetime | str) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def chunk_evenly(items: Sequence[T], max_chunks: int) -> list[Sequence[T]]:
    """Split ``items`` into at most ``max_chunks`` contiguous, non-empty chunks.

    If there are no more items than chunks, each item is its own chunk.
    """
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    n = len(items)
    if n <= max_chunks:
        return [items[i : i + 1] for i in range(n)]
    bounds = [i * n // max_chunks for i in range(max_chunks + 1)]
    return [items[bounds[i] : bounds[i + 1]] for i in range(max_chunks)]
