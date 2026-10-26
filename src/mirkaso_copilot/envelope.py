"""Shared result envelope returned by every tool.

A successful call returns a ``ToolResult``; a failed call returns an ``ErrorResult``
shaped as ``{"error": {"code": ..., "message": ...}}``. Error messages are fixed
strings chosen by this package: they never carry exception text, SQL or tracebacks.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_utc_iso(value: datetime | str) -> str:
    """Normalise an aware datetime or ISO-8601 string to ``YYYY-MM-DDTHH:MM:SSZ``."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str = Field(description="UTC ISO-8601 time the underlying data refers to.")
    source_id: str = Field(description="Stable identifier of the data source.")
    gated: bool = Field(default=False, description="True if some fields are hidden for this plan.")
    gated_fields: list[str] = Field(default_factory=list)
    data: Any

    @field_validator("as_of", mode="before")
    @classmethod
    def _as_of_utc(cls, value: datetime | str) -> str:
        return to_utc_iso(value)

    @model_validator(mode="after")
    def _gated_fields_imply_gated(self) -> "ToolResult":
        if self.gated_fields and not self.gated:
            raise ValueError("gated_fields set on an ungated result")
        return self


class LiquidationsResult(ToolResult):
    newest_event_at: str | None = Field(
        default=None, description="UTC ISO-8601 time of the newest liquidation in the period."
    )

    @field_validator("newest_event_at", mode="before")
    @classmethod
    def _newest_utc(cls, value: datetime | str | None) -> str | None:
        return None if value is None else to_utc_iso(value)


class ErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    INTERNAL_ERROR = "internal_error"
    TIMEOUT = "timeout"
    DUPLICATE_CALL = "duplicate_call"


DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNKNOWN_TOOL: "Unknown tool.",
    ErrorCode.INVALID_ARGUMENTS: "Invalid arguments.",
    ErrorCode.NOT_FOUND: "No data found for the requested item.",
    ErrorCode.UPSTREAM_UNAVAILABLE: "Market data is temporarily unavailable. Try again later.",
    ErrorCode.INTERNAL_ERROR: "The tool failed unexpectedly.",
    ErrorCode.TIMEOUT: "The tool did not respond in time.",
    ErrorCode.DUPLICATE_CALL: "This exact call was already made; reuse its earlier result.",
}


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str


class ErrorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody


def error_result(code: ErrorCode, message: str | None = None) -> ErrorResult:
    return ErrorResult(error=ErrorBody(code=code, message=message or DEFAULT_MESSAGES[code]))
