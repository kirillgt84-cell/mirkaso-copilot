from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mirkaso_copilot.envelope import ErrorCode, ToolResult, error_result, to_utc_iso


def test_to_utc_iso_normalises_offsets() -> None:
    assert to_utc_iso("2026-10-26T14:00:00+02:00") == "2026-10-26T12:00:00Z"
    assert to_utc_iso("2026-10-26T12:00:00Z") == "2026-10-26T12:00:00Z"
    cet = timezone(timedelta(hours=1))
    assert to_utc_iso(datetime(2026, 10, 26, 13, tzinfo=cet)) == "2026-10-26T12:00:00Z"


def test_to_utc_iso_rejects_naive() -> None:
    with pytest.raises(ValueError):
        to_utc_iso("2026-10-26T12:00:00")


def test_tool_result_defaults() -> None:
    result = ToolResult(as_of=datetime(2026, 10, 26, tzinfo=UTC), source_id="x", data={})
    assert result.model_dump() == {
        "as_of": "2026-10-26T00:00:00Z",
        "source_id": "x",
        "gated": False,
        "gated_fields": [],
        "data": {},
    }


def test_gated_fields_require_gated() -> None:
    with pytest.raises(ValidationError):
        ToolResult(as_of="2026-10-26T00:00:00Z", source_id="x", gated_fields=["a"], data={})


def test_error_shape() -> None:
    dumped = error_result(ErrorCode.NOT_FOUND).model_dump(mode="json")
    assert set(dumped) == {"error"}
    assert set(dumped["error"]) == {"code", "message"}
    assert dumped["error"]["code"] == "not_found"
