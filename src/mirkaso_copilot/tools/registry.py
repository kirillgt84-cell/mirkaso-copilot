"""Tool registry: OpenAI-format definitions and a single safe entry point for execution."""

import json
import logging
from types import ModuleType
from typing import Any

from pydantic import ValidationError

from mirkaso_copilot.envelope import ErrorCode, error_result
from mirkaso_copilot.providers import CallerContext, DataProvider, NotFound, ProviderUnavailable
from mirkaso_copilot.tools import (
    coin_get_derivatives,
    macro_get_regime,
    market_get_liquidations,
    scanner_get_signals,
    sectors_get_indices,
)

logger = logging.getLogger(__name__)

_TOOLS: dict[str, ModuleType] = {
    m.NAME: m
    for m in (
        scanner_get_signals,
        sectors_get_indices,
        macro_get_regime,
        coin_get_derivatives,
        market_get_liquidations,
    )
}

_MAX_FIELD_NAME = 40


def tool_names() -> list[str]:
    return list(_TOOLS)


def tool_definitions() -> list[dict[str, Any]]:
    """Tool definitions in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": m.NAME,
                "description": m.DESCRIPTION,
                "parameters": m.parameters(),
            },
        }
        for m in _TOOLS.values()
    ]


def _invalid_fields_message(exc: ValidationError) -> str:
    fields = sorted(
        {".".join(str(p) for p in err["loc"])[:_MAX_FIELD_NAME] or "(root)" for err in exc.errors()}
    )
    return "Invalid arguments: " + ", ".join(fields) + "."


def canonical_call(name: str, args: dict[str, Any]) -> str | None:
    """Canonical JSON of the validated params (defaults applied), or None if not valid.

    Two calls with the same name and canonical params request the same data.
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    try:
        params = module.Params.model_validate(args)
    except ValidationError:
        return None
    return json.dumps(params.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


async def execute_tool(
    name: str,
    args: str | bytes | dict[str, Any] | None,
    provider: DataProvider,
    ctx: CallerContext,
) -> dict[str, Any]:
    """Validate ``args`` and run tool ``name``. Never raises; failures become error dicts."""
    module = _TOOLS.get(name)
    if module is None:
        return error_result(ErrorCode.UNKNOWN_TOOL).model_dump(mode="json")

    try:
        raw = json.loads(args) if isinstance(args, str | bytes) else (args or {})
        if not isinstance(raw, dict):
            raise TypeError("arguments must be an object")
        params = module.Params.model_validate(raw)
    except ValidationError as exc:
        return error_result(ErrorCode.INVALID_ARGUMENTS, _invalid_fields_message(exc)).model_dump(
            mode="json"
        )
    except (ValueError, TypeError):
        return error_result(
            ErrorCode.INVALID_ARGUMENTS, "Arguments must be a JSON object."
        ).model_dump(mode="json")

    try:
        result = await module.execute(provider, ctx, params)
        return result.model_dump(mode="json")
    except NotFound:
        return error_result(ErrorCode.NOT_FOUND).model_dump(mode="json")
    except ProviderUnavailable:
        logger.warning("tool %s: provider unavailable", name, exc_info=True)
        return error_result(ErrorCode.UPSTREAM_UNAVAILABLE).model_dump(mode="json")
    except Exception:
        logger.exception("tool %s failed", name)
        return error_result(ErrorCode.INTERNAL_ERROR).model_dump(mode="json")
