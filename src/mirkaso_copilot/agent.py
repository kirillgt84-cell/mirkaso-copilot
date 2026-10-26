"""Tool-calling agent loop over the Meta Model API.

``run_agent`` sends the question with every tool definition, executes the tool calls the
model asks for (concurrently, each under its own timeout), feeds the results back and
repeats until the model answers without tool calls or a limit is hit. It never raises
for tool or model failures: they end up as structured tool errors or as a stop reason.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from mirkaso_copilot.envelope import ErrorCode, error_result
from mirkaso_copilot.meta_client import REASONING_EFFORTS, describe_error, parse_usage
from mirkaso_copilot.providers import CallerContext, DataProvider
from mirkaso_copilot.tools.registry import canonical_call, execute_tool, tool_definitions

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a market-data assistant for Mirkaso.
- Call the tools for any market number you mention. Use ONLY numbers returned by tools; \
never invent or estimate a value.
- If a field is null and the tool result says gated=true, say that the value is not \
available on the user's plan. Never treat null as zero.
- price_change_1h_pct is a 1-hour price move, not a daily one.
- Mention the as_of time of the data you use.
- Give no investment advice: no buy or sell recommendations and no price predictions.
- If the data needed to answer is not available through the tools, say so."""

FALLBACK_ANSWERS: dict[str, str] = {
    "max_steps": "I could not finish answering within the allowed number of steps. "
    "Please try a narrower question.",
    "max_tool_calls": "I could not finish answering within the allowed number of data "
    "lookups. Please try a narrower question.",
    "max_tokens": "I could not finish answering within the allowed response budget. "
    "Please try a narrower question.",
    "timeout": "I could not finish answering in time. Please try again.",
    "error": "I could not answer because the assistant service failed. Please try again.",
}


class ChatClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_steps: int = 6
    max_tool_calls_total: int = 12
    # Reasoning tokens count toward completion tokens: small values truncate answers.
    request_max_tokens: int = 3000
    reasoning_effort: str = "low"
    total_timeout_s: float = 90.0
    per_tool_timeout_s: float = 15.0
    max_total_tokens: int = 40000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
        for name in ("max_steps", "max_tool_calls_total", "request_max_tokens",
                     "total_timeout_s", "per_tool_timeout_s", "max_total_tokens"):  # fmt: skip
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(slots=True)
class AgentUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def add(self, body: dict[str, Any]) -> None:
        usage = parse_usage(body)
        prompt = usage.prompt_tokens or 0
        completion = usage.completion_tokens or 0
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.reasoning_tokens += usage.reasoning_tokens or 0
        self.total_tokens += (
            usage.total_tokens if usage.total_tokens is not None else prompt + completion
        )


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    step: int
    name: str
    args: dict[str, Any] | None  # None when the model sent arguments that are not a JSON object
    ok: bool
    error_code: str | None
    latency_ms: float
    source_id: str | None
    as_of: str | None


@dataclass(slots=True)
class AgentResult:
    answer: str | None
    stop_reason: str  # final|max_steps|max_tool_calls|max_tokens|timeout|truncated|error
    steps: int
    tool_calls: list[ToolCallRecord]
    usage: AgentUsage
    latency_ms: float
    messages: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    # Opt-in diagnostics, one dict per warning that has details (see
    # ``meta_client.describe_error``). Never sent to the model and never logged.
    warnings_detail: list[dict[str, Any]] = field(default_factory=list)


class _Stop(Exception):
    def __init__(
        self, reason: str, warning: str | None = None, detail: dict[str, Any] | None = None
    ) -> None:
        self.reason = reason
        self.warning = warning
        self.detail = detail


def _parse_args(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class _Run:
    def __init__(
        self,
        question: str,
        provider: DataProvider,
        ctx: CallerContext,
        client: ChatClient,
        config: AgentConfig,
    ) -> None:
        self.provider = provider
        self.ctx = ctx
        self.client = client
        self.config = config
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": question},
        ]
        self.steps = 0
        self.records: list[ToolCallRecord] = []
        self.usage = AgentUsage()
        self.seen: set[tuple[str, str]] = set()
        self.warnings: list[str] = []
        self.warnings_detail: list[dict[str, Any]] = []

    async def loop(self) -> str:
        tools = tool_definitions()
        while True:
            self.steps += 1
            try:
                body = await self.client.chat(
                    self.messages,
                    tools=tools,
                    tool_choice="auto",
                    reasoning_effort=self.config.reasoning_effort,
                    max_tokens=self.config.request_max_tokens,
                )
                choice = body["choices"][0]
                message = choice["message"]
            except Exception as exc:
                logger.warning("agent step %d: model request failed (%s)", self.steps,
                               type(exc).__name__)  # fmt: skip
                raise _Stop("error", "model_request_failed", describe_error(exc)) from None
            self.usage.add(body)

            if choice.get("finish_reason") == "length":
                raise _Stop("truncated", "finish_reason_length")

            calls = message.get("tool_calls") or []
            content = message.get("content")
            self.messages.append(self._assistant_message(content, calls))
            if not calls:
                if not isinstance(content, str) or not content.strip():
                    raise _Stop("error", "empty_answer")
                return content

            if self.usage.total_tokens > self.config.max_total_tokens:
                raise _Stop("max_tokens")
            if self.steps >= self.config.max_steps:
                raise _Stop("max_steps")
            if len(self.records) + len(calls) > self.config.max_tool_calls_total:
                raise _Stop("max_tool_calls")

            outcomes = await asyncio.gather(*self._schedule(calls))
            for call, (result, record) in zip(calls, outcomes, strict=True):
                self.records.append(record)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

    @staticmethod
    def _assistant_message(content: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        return msg

    def _schedule(self, calls: list[dict[str, Any]]) -> list[Any]:
        """Normalise calls in place and return one awaitable per call.

        Duplicate detection happens here, in call order, so two identical calls in the
        same parallel batch also count as a duplicate. Calls are compared by their
        validated params, so omitted defaults and explicit defaults match. Invalid calls
        are never duplicates: they get their own structured error.
        """
        aws = []
        for i, call in enumerate(calls):
            call.setdefault("id", f"call_{self.steps}_{i}")
            call.setdefault("type", "function")
            fn = call.get("function") or {}
            call["function"] = fn
            name = str(fn.get("name") or "")
            args = _parse_args(fn.get("arguments"))
            canonical = canonical_call(name, args) if args is not None else None
            key = (name, canonical)
            if canonical is not None and key in self.seen:
                aws.append(self._finish(name, args, error_result(ErrorCode.DUPLICATE_CALL), 0.0))
            else:
                if canonical is not None:
                    self.seen.add(key)
                aws.append(self._execute(name, args))
        return aws

    async def _finish(
        self, name: str, args: dict[str, Any] | None, result: dict[str, Any] | Any, ms: float
    ) -> tuple[dict[str, Any], ToolCallRecord]:
        if not isinstance(result, dict):
            result = result.model_dump(mode="json")
        error = result.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        record = ToolCallRecord(
            step=self.steps,
            name=name,
            args=args,
            ok=code is None,
            error_code=code,
            latency_ms=round(ms, 1),
            source_id=result.get("source_id"),
            as_of=result.get("as_of"),
        )
        logger.info(
            "agent tool call step=%d name=%s ok=%s error_code=%s latency_ms=%.1f",
            record.step, record.name, record.ok, record.error_code, record.latency_ms,
        )  # fmt: skip
        return result, record

    async def _execute(
        self, name: str, args: dict[str, Any] | None
    ) -> tuple[dict[str, Any], ToolCallRecord]:
        start = time.perf_counter()
        if args is None:
            result: Any = error_result(
                ErrorCode.INVALID_ARGUMENTS, "Arguments must be a JSON object."
            )
        else:
            try:
                async with asyncio.timeout(self.config.per_tool_timeout_s):
                    result = await execute_tool(name, args, self.provider, self.ctx)
            except TimeoutError:
                result = error_result(ErrorCode.TIMEOUT)
            except Exception as exc:
                logger.warning("agent tool %s raised %s", name, type(exc).__name__)
                result = error_result(ErrorCode.INTERNAL_ERROR)
        return await self._finish(name, args, result, (time.perf_counter() - start) * 1000)


async def run_agent(
    question: str,
    *,
    provider: DataProvider,
    ctx: CallerContext,
    client: ChatClient,
    config: AgentConfig | None = None,
) -> AgentResult:
    config = config or AgentConfig()
    run = _Run(question, provider, ctx, client, config)
    start = time.perf_counter()
    answer: str | None = None
    try:
        async with asyncio.timeout(config.total_timeout_s):
            answer = await run.loop()
        stop_reason = "final"
    except _Stop as stop:
        stop_reason = stop.reason
        if stop.warning:
            run.warnings.append(stop.warning)
            if stop.detail is not None:
                run.warnings_detail.append({"warning": stop.warning, **stop.detail})
    except TimeoutError:
        stop_reason = "timeout"
    if stop_reason not in ("final", "truncated"):
        answer = FALLBACK_ANSWERS[stop_reason]

    result = AgentResult(
        answer=answer,
        stop_reason=stop_reason,
        steps=run.steps,
        tool_calls=run.records,
        usage=run.usage,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        messages=run.messages,
        warnings=run.warnings,
        warnings_detail=run.warnings_detail,
    )
    logger.info(
        "agent run stop_reason=%s steps=%d tool_calls=%d prompt_tokens=%d "
        "completion_tokens=%d reasoning_tokens=%d total_tokens=%d latency_ms=%.1f",
        result.stop_reason, result.steps, len(result.tool_calls), result.usage.prompt_tokens,
        result.usage.completion_tokens, result.usage.reasoning_tokens,
        result.usage.total_tokens, result.latency_ms,
    )  # fmt: skip
    return result
