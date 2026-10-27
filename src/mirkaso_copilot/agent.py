"""Tool-calling agent loop over the Meta Model API.

``run_agent`` sends the question with every tool definition, executes the tool calls the
model asks for (concurrently, each under its own timeout), feeds the results back and
repeats until the model answers without tool calls or a limit is hit. It never raises
for tool or model failures: they end up as structured tool errors or as a stop reason.

With ``structured=True`` the answer is verified before it is returned (card 10). A
second phase follows a final answer: one request without tools asks for the answer again
as a ``StructuredAnswer`` (strict JSON schema). Its value pointers are resolved against
the tool results (``resolve.py``) and the result is guarded (``guard.py``). A rejected
answer gets one regeneration with a correction message; if that fails too, the run
returns a fixed refusal without numbers, never the unverified phase A text.
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import ValidationError

from mirkaso_copilot.answer import MAX_CLAIMS, MAX_VALUES, StructuredAnswer, response_format
from mirkaso_copilot.envelope import ErrorCode, error_result
from mirkaso_copilot.guard import OutputGuard, check_answer
from mirkaso_copilot.meta_client import REASONING_EFFORTS, describe_error, parse_usage
from mirkaso_copilot.providers import CallerContext, DataProvider
from mirkaso_copilot.resolve import ResolutionError, ResolvedAnswer, call_alias, resolve_answer
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

PLACEHOLDER_RULES = f"""\
- Never write a number yourself. Wherever a number, price, percentage, ratio, count or \
time goes, in summary, claims and caveats alike, write a placeholder such as {{v1}} and \
add one entry with that id to values. Digits may appear in the text only in timeframes \
written as 15m, 1h, 4h, 24h, 7d or 30d and in asset or sector names.
- Each value points into one tool result: call_ref is the alias of the call that \
returned it (t1, t2, ...), and path is a JSON Pointer into that result exactly as it was \
returned, e.g. /data/sectors/0/sector_score, /data/summary/total_usd or /as_of.
- kind: usd for US dollar amounts, percent for fields that already are percentages \
(names ending in _pct), ratio for ratios, count for numbers of items, timestamp for \
times, number for any other number, text for strings such as a regime label. Set \
decimals to null unless a specific precision matters.
- Do not point at results that are errors.
- Be concise: at most {MAX_CLAIMS} claims and at most {MAX_VALUES} values, short sentences.
- Describe the data only: no advice, recommendations or predictions."""

STRUCTURED_INSTRUCTION = f"""\
Now give your answer again as JSON that matches the response schema. Use only the tool \
results above.
{PLACEHOLDER_RULES}"""

# Appended to the system prompt in single-phase mode, where no call list can be shown.
SINGLE_PHASE_RULES = f"""

When you answer without calling tools, answer as JSON that matches the response schema.
{PLACEHOLDER_RULES}
- Your tool calls are numbered t1, t2, ... in the order you made them across the whole \
conversation: the first call of your first request is t1."""

CORRECTION_INTRO = "Your previous answer was rejected for these problems:"
CORRECTION_OUTRO = (
    "Fix them and give the complete answer again as JSON that matches the response "
    "schema, following the same rules."
)
MAX_CORRECTION_LINES = 12
LENGTH_PROBLEM = "the reply was cut off at the length limit; write fewer, shorter claims"

MAX_STRUCTURED_ATTEMPTS = 2
STRUCTURED_MODES = ("two_phase", "single_phase")
# Contains no number, so a refusal can never carry an unverified value.
REFUSAL_ANSWER = (
    "I couldn't produce a verified answer for this question. "
    "Please try again or ask more specifically."
)

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
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_steps: int = 6
    max_tool_calls_total: int = 12
    # Reasoning tokens count toward completion tokens: small values truncate answers.
    request_max_tokens: int = 3000
    # The structured reply is long JSON and reasoning counts toward it.
    structured_max_tokens: int = 6000
    reasoning_effort: str = "low"
    total_timeout_s: float = 90.0
    per_tool_timeout_s: float = 15.0
    max_total_tokens: int = 40000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
        for name in ("max_steps", "max_tool_calls_total", "request_max_tokens",
                     "structured_max_tokens", "total_timeout_s", "per_tool_timeout_s",
                     "max_total_tokens"):  # fmt: skip
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


@dataclass(slots=True)
class PhaseStats:
    """Usage and wall time of one phase: ``tool``, ``structured_1``, ``structured_2``."""

    name: str
    usage: AgentUsage = field(default_factory=AgentUsage)
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    step: int
    tool_call_id: str
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
    # Usage and latency per phase; ``usage`` and ``latency_ms`` are the totals.
    phases: list[PhaseStats] = field(default_factory=list)
    # Set only by a structured run (``run_agent(..., structured=True)``). ``structured``
    # and ``resolved`` are set only for a verified answer; ``resolution_errors`` are those
    # of the last attempt, for diagnostics.
    verified: bool | None = None
    refusal_reason: str | None = None
    structured: StructuredAnswer | None = None
    resolved: ResolvedAnswer | None = None
    resolution_errors: list[ResolutionError] = field(default_factory=list)
    guard_attempts: int = 0
    guard_rejections: int = 0
    rejected_reasons: list[list[str]] = field(default_factory=list)  # one list per rejection


class _Stop(Exception):
    def __init__(
        self, reason: str, warning: str | None = None, detail: dict[str, Any] | None = None
    ) -> None:
        self.reason = reason
        self.warning = warning
        self.detail = detail


def structured_instruction(records: Sequence[ToolCallRecord]) -> str:
    """The phase B instruction, ending with every executed call and its alias."""
    lines = [STRUCTURED_INSTRUCTION, "", "Tool calls, by alias (use the alias as call_ref):"]
    for i, r in enumerate(records):
        args = json.dumps(r.args, sort_keys=True) if r.args is not None else "invalid arguments"
        ok = f"ok, source {r.source_id}, as_of {r.as_of}"
        status = ok if r.ok else f"error {r.error_code}"
        lines.append(f"{call_alias(i)} = {r.name}({args}) {status}")
    return "\n".join(lines)


def correction_message(problems: Sequence[str]) -> str:
    shown = list(problems[:MAX_CORRECTION_LINES])
    if len(problems) > len(shown):
        shown.append(f"... and {len(problems) - len(shown)} more")
    return "\n".join([CORRECTION_INTRO, *(f"- {p}" for p in shown), CORRECTION_OUTRO])


def _validation_problems(exc: ValidationError) -> list[str]:
    """Location and message of each error; the reply text itself is never included."""
    problems = []
    for err in exc.errors(include_input=False, include_url=False)[:8]:
        loc = ".".join(str(part) for part in err["loc"])
        problems.append(f"invalid answer: {loc}: {err['msg']}" if loc else
                        f"invalid answer: {err['msg']}")  # fmt: skip
    return problems


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
        *,
        single_phase: bool = False,
        output_guard: OutputGuard | None = None,
    ) -> None:
        self.provider = provider
        self.ctx = ctx
        self.client = client
        self.config = config
        self.single_phase = single_phase
        self.output_guard = output_guard
        system = config.system_prompt + (SINGLE_PHASE_RULES if single_phase else "")
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        self.steps = 0
        self.records: list[ToolCallRecord] = []
        self.usage = AgentUsage()
        self.seen: set[tuple[str, str]] = set()
        self.warnings: list[str] = []
        self.warnings_detail: list[dict[str, Any]] = []
        self.phases: list[PhaseStats] = []
        self.phase = PhaseStats("tool")
        self._phase_start = time.perf_counter()
        # Structured verification state.
        self.verified = False
        self.refusal_reason: str | None = None
        self.structured: StructuredAnswer | None = None
        self.resolved: ResolvedAnswer | None = None
        self.resolution_errors: list[ResolutionError] = []
        self.guard_attempts = 0
        self.rejected_reasons: list[list[str]] = []

    def start_phase(self, name: str) -> None:
        self.phase = PhaseStats(name)
        self.phases.append(self.phase)
        self._phase_start = time.perf_counter()

    def end_phase(self) -> None:
        self.phase.latency_ms = round((time.perf_counter() - self._phase_start) * 1000, 1)

    def _add_usage(self, body: dict[str, Any]) -> None:
        self.usage.add(body)
        self.phase.usage.add(body)

    async def loop(self) -> str:
        tools = tool_definitions()
        # Single-phase experiment: the schema goes with every request of the loop.
        extra: dict[str, Any] = {}
        max_tokens = self.config.request_max_tokens
        if self.single_phase:
            extra["response_format"] = response_format()
            max_tokens = self.config.structured_max_tokens
        while True:
            self.steps += 1
            try:
                body = await self.client.chat(
                    self.messages,
                    tools=tools,
                    tool_choice="auto",
                    reasoning_effort=self.config.reasoning_effort,
                    max_tokens=max_tokens,
                    **extra,
                )
                choice = body["choices"][0]
                message = choice["message"]
            except Exception as exc:
                logger.warning("agent step %d: model request failed (%s)", self.steps,
                               type(exc).__name__)  # fmt: skip
                raise _Stop("error", "model_request_failed", describe_error(exc)) from None
            self._add_usage(body)

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
                aws.append(
                    self._finish(
                        call["id"], name, args, error_result(ErrorCode.DUPLICATE_CALL), 0.0
                    )
                )
            else:
                if canonical is not None:
                    self.seen.add(key)
                aws.append(self._execute(call["id"], name, args))
        return aws

    async def _finish(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any] | None,
        result: dict[str, Any] | Any,
        ms: float,
    ) -> tuple[dict[str, Any], ToolCallRecord]:
        if not isinstance(result, dict):
            result = result.model_dump(mode="json")
        error = result.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        record = ToolCallRecord(
            step=self.steps,
            tool_call_id=call_id,
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
        self, call_id: str, name: str, args: dict[str, Any] | None
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
        return await self._finish(call_id, name, args, result, (time.perf_counter() - start) * 1000)

    async def verify(self, final_content: str) -> None:
        """Get a structured answer that resolves and passes the guard, in at most two
        attempts. Sets ``verified`` or ``refusal_reason``.

        In single-phase mode the loop's final message is the first attempt; otherwise each
        attempt is one request without tools, with the strict ``response_format``.
        """
        problems: list[str] = []
        for attempt in range(1, MAX_STRUCTURED_ATTEMPTS + 1):
            if attempt == 1 and self.single_phase:
                reply: tuple[Any, Any] | None = (final_content, "stop")
            else:
                self.start_phase(f"structured_{attempt}")
                try:
                    reply = await self._structured_request(attempt, problems)
                finally:
                    self.end_phase()
            if reply is None:
                self.refusal_reason = "structured_request_failed"
                return
            self.guard_attempts += 1
            problems, failure = self._evaluate(*reply)
            if not problems:
                self.verified = True
                self.refusal_reason = None
                return
            self.rejected_reasons.append(problems)
            self.refusal_reason = failure
            logger.info("agent structured attempt %d rejected: %s", attempt, failure)

    async def _structured_request(
        self, attempt: int, problems: Sequence[str]
    ) -> tuple[Any, Any] | None:
        """One structured request; ``(content, finish_reason)``, or None if it failed."""
        if attempt == 1:
            prompt = structured_instruction(self.records)
        elif self.single_phase:
            # The call list was never shown in single-phase mode.
            prompt = structured_instruction(self.records) + "\n\n" + correction_message(problems)
        else:
            prompt = correction_message(problems)
        self.messages.append({"role": "user", "content": prompt})
        try:
            body = await self.client.chat(
                self.messages,
                response_format=response_format(),
                reasoning_effort=self.config.reasoning_effort,
                max_tokens=self.config.structured_max_tokens,
            )
            choice = body["choices"][0]
            content = choice["message"].get("content")
        except Exception as exc:
            logger.warning("agent structured attempt %d: model request failed (%s)", attempt,
                           type(exc).__name__)  # fmt: skip
            self.warnings.append("structured_failed")
            self.warnings_detail.append({"warning": "structured_failed", **describe_error(exc)})
            return None
        self._add_usage(body)
        self.messages.append({"role": "assistant", "content": content})
        return content, choice.get("finish_reason")

    def _evaluate(self, content: Any, finish_reason: Any) -> tuple[list[str], str | None]:
        """``(problems, refusal_reason)`` for one reply; no problems means verified."""
        if finish_reason == "length":
            return [LENGTH_PROBLEM], "structured_length"
        if not isinstance(content, str) or not content.strip():
            return ["the reply was empty; reply with the JSON object"], "structured_invalid"
        try:
            answer = StructuredAnswer.model_validate_json(content)
        except ValidationError as exc:
            return _validation_problems(exc), "structured_invalid"
        resolved = resolve_answer(answer, self.records, self.messages)
        self.resolution_errors = list(resolved.errors)
        guard = check_answer(answer, resolved, output_guard=self.output_guard)
        if not guard.ok:
            return guard.problems(), "guard_failed"
        self.structured, self.resolved = answer, resolved
        return [], None


async def run_agent(
    question: str,
    *,
    provider: DataProvider,
    ctx: CallerContext,
    client: ChatClient,
    config: AgentConfig | None = None,
    structured: bool = False,
    structured_mode: str = "two_phase",
    output_guard: OutputGuard | None = None,
) -> AgentResult:
    """Run the tool loop; with ``structured=True`` return only a verified answer.

    Verification runs only after a final answer, within what is left of
    ``total_timeout_s``, and does not count as a step; its usage and latency count toward
    the totals and appear per phase in ``phases``. ``stop_reason`` stays ``final`` either
    way. A verified answer sets ``verified=True`` and ``answer`` to the rendered
    structured text. Otherwise ``verified=False``, ``refusal_reason`` says why and
    ``answer`` is ``REFUSAL_ANSWER``.

    ``structured_mode="single_phase"`` is an experiment: the schema is sent with every
    request of the tool loop and the final message is parsed as the structured answer.
    ``output_guard`` replaces the default advice check (see ``guard.check_answer``).
    """
    if structured_mode not in STRUCTURED_MODES:
        raise ValueError(f"structured_mode must be one of {STRUCTURED_MODES}")
    config = config or AgentConfig()
    run = _Run(
        question, provider, ctx, client, config,
        single_phase=structured and structured_mode == "single_phase",
        output_guard=output_guard,
    )  # fmt: skip
    start = time.perf_counter()
    answer: str | None = None
    run.start_phase("tool")
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
    finally:
        run.end_phase()
    if stop_reason not in ("final", "truncated"):
        answer = FALLBACK_ANSWERS[stop_reason]

    if structured and stop_reason == "final":
        remaining = config.total_timeout_s - (time.perf_counter() - start)
        try:
            async with asyncio.timeout(max(remaining, 0)):
                await run.verify(answer or "")
        except TimeoutError:
            run.verified, run.refusal_reason = False, "timeout"
        # Never the unverified phase A text: only the resolved answer or the refusal.
        if run.verified and run.resolved is not None:
            answer = run.resolved.text
        else:
            run.verified = False
            run.structured = run.resolved = None
            answer = REFUSAL_ANSWER

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
        phases=run.phases,
    )
    if structured:
        result.verified = run.verified
        result.refusal_reason = run.refusal_reason
        result.structured = run.structured
        result.resolved = run.resolved
        result.resolution_errors = run.resolution_errors
        result.guard_attempts = run.guard_attempts
        result.guard_rejections = len(run.rejected_reasons)
        result.rejected_reasons = run.rejected_reasons
        logger.info(
            "agent structured verified=%s refusal_reason=%s guard_attempts=%d "
            "guard_rejections=%d",
            result.verified, result.refusal_reason, result.guard_attempts,
            result.guard_rejections,
        )  # fmt: skip
    logger.info(
        "agent run stop_reason=%s steps=%d tool_calls=%d prompt_tokens=%d "
        "completion_tokens=%d reasoning_tokens=%d total_tokens=%d latency_ms=%.1f",
        result.stop_reason, result.steps, len(result.tool_calls), result.usage.prompt_tokens,
        result.usage.completion_tokens, result.usage.reasoning_tokens,
        result.usage.total_tokens, result.latency_ms,
    )  # fmt: skip
    return result
