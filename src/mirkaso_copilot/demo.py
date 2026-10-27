"""Live demo: one question through the Meta Model API, the agent loop and fixture data.

Usage::

    mirkaso-copilot-demo "Which sectors are strongest right now?" [--tier trader]
        [--effort low] [--json] [--show-tools] [--fixed-time] [--verbose] [--structured]
        [--single-phase]

Reads ``META_API_KEY`` and ``META_MODEL`` (both required) and ``META_API_BASE``
(optional) from the environment. The key is never printed. If ``META_PRICE_IN_PER_M``
and ``META_PRICE_OUT_PER_M`` (USD per million prompt / completion tokens) are set, an
estimated cost is printed too. All market data is synthetic demo data, shifted to the
current time unless ``--fixed-time`` is given. ``--structured`` returns only a verified
answer: a structured answer whose values are resolved against the tool results and that
passes the grounding guard, or a refusal. ``--single-phase`` (implies ``--structured``)
is an experiment that sends the schema with every tool-loop request.

Exit status: 0 when the model gave a final answer (with ``--structured``: a verified
one), 1 otherwise, 2 for configuration errors.
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, TextIO

from mirkaso_copilot.agent import AgentConfig, AgentResult, ChatClient, run_agent
from mirkaso_copilot.fixture_provider import BASE_TIME, SYNTHETIC_NOTICE, FixtureProvider
from mirkaso_copilot.meta_client import MetaClient, MetaConfigError
from mirkaso_copilot.providers import CallerContext, DataProvider, Tier

EFFORTS = ("low", "medium", "high")
REQUIRED_ENV = ("META_API_KEY", "META_MODEL")
PRICE_IN_ENV = "META_PRICE_IN_PER_M"
PRICE_OUT_ENV = "META_PRICE_OUT_PER_M"
# Contributor model variants may be trained on the data sent to them.
FORBIDDEN_MODEL_SUFFIX = "-contributor"


class DemoConfigError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mirkaso-copilot-demo",
        description="Ask Mirkaso Copilot a question over synthetic demo market data.",
    )
    parser.add_argument("question", help="The question to ask, in quotes.")
    parser.add_argument(
        "--tier", choices=[t.value for t in Tier], default=Tier.TRADER.value,
        help="Plan tier the data is gated for (default: trader).",
    )  # fmt: skip
    parser.add_argument(
        "--effort", choices=EFFORTS, default="low",
        help="Model reasoning effort (default: low).",
    )  # fmt: skip
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full AgentResult as JSON (tool call records, usage, transcript).",
    )  # fmt: skip
    parser.add_argument(
        "--show-tools", action="store_true",
        help="Print each tool call with its arguments, outcome and latency.",
    )  # fmt: skip
    parser.add_argument(
        "--fixed-time", action="store_true",
        help=f"Keep the stored fixture timestamps (base time {BASE_TIME}) instead of "
        "shifting them to the current time.",
    )  # fmt: skip
    parser.add_argument(
        "--verbose", action="store_true",
        help="On a failed model request, print the HTTP status and the API's error message.",
    )  # fmt: skip
    parser.add_argument(
        "--structured", action="store_true",
        help="Return only a verified answer: numbers resolved by the server from the tool "
        "results and checked by the grounding guard, or a refusal. Prints the values table, "
        "guard metrics and per-phase usage.",
    )  # fmt: skip
    parser.add_argument(
        "--single-phase", action="store_true",
        help="Experiment (implies --structured): send the answer schema with every tool-loop "
        "request instead of making a separate structured request.",
    )  # fmt: skip
    return parser


def read_prices(env: Mapping[str, str]) -> tuple[float, float] | None:
    """``(in, out)`` USD per million tokens, or None if either is unset."""
    raw_in, raw_out = env.get(PRICE_IN_ENV), env.get(PRICE_OUT_ENV)
    if not raw_in or not raw_out:
        return None
    try:
        prices = float(raw_in), float(raw_out)
    except ValueError:
        raise DemoConfigError(f"{PRICE_IN_ENV} and {PRICE_OUT_ENV} must be numbers") from None
    if min(prices) < 0:
        raise DemoConfigError(f"{PRICE_IN_ENV} and {PRICE_OUT_ENV} must not be negative")
    return prices


def estimate_cost(result: AgentResult, prices: tuple[float, float]) -> float:
    # Reasoning tokens are already part of completion_tokens.
    price_in, price_out = prices
    return (
        result.usage.prompt_tokens * price_in + result.usage.completion_tokens * price_out
    ) / 1_000_000


def check_env(env: Mapping[str, str]) -> None:
    """Fail before any work if the Meta client could not start or the model is unsafe."""
    for name in REQUIRED_ENV:
        if not env.get(name):
            raise DemoConfigError(f"{name} is not set")
    if env["META_MODEL"].endswith(FORBIDDEN_MODEL_SUFFIX):
        raise DemoConfigError(
            f"META_MODEL ends with '{FORBIDDEN_MODEL_SUFFIX}'; use the standard model instead"
        )


def format_tool_calls(result: AgentResult) -> list[str]:
    lines = []
    for r in result.tool_calls:
        args = json.dumps(r.args, sort_keys=True) if r.args is not None else "<invalid JSON>"
        outcome = "ok" if r.ok else f"error={r.error_code}"
        lines.append(f"  [step {r.step}] {r.name} {args} -> {outcome} ({r.latency_ms:.0f} ms)")
    return lines


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  " + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
            for row in rows]  # fmt: skip


def format_structured(result: AgentResult) -> list[str]:
    """The verified summary, claims and caveats, then the table of resolved values."""
    resolved = result.resolved
    assert resolved is not None
    lines = [resolved.summary]
    lines += [f"- {claim}" for claim in resolved.claims]
    lines += [f"Caveat: {caveat}" for caveat in resolved.caveats]
    lines += ["", "values:"]
    if resolved.values:
        rows = [["id", "value", "call", "tool", "path", "as_of"]]
        rows += [[v.id, v.formatted, v.call_ref, v.tool, v.path, v.as_of or "-"]
                 for v in resolved.values]  # fmt: skip
        lines += _table(rows)
    else:
        lines.append("  (none)")
    if resolved.unused_values:
        lines.append("unused values: " + ", ".join(resolved.unused_values))
    return [*lines, ""]


def format_guard(result: AgentResult) -> list[str]:
    """Verification outcome, guard metrics and the reasons for each rejected attempt."""
    verified = "yes" if result.verified else f"no ({result.refusal_reason or result.stop_reason})"
    lines = [
        f"verified: {verified}",
        f"guard: attempts={result.guard_attempts} rejections={result.guard_rejections}",
    ]
    for i, reasons in enumerate(result.rejected_reasons, 1):
        lines.append(f"  rejected attempt {i}:")
        lines += [f"    - {reason}" for reason in reasons]
    return lines


def _usage_line(label: str, u: Any, latency_ms: float) -> str:
    return (
        f"{label}: prompt={u.prompt_tokens} completion={u.completion_tokens} "
        f"reasoning={u.reasoning_tokens} total={u.total_tokens}  {latency_ms / 1000:.2f} s"
    )


def format_report(
    result: AgentResult, prices: tuple[float, float] | None, *, structured: bool = False
) -> list[str]:
    """The answer and run statistics. ``structured`` shows the verified answer with its
    values, or the refusal, plus guard metrics and usage per phase."""
    u = result.usage
    if structured and result.verified:
        lines = format_structured(result)
    else:
        answer = result.answer
        lines = [answer if answer is not None else "(no answer: the response was cut off)", ""]
    if structured:
        lines += format_guard(result)
    lines += [
        f"stop_reason: {result.stop_reason}",
        f"steps: {result.steps}  tool calls: {len(result.tool_calls)}",
    ]
    if structured:
        lines.append("usage by phase:")
        lines += [_usage_line(f"  {p.name}", p.usage, p.latency_ms) for p in result.phases]
        lines.append(_usage_line("  total", u, result.latency_ms))
    else:
        lines += [
            f"tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} "
            f"reasoning={u.reasoning_tokens} total={u.total_tokens}",
            f"latency: {result.latency_ms / 1000:.2f} s",
        ]
    if prices is not None:
        lines.append(f"estimated cost: ${estimate_cost(result, prices):.6f}")
    if result.warnings:
        lines.append("warnings: " + ", ".join(result.warnings))
    return lines


def format_details(result: AgentResult, secret: str | None) -> list[str]:
    """One line per detailed warning. ``secret`` is masked should it ever appear."""
    lines = []
    for d in result.warnings_detail:
        if d.get("status_code") is None:
            line = f"model error: {d['error_type']}"
        else:
            line = f"model error: HTTP {d['status_code']} ({d['error_type']})"
            if d.get("message"):
                line += f": {d['message']}"
        if secret:
            line = line.replace(secret, "***")
        lines.append(line)
    return lines


def result_json(
    result: AgentResult, prices: tuple[float, float] | None, *, verbose: bool
) -> dict[str, Any]:
    out = asdict(result)
    out["structured"] = result.structured.model_dump() if result.structured else None
    if not verbose:
        del out["warnings_detail"]
    if prices is not None:
        out["estimated_cost_usd"] = estimate_cost(result, prices)
    return out


async def ask(args: argparse.Namespace, client: ChatClient, provider: DataProvider) -> AgentResult:
    return await run_agent(
        args.question,
        provider=provider,
        ctx=CallerContext(tier=args.tier),
        client=client,
        config=AgentConfig(reasoning_effort=args.effort),
        structured=args.structured or args.single_phase,
        structured_mode="single_phase" if args.single_phase else "two_phase",
    )


async def _run_with_meta(args: argparse.Namespace, provider: DataProvider) -> AgentResult:
    async with MetaClient() as client:
        return await ask(args, client, provider)


def main(
    argv: list[str] | None = None,
    *,
    client: ChatClient | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the demo. ``client`` replaces the Meta client (tests use a scripted fake)."""
    args = build_parser().parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    provider = FixtureProvider(fixed_time=args.fixed_time)

    try:
        prices = read_prices(os.environ)
        if client is None:
            check_env(os.environ)
        timing = (
            f"fixed at {BASE_TIME}"
            if args.fixed_time
            else f"shifted to the current time, as_of {provider.base_time}"
        )
        print(f"note: {SYNTHETIC_NOTICE} ({timing}; tier {args.tier})", file=stderr)
        if client is None:
            result = asyncio.run(_run_with_meta(args, provider))
        else:
            result = asyncio.run(ask(args, client, provider))
    except (DemoConfigError, MetaConfigError) as exc:
        print(f"error: {exc}", file=stderr)
        return 2

    secret = os.environ.get("META_API_KEY")
    if args.verbose:
        for line in format_details(result, secret):
            print(line, file=stderr)
    if args.json:
        out = json.dumps(result_json(result, prices, verbose=args.verbose), indent=2,
                         ensure_ascii=False)  # fmt: skip
        print(out.replace(secret, "***") if secret else out, file=stdout)
    else:
        if args.show_tools:
            print("tool calls:", file=stdout)
            for line in format_tool_calls(result) or ["  (none)"]:
                print(line, file=stdout)
            print("", file=stdout)
        structured = args.structured or args.single_phase
        print("\n".join(format_report(result, prices, structured=structured)), file=stdout)
    if (args.structured or args.single_phase) and not result.verified:
        return 1
    return 0 if result.stop_reason == "final" else 1


if __name__ == "__main__":
    sys.exit(main())
