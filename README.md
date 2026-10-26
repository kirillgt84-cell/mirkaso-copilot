# Mirkaso Copilot

An agentic market assistant built on **Meta Muse Spark** with function calling over
Mirkaso's public market data: scanner signals, sector indices, the macro risk regime,
per-coin derivatives and liquidations.

Built during the **Meta Global AI Developer Hackathon, Oct 26 – Nov 4, 2026**. Everything
in this repository was written during the hackathon.

## How it works

```
user ──► host app ──► Muse Spark (Meta Model API, /chat/completions)
            │                 │ tool_calls
            │                 ▼
            │        tools.registry.execute_tool(name, args, provider, ctx)
            │                 │ validate args (pydantic) → gate by tier → downsample
            │                 ▼
            │          DataProvider (adapter) ──► Mirkaso public market data
            ▼
   CallerContext(tier) resolved by the host, never by the model
```

- **Tool definitions** are generated from pydantic parameter models
  (`tool_definitions()`, OpenAI function-calling format). Every schema has
  `additionalProperties: false` and hard caps on limits and windows, so the model
  cannot request unbounded data.
- **One envelope for every result**: `as_of` (UTC ISO-8601), `source_id`, `gated`,
  `gated_fields`, `data`. Liquidations also carry `newest_event_at` so the model can
  judge freshness. Liquidations are aggregates only (summary, top coins, time series),
  never individual events.
- **Errors are structured** (`{"error": {"code", "message"}}`) with fixed messages.
  Exception text, SQL and tracebacks never reach the model; they are logged on the
  host side only.
- **Gating by plan tier** happens in pure functions before the data reaches the model.
  The tier comes from `CallerContext`, which the host app builds from its own session.
  The model cannot raise its own tier through tool arguments.
  See [Tiers](#tiers).
- **Downsampling**: history series are rebucketed to at most 48 points to keep the
  context small.
- `meta_client.MetaClient` is a minimal async client for the Meta Model API
  (`tools`, `tool_choice`, `response_format`, `reasoning_effort` low/medium/high) that
  logs token usage, including reasoning tokens.

The tool contract is documented in [docs/tools.md](docs/tools.md).

## Agent loop

`mirkaso_copilot.agent.run_agent(question, provider=..., ctx=..., client=...)` runs the
tool-calling loop: it sends the question with every tool definition
(`tool_choice="auto"`), executes all tool calls of a step concurrently, feeds each
result back as a `tool` message, and repeats until the model answers without tool calls.

Limits (`AgentConfig`, all overridable):

| Setting | Default | Meaning |
|---|---|---|
| `max_steps` | 6 | Model requests per run |
| `max_tool_calls_total` | 12 | Tool calls per run |
| `max_total_tokens` | 40000 | Tokens summed over all requests |
| `request_max_tokens` | 3000 | `max_tokens` per request (reasoning tokens count toward it) |
| `reasoning_effort` | `low` | `low`, `medium` or `high` |
| `per_tool_timeout_s` | 15 | Time limit for one tool call |
| `total_timeout_s` | 90 | Time limit for the whole run |

Tool failures never stop the run. A tool error, a tool timeout, arguments that are not a
JSON object, an unknown tool, or a repeat of the same tool with the same validated arguments
(defaults applied, so `{}` and `{"history_days": 7}` match) all
come back to the model as a structured error (`{"error": {"code", "message"}}`), so it
can recover or explain.

`AgentResult` carries `answer`, `stop_reason`, `steps`, a record per tool call (name,
args, error code, latency, `source_id`, `as_of`), summed token usage including reasoning
tokens, total latency, `warnings` and the full message transcript.

| `stop_reason` | When | `answer` |
|---|---|---|
| `final` | The model answered without tool calls | The model's answer |
| `max_steps` | The step limit was reached | Fixed fallback text |
| `max_tool_calls` | The next step would exceed the tool-call limit | Fixed fallback text |
| `max_tokens` | The token budget was exceeded | Fixed fallback text |
| `timeout` | `total_timeout_s` elapsed | Fixed fallback text |
| `truncated` | The model hit `finish_reason: "length"` | `None` |
| `error` | The model request failed or returned an empty answer | Fixed fallback text |

Fallback texts contain no numbers. Structured answers and numeric grounding (checking
every number in an answer against the tool results in the transcript) are separate
modules that are coming next.

## Tiers

Gating uses four public tiers, lowest first:

```
guest < starter < trader < pro
```

| Tier | Scanner | Sector `sector_score_delta_24h` |
|---|---|---|
| guest | masked | hidden |
| starter | masked | visible |
| trader | full fidelity | visible |
| pro | full fidelity | visible |

The host application maps its own plan names to these four tiers when it builds
`CallerContext`. The mapping is not part of this repository. A tier that is not one of
the four is treated as `guest`, so a mapping mistake fails closed.

## Boundary: what is in this repo and what is not

| In this repository (new, open, MIT) | In the private Mirkaso product (not here) |
|---|---|
| Tool definitions, parameter schemas and caps | Data ingestion pipelines |
| Result envelope and error model | Database schema, SQL and storage |
| Tier gating and downsampling (pure functions) | Scanner and scoring logic |
| `DataProvider` protocol and a fake for tests | The production `DataProvider` implementation |
| Meta Model API client | Accounts, billing and tier resolution |

The private product plugs in by implementing `mirkaso_copilot.providers.DataProvider`.
This repository contains no credentials, no internal table names and no proprietary
signal logic.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Tests use a fake provider and make no network calls.

To call the Meta Model API, copy `.env.example` to `.env` (git-ignored) and set
`META_API_KEY`, `META_MODEL` (e.g. `muse-spark-1.3`) and optionally `META_API_BASE`.
The code has no default model.

> **Never use a model ID ending in `-contributor`.** Those variants may be trained on
> the data you send. Use only the standard model.

The client reads these
from the process environment only.

## License

[MIT](LICENSE)
