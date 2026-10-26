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
