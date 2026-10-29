# Tool specification

Four tools are exposed to the model: `scanner_get_signals`, `sectors_get_indices`,
`macro_get_regime` and `coin_get_derivatives`. Parameter caps are enforced by the JSON
schema (`additionalProperties: false`) and again by server-side validation.

Tool descriptions state no data numbers (thresholds, scales, band ranges): only
timeframes such as `1h` or `24h` and caps on what a call may return. Every number an
answer may cite is in a tool result, where a structured answer can point at it with a
JSON Pointer (see [Structured answers](../README.md#structured-answers)). A test checks
the descriptions for this.

## Result envelope

Every successful call returns:

| Field | Type | Meaning |
|---|---|---|
| `as_of` | string | UTC ISO-8601 time the data refers to (`2026-10-26T12:00:00Z`) |
| `source_id` | string | Stable source identifier, e.g. `mirkaso.scanner` |
| `gated` | bool | `true` if some fields are hidden on the caller's plan |
| `gated_fields` | string[] | Names of hidden fields; empty when `gated` is false |
| `data` | object | Tool-specific payload |

Failures return `{"error": {"code": ..., "message": ...}}` with one of these codes:
`unknown_tool`, `invalid_arguments`, `not_found`, `upstream_unavailable`, `internal_error`.
The agent loop adds `timeout` (per-tool time limit) and `duplicate_call` (same tool with
the same validated arguments, defaults applied, already called in this run).
Messages are fixed text. For `invalid_arguments` they list only the names of the
offending fields.

Plan tiers, lowest first: `guest` < `starter` < `trader` < `pro`. The host application
maps its own plan names to these tiers. Unknown tiers are treated as `guest`.

## scanner_get_signals

Current market-scanner signals for crypto perpetual futures.

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `min_score` | integer | 8 | 0–13 |
| `direction` | string | – | `LONG` \| `SHORT` |
| `confidence` | string | – | `high` \| `medium` \| `low` |
| `base_asset` | string | – | alphanumeric, 1–15 chars |
| `limit` | integer | 10 | 1–20 |

`data.signals[]`: `symbol`, `base_asset`, `direction`, `score`, `confidence`, `price`,
`price_change_1h_pct` (1-hour move, never 24h), `volume_ratio`, `oi_change_pct`,
`quote_volume_24h`, `detected_at`.

`data.score_max`: the highest possible signal score (13); scores start at 0. It is
present in every result, including an empty or gated one.

Gating:
- **trader and above**: full fidelity.
- **below trader**: `gated=true`. `symbol`/`base_asset` are masked as `***` except for
  BTC and ETH. `price` becomes an order-of-magnitude range label (`<0.01`, `0.01-0.1`,
  …, `10k-100k`, `>=100k`). `volume_ratio`, `oi_change_pct`, `price_change_1h_pct` and
  `quote_volume_24h` are `null`. `gated_fields` lists all of these.
- **below trader with a `base_asset` filter other than BTC/ETH**: returns no signals,
  with `gated=true` and `base_asset_filter` in `gated_fields`. This stops the filter
  from revealing whether a masked coin has signals.

## sectors_get_indices

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `asset_class` | string | – (all) | `crypto` \| `tradfi` |

`data.sectors[]`: `slug`, `asset_class`, `sector_score`, `sector_score_delta_24h`,
`price_change_24h_pct`, `oi_change_24h_pct`, `breadth_ratio`, `relative_strength_btc`.
There is no sparkline.

Gating: for **guest**, `sector_score_delta_24h` is `null` and `gated=true`. It is
visible to **starter and above**.

## macro_get_regime

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `include_history` | bool | false | |
| `history_days` | integer | 7 | 1–30 |

`data`: `score` (0–100), `regime`, `panic_override`, `bands[]`, and `history[]` when
requested.

`bands[]` lists every regime band in ascending order as `{label, min, max}`, so an
answer can cite a band's range from the data:

| `label` | `min` | `max` |
|---|---|---|
| Panic | 0 | 20 |
| Risk OFF | 20 | 35 |
| Cautious | 35 | 50 |
| Neutral | 50 | 65 |
| Risk ON | 65 | 80 |
| Strong Risk ON | 80 | 100 |

`min` is inclusive and `max` exclusive, except for the top band, which includes 100.
`panic_override=true` forces `Panic` regardless of score.

History is rebucketed to at most 48 time-ordered points. Each point has `ts` (latest
in the bucket), `score` (the bucket average, 2 decimals), `regime` (the latest row's
regime) and `panic_override` (true if any row in the bucket had it, which forces
`regime` to `Panic`).

## coin_get_derivatives

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `symbol` | string | **required** | alphanumeric, 1–20 chars, upper-cased |
| `include_history` | bool | false | |
| `history_points` | integer | 24 | 1–48 |

`data`: `symbol`, `price`, `price_change_1h_pct`, `price_change_4h_pct`,
`price_change_24h_pct`, `open_interest_usd`, `oi_change_1h_pct`, `oi_change_24h_pct`,
`funding_rate`, `trend`, `market_score`, and `history[]` (`ts`, `open_interest_usd`,
`price`) when requested. History is rebucketed to at most `history_points` points,
keeping the latest open interest and price in each bucket.
`open_interest_usd` is open interest in US dollars (base-asset units × price), not
a number of coins; point a value at it with kind `usd`. An unknown symbol returns
`not_found`.

## Not exposed: liquidations

Liquidations are intentionally excluded until a reliable data source exists. The data
source behind them is currently unavailable, and a tool over it would report empty totals
as if nothing had been liquidated. A call to the former `market_get_liquidations` name
returns `unknown_tool`.
