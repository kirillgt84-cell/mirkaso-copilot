# Tool specification

Five tools are exposed to the model. Parameter caps are enforced by the JSON schema
(`additionalProperties: false`) and again by server-side validation.

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

`data`: `score` (0–100), `regime`, `panic_override`, and `history[]` when requested.

| Score | Regime |
|---|---|
| < 20 | Panic |
| 20–35 | Risk OFF |
| 35–50 | Cautious |
| 50–65 | Neutral |
| 65–80 | Risk ON |
| 80–100 | Strong Risk ON |

Lower bounds are inclusive. `panic_override=true` forces `Panic` regardless of score.

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
`price_change_24h_pct`, `open_interest`, `oi_change_1h_pct`, `oi_change_24h_pct`,
`funding_rate`, `trend`, `market_score`, and `history[]` (`ts`, `open_interest`,
`price`) when requested. History is rebucketed to at most `history_points` points,
keeping the latest open interest and price in each bucket. An unknown symbol returns
`not_found`.

## market_get_liquidations

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `period` | string | `24h` | `1h` \| `4h` \| `24h` \| `7d` |
| `limit` | integer | 10 | 1–20 (number of `top_coins`) |

Aggregates only. Individual liquidation events are never returned.

The envelope has a top-level `newest_event_at` next to `as_of`: the UTC time of the
newest liquidation seen, or `null` if there is none.

`data`:

| Field | Shape |
|---|---|
| `period` | the requested period |
| `summary` | `{total_usd, long_usd, short_usd, long_pct, short_pct, count}` |
| `top_coins[]` | `{symbol, total_usd, long_usd, short_usd, count}`, at most `limit` entries |
| `time_series[]` | `{bucket, long_usd, short_usd}`, `bucket` is UTC ISO-8601, at most 168 points |

If the source has more than 168 time-series points, they are rebucketed evenly into
168 time-ordered buckets. Each bucket sums `long_usd` and `short_usd` and is labelled
with the start of its first source bucket. Only the fields listed above are returned.
