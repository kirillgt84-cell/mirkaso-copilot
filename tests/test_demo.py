"""Demo CLI: argument parsing, config errors and an end-to-end run. No network."""

import io
import json
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest

from mirkaso_copilot import demo
from mirkaso_copilot.demo import build_parser, main
from mirkaso_copilot.envelope import utc_now
from mirkaso_copilot.meta_client import REASONING_EFFORTS, MetaAPIError
from tests.conftest import ScriptedClient, call, reply

FAKE_KEY = "demo-test-key-not-real-123"
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "total_tokens": 1200,
    "completion_tokens_details": {"reasoning_tokens": 50},
}


@pytest.fixture(autouse=True)
def meta_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", FAKE_KEY)
    monkeypatch.setenv("META_MODEL", "test-model")
    for name in ("META_API_BASE", demo.PRICE_IN_ENV, demo.PRICE_OUT_ENV):
        monkeypatch.delenv(name, raising=False)


def scripted() -> ScriptedClient:
    return ScriptedClient(
        reply(
            tool_calls=[
                call("sectors_get_indices", {"asset_class": "crypto"}, "c1"),
                call("macro_get_regime", {}, "c2"),
                call("coin_get_derivatives", {"symbol": "NOPE"}, "c3"),
            ],
            usage=USAGE,
        ),
        reply("AI leads the crypto sectors; the macro regime is Neutral (54.3).", usage=USAGE),
    )


def run_main(argv: list[str], client: ScriptedClient | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, client=client, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["What is up?"])
    assert args.question == "What is up?"
    assert (args.tier, args.effort, args.json, args.show_tools) == ("trader", "low", False, False)


def test_effort_choices_match_the_client() -> None:
    assert set(demo.EFFORTS) == REASONING_EFFORTS


def test_parser_options() -> None:
    args = build_parser().parse_args(
        ["q", "--tier", "guest", "--effort", "high", "--json", "--show-tools"]
    )
    assert (args.tier, args.effort, args.json, args.show_tools) == ("guest", "high", True, True)


@pytest.mark.parametrize(
    "argv", [[], ["q", "--tier", "platinum"], ["q", "--effort", "max"], ["q", "--bogus"]]
)
def test_parser_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("missing", ["META_API_KEY", "META_MODEL"])
def test_missing_env_is_a_config_error(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(missing)
    code, out, err = run_main(["q"])
    assert code == 2
    assert f"{missing} is not set" in err
    assert FAKE_KEY not in out + err


def test_contributor_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_MODEL", "muse-spark-contributor")
    code, _, err = run_main(["q"])
    assert code == 2
    assert "-contributor" in err


def test_bad_prices_are_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(demo.PRICE_IN_ENV, "cheap")
    monkeypatch.setenv(demo.PRICE_OUT_ENV, "1")
    code, _, err = run_main(["q"], scripted())
    assert code == 2
    assert demo.PRICE_IN_ENV in err


def test_end_to_end_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(demo.PRICE_IN_ENV, "0.5")
    monkeypatch.setenv(demo.PRICE_OUT_ENV, "2")
    client = scripted()
    code, out, err = run_main(["Strongest sectors?", "--show-tools", "--fixed-time"], client)

    assert code == 0
    assert "AI leads the crypto sectors" in out
    assert "stop_reason: final" in out
    assert "steps: 2  tool calls: 3" in out
    assert "tokens: prompt=2000 completion=400 reasoning=100 total=2400" in out
    assert "latency:" in out
    # 2000 * 0.5 / 1e6 + 400 * 2 / 1e6
    assert "estimated cost: $0.001800" in out
    assert 'sectors_get_indices {"asset_class": "crypto"} -> ok' in out
    assert 'coin_get_derivatives {"symbol": "NOPE"} -> error=not_found' in out
    assert "synthetic demo data" in err
    assert "fixed at 2026-10-26T12:00:00Z" in err
    assert FAKE_KEY not in out + err

    # The tools really ran against the fixtures and their results reached the model.
    tool_results = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    sectors = json.loads(tool_results[0]["content"])
    assert sectors["as_of"] == "2026-10-26T12:00:00Z"
    assert {s["asset_class"] for s in sectors["data"]["sectors"]} == {"crypto"}
    assert json.loads(tool_results[1]["content"])["data"]["score"] == 54.3
    assert client.requests[0]["reasoning_effort"] == "low"


def test_end_to_end_json_output() -> None:
    client = scripted()
    code, out, _ = run_main(["q", "--json", "--effort", "medium", "--tier", "guest"], client)

    assert code == 0
    payload = json.loads(out)
    assert payload["stop_reason"] == "final"
    assert payload["steps"] == 2
    assert [c["name"] for c in payload["tool_calls"]] == [
        "sectors_get_indices",
        "macro_get_regime",
        "coin_get_derivatives",
    ]
    assert payload["tool_calls"][2]["error_code"] == "not_found"
    assert payload["usage"]["reasoning_tokens"] == 100
    assert "estimated_cost_usd" not in payload
    assert "warnings_detail" not in payload
    assert client.requests[0]["reasoning_effort"] == "medium"
    # Guest tier: the sector delta is gated.
    sectors = json.loads(payload["messages"][3]["content"])
    assert sectors["gated"] is True
    assert FAKE_KEY not in out


def test_non_final_stop_exits_1() -> None:
    client = ScriptedClient(reply(None, usage=USAGE))  # empty answer -> stop_reason "error"
    code, out, _ = run_main(["q"], client)
    assert code == 1
    assert "stop_reason: error" in out
    assert "warnings: empty_answer" in out


def test_default_run_shifts_data_to_now() -> None:
    client = scripted()
    code, _, err = run_main(["q"], client)
    assert code == 0
    assert "shifted to the current time" in err
    sectors = json.loads(client.requests[1]["messages"][3]["content"])
    as_of = datetime.fromisoformat(sectors["as_of"])
    assert utc_now() - timedelta(minutes=1) <= as_of <= utc_now()
    assert sectors["as_of"].endswith("Z")


# --- --verbose ---------------------------------------------------------------------

UNAUTHORIZED_BODY = json.dumps(
    {"error": {"type": "invalid_request_error", "message": "Invalid API key. " + "x" * 300}}
)


class FailingClient:
    """Fails the first request the way MetaClient does."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raise self.exc


def unauthorized() -> MetaAPIError:
    return MetaAPIError("Meta API returned HTTP 401", status_code=401, body=UNAUTHORIZED_BODY)


def connection_error() -> MetaAPIError:
    try:
        try:
            raise httpx.ConnectError(f"cannot reach api.example with {FAKE_KEY}")
        except httpx.ConnectError as cause:
            raise MetaAPIError("request failed: ConnectError") from cause
    except MetaAPIError as exc:
        return exc


def test_verbose_prints_http_status_and_message() -> None:
    code, out, err = run_main(["q", "--verbose"], FailingClient(unauthorized()))
    assert code == 1
    assert "stop_reason: error" in out
    line = next(ln for ln in err.splitlines() if ln.startswith("model error:"))
    assert line.startswith("model error: HTTP 401 (MetaAPIError): Invalid API key. xxx")
    message = line.split(": ", 2)[2]
    assert len(message) == 200
    assert FAKE_KEY not in out + err
    assert "Authorization" not in out + err


def test_verbose_connection_error_prints_type_only() -> None:
    code, out, err = run_main(["q", "--verbose"], FailingClient(connection_error()))
    assert code == 1
    assert "model error: ConnectError" in err.splitlines()
    assert "api.example" not in out + err
    assert FAKE_KEY not in out + err


def test_verbose_json_includes_details() -> None:
    _, out, _ = run_main(["q", "--verbose", "--json"], FailingClient(unauthorized()))
    detail = json.loads(out)["warnings_detail"]
    assert detail == [
        {
            "warning": "model_request_failed",
            "error_type": "MetaAPIError",
            "status_code": 401,
            "message": json.loads(UNAUTHORIZED_BODY)["error"]["message"][:200],
        }
    ]


def test_verbose_masks_the_key_if_the_api_echoes_it() -> None:
    body = json.dumps({"error": {"message": f"key {FAKE_KEY} is revoked"}})
    exc = MetaAPIError("Meta API returned HTTP 401", status_code=401, body=body)
    _, out, err = run_main(["q", "--verbose", "--json"], FailingClient(exc))
    assert "key *** is revoked" in err
    assert FAKE_KEY not in out + err


@pytest.mark.parametrize("make_exc", [unauthorized, connection_error])
def test_without_verbose_no_details_are_printed(make_exc: Any) -> None:
    code, out, err = run_main(["q"], FailingClient(make_exc()))
    assert code == 1
    assert "warnings: model_request_failed" in out
    assert "model error" not in out + err
    assert "Invalid API key" not in out + err
    assert "ConnectError" not in out + err


# --- --structured ------------------------------------------------------------------


def structured_reply(values: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    body = {"summary": summary, "claims": [{"text": "Macro score {v2}."}], "values": values,
            "caveats": []}  # fmt: skip
    return reply(json.dumps(body), usage=USAGE)


def value(vid: str, call_id: str, path: str, kind: str) -> dict[str, Any]:
    return {"id": vid, "tool_call_id": call_id, "path": path, "kind": kind, "decimals": None}


SECTOR_VALUES = [
    value("v1", "c1", "/data/sectors/0/slug", "text"),
    value("v2", "c2", "/data/score", "number"),
    value("v3", "c2", "/as_of", "timestamp"),
]


def scripted_structured(phase_b: dict[str, Any]) -> ScriptedClient:
    c = scripted()
    c.responses.append(phase_b)
    return c


def test_structured_text_output() -> None:
    c = scripted_structured(structured_reply(SECTOR_VALUES, "{v1} leads as of {v3}."))
    code, out, _ = run_main(["q", "--structured", "--fixed-time"], c)

    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "ai leads as of 2026-10-26 12:00 UTC."
    assert lines[1] == "- Macro score 54.3."
    assert "AI leads the crypto sectors" not in out  # the plain answer is replaced
    assert "values:" in lines
    row = next(ln for ln in lines if ln.lstrip().startswith("v2 "))
    assert row.split() == ["v2", "54.3", "macro_get_regime", "/data/score", "2026-10-26T12:00:00Z"]
    assert "resolution errors" not in out
    assert "steps: 2  tool calls: 3" in out
    assert "tokens: prompt=3000 completion=600 reasoning=150 total=3600" in out
    assert c.requests[2]["response_format"]["json_schema"]["strict"] is True


def test_structured_output_lists_resolution_errors() -> None:
    values = [*SECTOR_VALUES[:2], value("v3", "c3", "/data/price", "usd")]  # c3 is not_found
    c = scripted_structured(structured_reply(values, "{v1} leads; BTC {v3}; {v4}."))
    _, out, _ = run_main(["q", "--structured"], c)

    assert "ai leads; BTC {v3}; {v4}." in out
    assert "  v3: tool_call_failed - " in out
    assert "  v4: unknown_placeholder - " in out


def test_structured_failure_falls_back_to_the_plain_answer() -> None:
    c = scripted_structured(reply("not json", usage=USAGE))
    code, out, _ = run_main(["q", "--structured"], c)
    assert code == 0
    assert "(no structured answer: structured_failed; plain answer below)" in out
    assert "AI leads the crypto sectors" in out
    assert "warnings: structured_failed" in out


def test_structured_json_output() -> None:
    c = scripted_structured(structured_reply(SECTOR_VALUES, "{v1} leads as of {v3}."))
    _, out, _ = run_main(["q", "--structured", "--json", "--fixed-time"], c)

    payload = json.loads(out)
    assert payload["structured"]["values"][0]["path"] == "/data/sectors/0/slug"
    assert payload["resolved"]["summary"] == "ai leads as of 2026-10-26 12:00 UTC."
    assert payload["resolved"]["values"][1]["formatted"] == "54.3"
    assert payload["resolution_errors"] == []
    assert [t["tool_call_id"] for t in payload["tool_calls"]] == ["c1", "c2", "c3"]


def test_default_run_has_no_structured_phase() -> None:
    c = scripted()
    _, out, _ = run_main(["q", "--json"], c)
    assert len(c.requests) == 2
    payload = json.loads(out)
    assert (payload["structured"], payload["resolved"], payload["resolution_errors"]) == (
        None, None, [],
    )  # fmt: skip
