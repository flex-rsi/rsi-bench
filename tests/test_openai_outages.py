"""OpenAI outage handling and response metadata through the SDK's HTTP layer."""

from __future__ import annotations

from email.utils import formatdate

import pytest

openai = pytest.importorskip("openai")
try:  # OpenAI SDK 3 uses httpx2; earlier releases use httpx.
    import httpx2 as httpx
except ImportError:
    import httpx

from rsibench.models import openai_model
from rsibench.models.base import ModelError


class Clock:
    now = 100.0
    epoch = 1_700_000_000.0

    def __init__(self):
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.epoch + self.now - 100

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def completion(**extra):
    return {
        "object": "chat.completion", "created": 1, "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        **extra,
    }


def error(status, headers=None):
    return httpx.Response(status, headers=headers,
                          json={"error": {"message": "temporarily unavailable"}})


@pytest.fixture
def http_model(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(openai_model, "time", clock)
    monkeypatch.setenv("RSIB_TEST_OPENAI_KEY", "test-key")
    real_client = openai.OpenAI
    models = []

    def make(replies, outage_wait_s=10, **params):
        replies = iter(replies)
        requests = []

        def handle(request):
            requests.append(request)
            reply = next(replies)
            return reply(request) if callable(reply) else reply

        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: real_client(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle))))
        model_params = {
            "base_url": "https://openrouter.ai/api/v1", "api_key_env": "RSIB_TEST_OPENAI_KEY",
            "max_retries": 0, "outage_backoff_s": 4, **params,
        }
        if outage_wait_s is not None:  # None leaves the outage wait parameter unset.
            model_params["outage_wait_s"] = outage_wait_s
        model = openai_model.OpenAIModel("test-model", params=model_params)
        models.append(model)
        return model, requests, clock

    yield make
    for model in models:
        model.client.close()


def call(model):
    return model.complete("system", [{"role": "user", "content": "hello"}], [], 16, {})


@pytest.mark.parametrize("status", [429, 402])
@pytest.mark.parametrize("headers,backoff,delay", [
    ({}, 4, 4),
    ({"Retry-After": "7"}, 4, 7),
    ({"Retry-After": formatdate(1_700_000_007, usegmt=True)}, 4, 7),
    ({"retry-after-ms": "1500"}, 4, 1.5),
    ({"retry-after-ms": "1500", "Retry-After": "7"}, 4, 1.5),
    ({"Retry-After": "300"}, 4, 120),
    ({"retry-after-ms": "300000"}, 4, 120),
    ({}, 300, 120),
    ({"Retry-After": "invalid"}, 4, 4),
    ({"Retry-After": "-1"}, 4, 4),
    ({"Retry-After": "nan"}, 4, 4),
    ({"retry-after-ms": "invalid", "Retry-After": "7"}, 4, 7),
])
def test_402_and_429_retries_with_hint_or_backoff_and_records_outage(
        http_model, capsys, status, headers, backoff, delay):
    model, requests, clock = http_model(
        [error(status, headers), httpx.Response(200, json=completion())],
        outage_wait_s=delay, outage_backoff_s=backoff)
    response = call(model)
    assert response.text == "ok"
    assert clock.sleeps == [delay]
    assert response.extra == {"outage": [100.0, 100.0 + delay]}
    assert len(requests) == 2
    assert all(request.url.path == "/api/v1/chat/completions" for request in requests)
    stderr = capsys.readouterr().err
    if status == 402:
        assert stderr == f"[openai] credits exhausted (402); waiting {delay:.0f}s for a top-up\n"
    else:
        assert "endpoint unavailable (RateLimitError)" in stderr
        assert "credits exhausted" not in stderr


@pytest.mark.parametrize("status", [429, 402])
def test_outage_interval_starts_at_first_failure_and_ends_at_success(http_model, status):
    def fail(request):
        clock.now += 2
        return error(status)

    def succeed(request):
        clock.now += 3
        return httpx.Response(200, json=completion(provider="Example Provider", id="gen-123"))

    model, requests, clock = http_model([fail, succeed], outage_wait_s=10)
    assert call(model).extra == {"outage": [102.0, 109.0], "provider": "Example Provider",
                                 "response_id": "gen-123"}
    assert clock.sleeps == [4]
    assert len(requests) == 2


@pytest.mark.parametrize("status", [429, 402])
def test_402_and_429_wait_is_excluded_from_gateway_wall_budget(
        http_model, tmp_path, monkeypatch, status):
    from conftest import stub_services
    from rsibench import budget
    from rsibench.orchestrator import gateway

    model, requests, clock = http_model([
        error(status, {"Retry-After": "7"}),
        httpx.Response(200, json=completion(provider="Example Provider", id="gen-123")),
    ])
    monkeypatch.setattr(budget, "time", clock)
    monkeypatch.setattr(gateway, "time", clock)
    services, token, usage = stub_services(tmp_path, model)
    grant = services.authorize(token)
    grant.budget._start = clock.now  # the dataclass factory retains the original clock
    response = services.complete(grant, {
        "system": "system", "messages": [{"role": "user", "content": "hello"}], "tools": [],
    })

    assert response["text"] == "ok"
    assert clock.sleeps == [7]
    assert len(requests) == 2
    assert grant.budget.used().queue_s == sum(clock.sleeps)
    assert grant.budget.used().wall_s == 0
    assert usage[0]["provider"] == "Example Provider"
    assert usage[0]["response_id"] == "gen-123"
    assert usage[0]["latency_s"] == 7


@pytest.mark.parametrize("status", [429, 402])
@pytest.mark.parametrize("headers,wait,sleeps", [
    ({}, 9, [4, 4]),
    ({"Retry-After": "3"}, 7, [3, 3]),
    ({"Retry-After": "7"}, 6, []),
    ({"Retry-After": "0"}, 0, []),
])
def test_persistent_402_and_429_keeps_existing_model_error(
        http_model, status, headers, wait, sleeps):
    model, requests, clock = http_model([error(status, headers)] * 4, outage_wait_s=wait)
    with pytest.raises(ModelError, match=f"^API error {status}:") as caught:
        call(model)
    expected_cause = openai.RateLimitError if status == 429 else openai.APIStatusError
    assert isinstance(caught.value.__cause__, expected_cause)
    assert caught.value.__cause__.status_code == status
    assert clock.sleeps == sleeps
    assert len(requests) == len(sleeps) + 1


@pytest.mark.parametrize("status", [429, 402])
def test_402_and_429_without_outage_wait_raise_immediately(http_model, status):
    model, requests, clock = http_model([error(status, {"Retry-After": "1"})],
                                      outage_wait_s=None)
    assert "outage_wait_s" not in model.params
    with pytest.raises(ModelError, match=f"^API error {status}:") as caught:
        call(model)
    assert isinstance(caught.value.__cause__, openai.APIStatusError)
    assert caught.value.__cause__.status_code == status
    assert clock.sleeps == []
    assert len(requests) == 1


@pytest.mark.parametrize("status", [502, 503, 504])
def test_existing_gateway_outages_keep_backoff(http_model, status):
    model, requests, clock = http_model([
        error(status, {"Retry-After": "8"}), httpx.Response(200, json=completion())])
    assert call(model).extra == {"outage": [100.0, 104.0]}
    assert clock.sleeps == [4]
    assert len(requests) == 2


def test_existing_connection_outage_keeps_backoff(http_model):
    def fail(request):
        raise httpx.ConnectError("connection refused", request=request)

    model, requests, clock = http_model([fail, httpx.Response(200, json=completion())])
    assert call(model).extra == {"outage": [100.0, 104.0]}
    assert clock.sleeps == [4]
    assert len(requests) == 2


def test_500_does_not_enter_extended_outage_wait(http_model):
    model, requests, clock = http_model([error(500, {"Retry-After": "1"})])
    with pytest.raises(ModelError, match="^API error 500:"):
        call(model)
    assert clock.sleeps == []
    assert len(requests) == 1


@pytest.mark.parametrize("base_url,fields,expected", [
    ("https://openrouter.ai/api/v1", {"provider": "Example Provider", "id": "gen-123"},
     {"provider": "Example Provider", "response_id": "gen-123"}),
    ("https://api.openai.com/v1", {"id": "chatcmpl-123"}, {"response_id": "chatcmpl-123"}),
    ("https://self-hosted.test/v1", {}, {}),
])
def test_optional_upstream_metadata(http_model, base_url, fields, expected):
    model, requests, clock = http_model([httpx.Response(200, json=completion(**fields))],
                                      base_url=base_url)
    assert call(model).extra == expected
    assert clock.sleeps == []
    assert len(requests) == 1
