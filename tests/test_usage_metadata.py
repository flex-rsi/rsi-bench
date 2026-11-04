"""Optional upstream metadata reaches schema-checked usage records without changing old shapes."""

import json

import pytest

from conftest import stub_services
from rsibench.models.mock import FinishModel
from rsibench.models.openai_model import OpenAIModel
from rsibench.records import Records, validate


PAYLOAD = {"system": "SYS", "messages": [{"role": "user", "content": "go"}], "tools": []}
USAGE_KEYS = {"ts", "task_id", "round", "phase", "model", "input_tokens", "output_tokens",
              "cache_read_tokens", "cache_write_tokens", "usd", "stop_reason", "latency_s",
              "lowerings"}


def _record_call(tmp_path, model):
    services, token, _ = stub_services(tmp_path, model)
    records = Records(tmp_path)
    records.ensure_layout()
    services.engine.records = records
    services.complete(services.authorize(token), PAYLOAD)
    rows = records.read("usage")
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("metadata", [{}, {"provider": "Test Provider"}, {"id": "gen-123"},
                                     {"provider": "Test Provider", "id": "gen-123"}])
def test_http_response_metadata_in_usage(tmp_path, metadata):
    pytest.importorskip("openai")
    try:
        import httpx2 as httpx
    except ImportError:  # older OpenAI SDK releases
        import httpx

    base_url = "https://openrouter.ai/api/v1" if metadata else "https://compatible.example/v1"
    body = {"object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            **metadata}

    def respond(request):
        assert request.url.path == ("/api/v1/chat/completions" if metadata else
                                    "/v1/chat/completions")
        assert json.loads(request.content)["model"] == "test-model"
        return httpx.Response(200, json=body)

    model = OpenAIModel("test-model", params={"base_url": base_url, "max_retries": 0})
    model.client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model.client = model._openai.OpenAI(api_key="test", base_url=model.params["base_url"],
                                            max_retries=0, http_client=client)
        row = _record_call(tmp_path, model)
    expected = {key: metadata[source] for key, source in (("provider", "provider"),
                                                         ("response_id", "id"))
                if source in metadata}
    assert set(row) == USAGE_KEYS | set(expected)
    assert {key: row[key] for key in expected} == expected
    assert row["input_tokens"] == 10 and row["output_tokens"] == 5
    assert validate("usage", row) == []


def test_other_adapter_usage_shape_is_unchanged(tmp_path):
    row = _record_call(tmp_path, FinishModel())
    assert set(row) == USAGE_KEYS
    assert row["model"] == "mock-finish"
    assert validate("usage", row) == []


@pytest.mark.parametrize("key", ["provider", "response_id"])
def test_usage_schema_rejects_null_metadata(tmp_path, key):
    row = _record_call(tmp_path, FinishModel())
    assert validate("usage", {**row, key: None}) == [f"{key}: None is not of type 'string'"]
