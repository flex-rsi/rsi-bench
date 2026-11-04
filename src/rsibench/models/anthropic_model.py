"""Claude through the official `anthropic` SDK (pip install 'agent-rsi-benchmarking[anthropic]').

Credentials: the key named by `api_key_env` (default ANTHROPIC_API_KEY), else ANTHROPIC_AUTH_TOKEN,
each from the environment or the dotenv file (rsibench.localenv); with neither, the SDK resolves
its own (an `ant auth login` profile). Notes on choices that matter for a benchmark:

* Assistant turns are replayed exactly as returned (`raw`), including thinking blocks; the
  harness keeps histories append-only.
* Effort is fixed per run (`output_config.effort`) and cannot be changed by the harness.
* Server-side refusal fallbacks are NOT enabled: a fallback silently switches to a different
  model, which would contaminate a per-model measurement. A refusal ends the phase and is recorded.
* Requests stream (long turns can run for minutes) and use automatic prompt caching.
* The context needs no lowering: images go into user content and tool results in place.
"""

from __future__ import annotations

from .. import localenv
from . import context
from .base import ChatModel, ModelError, ModelResponse

_STOP_MAP = {
    "end_turn": "end_turn", "stop_sequence": "end_turn", "tool_use": "tool_use",
    "max_tokens": "max_tokens", "refusal": "refusal", "pause_turn": "pause_turn",
}


def _result_content(content) -> str | list[dict]:
    items = context.result_parts(content)
    if len(items) == 1 and items[0]["type"] == "text":
        return items[0]["text"]
    return context.to_blocks(items)


def to_api_messages(messages: list[dict]) -> list[dict]:
    """Neutral -> Anthropic messages. Consecutive user-side items are merged into one turn."""
    out: list[dict] = []

    def push_user(blocks: list[dict]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": "user", "content": list(blocks)})

    for msg in messages:
        role = msg["role"]
        if role == "user":
            push_user(context.to_blocks(context.merge(context.parts(msg["content"]))))
        elif role == "tool_results":
            push_user([
                {"type": "tool_result", "tool_use_id": r["tool_call_id"],
                 "content": _result_content(r["content"]), "is_error": bool(r.get("is_error"))}
                for r in msg["results"]
            ])
        elif role == "assistant":
            raw = msg.get("raw")
            if isinstance(raw, list) and raw:
                content = raw
            else:  # history produced by another adapter: rebuild plain blocks
                content = []
                if msg.get("text"):
                    content.append({"type": "text", "text": msg["text"]})
                for tc in msg.get("tool_calls", []):
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"],
                                    "input": tc["arguments"]})
            out.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"unknown neutral role {role!r}")
    return out


def to_api_tools(tools: list[dict], streaming: bool) -> list[dict]:
    api = []
    for t in tools:
        spec = {"name": t["name"], "description": t["description"],
                "input_schema": t["input_schema"]}
        if streaming:
            # Large tool inputs (file contents) stream as they are generated. The harness
            # validates every tool input against its schema before running it.
            spec["eager_input_streaming"] = True
        api.append(spec)
    return api


class AnthropicModel(ChatModel):
    adapter = "anthropic"

    def __init__(self, name: str, effort: str | None = None, params: dict | None = None):
        super().__init__(name, effort, params)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("install the anthropic SDK: pip install anthropic") from exc
        self._anthropic = anthropic
        key = localenv.lookup(self.params.get("api_key_env", "ANTHROPIC_API_KEY"))
        token = None if key else localenv.lookup("ANTHROPIC_AUTH_TOKEN")
        auth = {"api_key": key} if key else {"auth_token": token} if token else {}
        self.client = anthropic.Anthropic(
            **auth,
            max_retries=int(self.params.get("max_retries", 4)),
            timeout=float(self.params.get("timeout_s", 1800)),
        )
        self.use_cache = bool(self.params.get("prompt_cache", True))

    def complete(self, system, messages, tools, max_tokens, meta) -> ModelResponse:
        kwargs = {
            "model": self.name,
            "max_tokens": int(max_tokens),
            "system": system,
            "messages": to_api_messages(messages),
        }
        if tools:  # a call without tools (e.g. a solution asking about an image) sends none
            kwargs["tools"] = to_api_tools(tools, streaming=True)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.use_cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        a = self._anthropic
        try:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except a.BadRequestError as exc:
            raise ModelError(f"bad request: {exc.message}") from exc
        except a.AuthenticationError as exc:
            raise ModelError("authentication failed; check Anthropic credentials") from exc
        except a.PermissionDeniedError as exc:
            raise ModelError(f"permission denied: {exc.message}") from exc
        except a.NotFoundError as exc:
            raise ModelError(f"model or endpoint not found: {exc.message}") from exc
        except a.APIStatusError as exc:  # 429 / 5xx after the SDK's own retries
            raise ModelError(f"API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise ModelError(f"connection error: {exc}") from exc

        data = msg.to_dict(mode="json")
        content = data.get("content", [])
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        tool_calls = [
            {"id": b["id"], "name": b["name"],
             "arguments": b.get("input") if isinstance(b.get("input"), dict) else {}}
            for b in content if b.get("type") == "tool_use"
        ]
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=_STOP_MAP.get(data.get("stop_reason") or "", "other"),
            raw=content,
            model=data.get("model", self.name),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            extra={"stop_details": data.get("stop_details"), "request_id": getattr(msg, "_request_id", None)},
        )
