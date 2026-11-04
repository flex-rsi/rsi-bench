"""OpenAI-compatible Chat Completions through the `openai` SDK.

Covers the OpenAI API and any compatible server (vLLM, SGLang, NIM, Gemini's OpenAI endpoint)
via `base_url`. Spec forms accepted by `rsib`:

    openai:<model>                       -> api.openai.com, key from OPENAI_API_KEY
    openai:<model>@<base_url>            -> custom endpoint
    params: api_key_env=<VAR>            -> read the key from another variable
            max_tokens_param=max_tokens  -> for servers that do not know max_completion_tokens
            outage_wait_s=<seconds>      -> keep retrying connection errors and 402/429/502/503/504
                                            for this long
            outage_backoff_s=<seconds>   -> pause between those retries (default 30)
            extra_body={...}             -> extra request fields (e.g. chat_template_kwargs)

The key is read from the environment or the dotenv file (rsibench.localenv).

Lowering `tool_results_as_user_message`: Chat Completions cannot put images in tool messages, so a
tool_results message holding an image becomes one stub tool message per call followed by one user
message with all of its results in order (wrapped as <tool_result> text, images in place).
Tool messages carry no error flag, so an error result's text starts with "ERROR: " (in the
lowered form, its <tool_result> tag is marked "(error)").
"""

from __future__ import annotations

import json
import math
import sys
import time
from email.utils import parsedate_to_datetime

from .. import localenv
from . import context
from .base import ChatModel, ModelError, ModelResponse

_STOP_MAP = {"stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use",
             "length": "max_tokens", "content_filter": "refusal"}
STUB = "(output in the next user message)"


def _image_results(msg: dict) -> bool:
    """A tool_results message holding an image: the one shape that needs a lowering."""
    return msg["role"] == "tool_results" and bool(context.images_in([msg]))


def _text(content) -> str:
    """Text-only content as one string (adjacent text parts read as one text)."""
    return content if isinstance(content, str) else "".join(p["text"] for p in content)


def to_api_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        if role == "user":
            content = msg["content"]
            out.append({"role": "user", "content": context.to_openai(context.merge(content))
                        if context.images_in([msg]) else _text(content)})
        elif _image_results(msg):
            out += [{"role": "tool", "tool_call_id": r["tool_call_id"], "content": STUB}
                    for r in msg["results"]]
            out.append({"role": "user", "content": context.to_openai(context.delta([msg]))})
        elif role == "tool_results":
            for r in msg["results"]:
                content = _text(context.result_parts(r["content"]))
                if r.get("is_error"):
                    content = "ERROR: " + content
                out.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": content})
        elif role == "assistant":
            raw = msg.get("raw")
            if isinstance(raw, dict) and raw.get("role") == "assistant":
                out.append(raw)
            else:
                entry = {"role": "assistant", "content": msg.get("text") or ""}
                if msg.get("tool_calls"):
                    entry["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                        for tc in msg["tool_calls"]
                    ]
                out.append(entry)
        else:
            raise ValueError(f"unknown neutral role {role!r}")
    return out


def to_api_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def _retry_after_s(headers) -> float | None:
    """Retry delay in seconds, preferring the millisecond header when both are present."""
    for key, scale in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(key)
        if value is None:
            continue
        try:
            delay = float(value) / scale
        except ValueError:
            if key == "retry-after-ms":
                continue
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay) and delay >= 0:
            return delay
    return None


class OpenAIModel(ChatModel):
    adapter = "openai"

    def __init__(self, name: str, effort: str | None = None, params: dict | None = None):
        super().__init__(name, effort, params)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ModelError("install the openai SDK: pip install openai") from exc
        self._openai = openai
        key_env = self.params.get("api_key_env", "OPENAI_API_KEY")
        self.client = openai.OpenAI(
            api_key=localenv.lookup(key_env) or "EMPTY",
            base_url=self.params.get("base_url") or None,
            max_retries=int(self.params.get("max_retries", 4)),
            timeout=float(self.params.get("timeout_s", 1800)),
        )

    def lowerings(self, system, messages, tools, meta=None) -> list[str]:
        return ["tool_results_as_user_message"] if any(map(_image_results, messages)) else []

    def complete(self, system, messages, tools, max_tokens, meta) -> ModelResponse:
        kwargs = {
            "model": self.name,
            "messages": to_api_messages(system, messages),
            self.params.get("max_tokens_param", "max_completion_tokens"): int(max_tokens),
        }
        if tools:  # OpenAI and vLLM refuse an empty tools array: a call without tools omits it
            kwargs["tools"] = to_api_tools(tools)
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        if self.params.get("extra_body"):
            kwargs["extra_body"] = dict(self.params["extra_body"])
        o = self._openai
        try:
            resp, outage = self._create_through_outages(kwargs)
        except o.BadRequestError as exc:
            raise ModelError(f"bad request: {exc}") from exc
        except o.AuthenticationError as exc:
            raise ModelError("authentication failed; check the API key") from exc
        except o.APIStatusError as exc:
            raise ModelError(f"API error {exc.status_code}: {exc}") from exc
        except o.APIConnectionError as exc:
            raise ModelError(f"connection error: {exc}") from exc

        if not resp.choices:
            raise ModelError("the server returned no choices")
        choice = resp.choices[0]
        raw = choice.message.model_dump(exclude_none=True)
        raw["role"] = "assistant"
        tool_calls = []
        for tc in raw.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"__invalid_json__": fn.get("arguments", "")}
            tool_calls.append({"id": tc["id"], "name": fn.get("name", ""), "arguments": args})
        usage = resp.usage
        cached = 0
        if usage is not None and getattr(usage, "prompt_tokens_details", None) is not None:
            cached = int(getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        extra = {"outage": list(outage)} if outage else {}
        for attr, key in (("provider", "provider"), ("id", "response_id")):
            value = getattr(resp, attr, None)
            if isinstance(value, str):
                extra[key] = value
        return ModelResponse(
            text=raw.get("content") or "",
            tool_calls=tool_calls,
            stop_reason=_STOP_MAP.get(choice.finish_reason or "", "other"),
            raw=raw,
            model=resp.model or self.name,
            input_tokens=max(0, prompt - cached),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cache_read_tokens=cached,
            extra=extra,
        )


    def _create_through_outages(self, kwargs: dict):
        """One request; with `outage_wait_s` set, connection failures, exhausted credits,
        rate limits and gateway errors are retried until that much time has passed.
        Returns (response, outage) where outage is the (start, end) monotonic interval the
        endpoint was unreachable, or None; the gateway leaves it out of the wall-time budget."""
        o = self._openai
        wait = float(self.params.get("outage_wait_s", 0))
        backoff = float(self.params.get("outage_backoff_s", 30))
        start = time.monotonic()
        down = None  # when the endpoint was first found unreachable during this call
        while True:
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp, ((down, time.monotonic()) if down is not None else None)
            except (o.APIConnectionError, o.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, o.APIConnectionError) or \
                    status in (402, 429, 502, 503, 504)
                delay = backoff
                if status in (402, 429):
                    retry_after = _retry_after_s(exc.response.headers)
                    delay = min(120, backoff if retry_after is None else retry_after)
                if not transient or wait <= 0 or time.monotonic() - start + delay > wait:
                    raise
                if down is None:
                    down = time.monotonic()
                if status == 402:
                    print(f"[openai] credits exhausted (402); waiting {delay:.0f}s for a top-up",
                          file=sys.stderr, flush=True)
                else:
                    print(f"[openai] endpoint unavailable ({type(exc).__name__}); retrying in "
                          f"{delay:.0f}s", file=sys.stderr, flush=True)
                time.sleep(delay)
