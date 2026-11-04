"""Provider-neutral chat interface used between the harness and the gateway.

Neutral message list (what the harness sends, schemas/messages.schema.json):

    {"role": "user", "content": <content>}
    {"role": "assistant", "text": "...", "tool_calls": [{"id", "name", "arguments"}], "raw": <opaque>}
    {"role": "tool_results", "results": [{"tool_call_id", "content": <content>, "is_error"}]}

<content> is a string or an ordered list of text and image parts (context.py). The model must see
messages and parts in list order; an adapter that cannot deliver a context exactly names each
deterministic change it makes in `lowerings()`.

`raw` is the provider-native assistant message returned by the adapter. The harness must send it
back unchanged: some providers (Anthropic thinking blocks) reject a history whose earlier turns
were edited. Harnesses therefore keep histories append-only and compact by starting a new
conversation from a summary, never by rewriting old turns.
"""

from __future__ import annotations

import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "refusal", "pause_turn", "other")


class ModelError(RuntimeError):
    """A provider failure the adapter gives up on (bad request, auth, retries used up)."""


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[dict]
    stop_reason: str
    raw: object
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    extra: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "text": self.text, "tool_calls": self.tool_calls, "stop_reason": self.stop_reason,
            "raw": self.raw, "model": self.model,
            "usage": {
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
            },
        }


class ChatModel(ABC):
    adapter: str = "base"

    def __init__(self, name: str, effort: str | None = None, params: dict | None = None):
        self.name = name
        self.effort = effort
        self.params = dict(params or {})
        # Availability: while a call waits for a provider limit to reset, the model is expected to
        # refuse calls until `_blocked_until` (time.monotonic()); `outages` counts the times it
        # became unavailable, so the orchestrator can tell whether a phase or evaluation ran into one.
        self._availability = threading.Lock()
        self._blocked_until = 0.0
        self.outages = 0

    def note_unavailable(self, until: float) -> None:
        """The provider refuses calls (e.g. a usage limit) until about `until` (monotonic)."""
        with self._availability:
            if self._blocked_until <= time.monotonic():
                self.outages += 1
            self._blocked_until = max(self._blocked_until, until)

    def note_available(self) -> None:
        """A call went through: the model takes calls again."""
        with self._availability:
            self._blocked_until = 0.0

    def blocked_until(self) -> float:
        """Monotonic time before which calls are expected to be refused; 0.0 when available."""
        with self._availability:
            return self._blocked_until if self._blocked_until > time.monotonic() else 0.0

    @abstractmethod
    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int,
                 meta: dict) -> ModelResponse:
        """One model call. `tools` are neutral specs: {name, description, input_schema}. `meta`
        names the caller: phase (its grant's consumer), task_id, round, session and grant (an id
        unique per grant; calls of different grants may run at once)."""

    def lowerings(self, system: str, messages: list[dict], tools: list[dict],
                  meta: dict | None = None) -> list[str]:
        """Names of the deterministic lowerings `complete` would apply to this context (parts
        reordered, relabelled, added or dropped); [] when the model receives it as given."""
        return []

    def describe(self) -> dict:
        return {"adapter": self.adapter, "name": self.name, "effort": self.effort,
                "params": self.params}
