"""Offline models for tests and dry runs. They never call a network.

mock:finish    calls `finish` on its first turn. Exercises the full round lifecycle.
mock:scripted  replays a JSON script (params.script or $RSIB_MOCK_SCRIPT):

    {
      "executor": {"1": [turn, ...], "toy-segmentation:2": [turn, ...], "*": [turn, ...]},
      "improver": {...},
      "meta": {...}
    }

    turn = {"text": "...", "tool_calls": [{"name": "write_file", "arguments": {...}}]}

Turns are looked up by "<task>:<round>", then "<round>", then "*", and consumed in order within
one phase of one round. When a list runs out the model calls `finish`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .base import ChatModel, ModelResponse


def _approx_tokens(obj) -> int:
    return max(1, len(json.dumps(obj, default=str)) // 4)


class _MockBase(ChatModel):
    adapter = "mock"

    def _response(self, turn: dict, request: dict) -> ModelResponse:
        calls = []
        for i, tc in enumerate(turn.get("tool_calls", [])):
            calls.append({"id": f"call_{self._counter}_{i}", "name": tc["name"],
                          "arguments": tc.get("arguments", {})})
        self._counter += 1
        raw = {"mock_turn": turn}
        return ModelResponse(
            text=turn.get("text", ""), tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn", raw=raw, model=self.name,
            input_tokens=_approx_tokens(request), output_tokens=_approx_tokens(turn),
        )


class FinishModel(_MockBase):
    def __init__(self, name: str = "mock-finish", effort=None, params=None):
        super().__init__(name, effort, params)
        self._counter = 0

    def complete(self, system, messages, tools, max_tokens, meta) -> ModelResponse:
        turn = {"text": "Nothing to do.",
                "tool_calls": [{"name": "finish", "arguments": {"summary": "mock: no changes"}}]}
        return self._response(turn, {"system": system, "messages": messages})


class ScriptedModel(_MockBase):
    def __init__(self, name: str = "mock-scripted", effort=None, params=None):
        super().__init__(name, effort, params)
        path = self.params.get("script") or os.environ.get("RSIB_MOCK_SCRIPT")
        if not path:
            raise ValueError("mock:scripted needs params.script or $RSIB_MOCK_SCRIPT")
        self.script = json.loads(Path(path).read_text())
        self._cursor: dict[tuple, int] = {}
        self._counter = 0

    def _turns(self, phase: str, task: str, rnd: int) -> list[dict]:
        table = self.script.get(phase, {})
        for key in (f"{task}:{rnd}", str(rnd), "*"):
            if key in table:
                return table[key]
        return []

    def complete(self, system, messages, tools, max_tokens, meta) -> ModelResponse:
        phase, task, rnd = meta.get("phase", "?"), meta.get("task_id", "?"), int(meta.get("round", 0))
        key = (phase, task, rnd)
        turns = self._turns(phase, task, rnd)
        idx = self._cursor.get(key, 0)
        self._cursor[key] = idx + 1
        if idx < len(turns):
            turn = turns[idx]
        else:
            turn = {"text": "Done.", "tool_calls": [
                {"name": "finish", "arguments": {"summary": f"mock {phase} round {rnd} done"}}]}
        return self._response(turn, {"system": system, "messages": messages})
