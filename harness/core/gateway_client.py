"""Client for the orchestrator gateway (model calls, experiments, dev evaluation, budget).

The gateway enforces the budget; this client only transports requests. $RSIB_GATEWAY_TOKEN is the
grant of the process it runs in (a phase, an experiment job), which fixes what the gateway serves
and is refused once that process's work is over. Standard library only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class GatewayError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class Gateway:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or os.environ["RSIB_GATEWAY_URL"]).rstrip("/")
        self.token = token or os.environ["RSIB_GATEWAY_TOKEN"]

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except ValueError:
                body = {}
            raise GatewayError(exc.code, body.get("error", "http_error"),
                               body.get("message", str(exc))) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GatewayError(0, "unreachable", str(exc)) from None

    def complete(self, system: str, messages: list, tools: list, max_tokens: int,
                 session: str = "0") -> dict:
        return self._post("/v1/complete", {"system": system, "messages": messages, "tools": tools,
                                           "max_tokens": max_tokens, "session": session},
                          timeout=3600)

    def run_experiment(self, name: str, command: str, gpus: int = 0,
                       timeout_min: float = 60) -> dict:
        return self._post("/v1/experiment", {"name": name, "command": command, "gpus": gpus,
                                             "timeout_min": timeout_min},
                          timeout=timeout_min * 60 + 24 * 3600)

    def evaluate(self, split: str = "dev", candidates: list | None = None) -> dict:
        payload = {"split": split}
        if candidates is not None:
            payload["candidates"] = candidates
        return self._post("/v1/evaluate", payload, timeout=24 * 3600)

    def budget(self) -> dict:
        return self._post("/v1/budget", {}, timeout=60)
