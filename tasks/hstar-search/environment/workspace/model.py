"""Calls the model under test through the gateway the evaluator hands the policy.

    chat(messages, system=..., max_tokens=...) -> (text, assistant_message)
        one model turn on a whole conversation. `messages` is the conversation so far, oldest
        first: {"role": "user", "content": <content>} and the assistant messages chat() returned,
        unchanged. Append the returned assistant message before the next user turn.
    image_part(path) -> dict
        an image part (JPEG) for a user content list.

A content is a string or an ordered list of parts: {"type": "text", "text": ...} and image parts.
A split of n episodes may make 20n model calls in total (HSTAR_MAX_CALLS); calls after that are
refused (HTTP 400, "call_budget"). On dev every call is also charged to the run's budget.
"""

import base64
import json
import os
import urllib.request
import uuid

MAX_CALLS = int(os.environ.get("HSTAR_MAX_CALLS", 0))  # this split's calls in total (0: unknown)


def image_part(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return {"type": "image", "mime": mime, "data": base64.b64encode(data).decode()}


def chat(messages: list, system: str = "", max_tokens: int = 8192,
         session: str | None = None) -> tuple[str, dict]:
    payload = {"system": system, "messages": messages, "tools": [], "max_tokens": max_tokens,
               "session": session or uuid.uuid4().hex}
    req = urllib.request.Request(
        os.environ["RSIB_GATEWAY_URL"].rstrip("/") + "/v1/complete",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['RSIB_GATEWAY_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.loads(resp.read())
    text = out.get("text") or ""
    return text, {"role": "assistant", "text": text, "raw": out.get("raw")}
