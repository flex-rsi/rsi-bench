"""Call the run's model from the controller (standard library plus Pillow).

The evaluator gives each episode its own gateway address in RSIB_GATEWAY_URL. A call returns the
gateway reply {"text", "tool_calls", "usage"}; HTTP errors raise ModelCallError with the status:
402 means the episode used up its model calls, 409 that a MAX Dynamic arm asked something other
than its Base arm did during the shared prefix.
"""
import base64
import io
import json
import os
import urllib.error
import urllib.request

from PIL import Image


class ModelCallError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"model call failed with HTTP {status}: {body}")
        self.status, self.body = status, body


def available():
    return bool(os.environ.get("RSIB_GATEWAY_URL"))


def image_part(rgb_uint8, fmt="PNG"):
    """[H, W, 3] uint8 -> a gateway image part."""
    buffer = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buffer, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return {"type": "image", "mime": mime, "data": base64.b64encode(buffer.getvalue()).decode()}


def text_part(text):
    return {"type": "text", "text": text}


def ask(system, content, max_tokens=1024, session="0", tools=(), history=(), timeout=900):
    """One model call: `content` is a list of parts for a new user message after `history`."""
    payload = {"system": system,
               "messages": list(history) + [{"role": "user", "content": list(content)}],
               "tools": list(tools), "max_tokens": int(max_tokens), "session": str(session)}
    request = urllib.request.Request(
        os.environ["RSIB_GATEWAY_URL"].rstrip("/") + "/v1/complete",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("RSIB_GATEWAY_TOKEN", "")})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except ValueError:
            body = {"error": "http_error"}
        raise ModelCallError(exc.code, body) from None
