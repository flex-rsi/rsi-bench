"""The solver's only route to the fixed backbone: an OpenAI-compatible proxy in the evaluator.

It forwards POST /v1/chat/completions to the frozen vLLM server ($FIXED_VLM_URL, serving
$FIXED_VLM_MODEL) with the model name forced to the backbone, and refuses (HTTP 400) a call whose
images are more than MAX_IMAGES or hold more than MAX_PIXELS pixels together (error
"field_of_view"), a call with any other media (error "media"), and every call after the split's
first CALLS_PER_QUESTION x n (error "call_budget"). max_tokens is capped at MAX_TOKENS."""

import base64
import io
import json
import secrets
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


def image_stats(payload: dict) -> tuple[int, int, str | None]:
    """(images, pixels, other media type) of the messages of a chat-completions request."""
    images = pixels = 0
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        for part in content if isinstance(content, list) else []:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in (None, "text"):
                continue
            if kind != "image_url":
                return images, pixels, kind
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                return images, pixels, "remote image url"
            w, h = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
            images, pixels = images + 1, pixels + w * h
    return images, pixels, None


class Proxy:
    def __init__(self, upstream: str, model: str, max_calls: int, max_images: int,
                 max_pixels: int, max_tokens: int):
        self.upstream, self.model = upstream.rstrip("/"), model
        self.max_calls, self.max_images, self.max_pixels = max_calls, max_images, max_pixels
        self.max_tokens = max_tokens
        self.key = secrets.token_hex(16)
        self.calls = self.refused = self.prompt_tokens = self.completion_tokens = 0
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def refuse(self, error: str, message: str) -> None:
                with proxy.lock:
                    proxy.refused += 1
                self.reply(400, json.dumps({"error": error, "message": message}).encode())

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Authorization") != f"Bearer {proxy.key}":
                    return self.reply(401, b'{"error": "unauthorized"}')
                if self.path.rstrip("/") != "/v1/chat/completions":
                    return self.reply(404, b'{"error": "only /v1/chat/completions"}')
                try:
                    payload = json.loads(raw or b"{}")
                    images, pixels, other = image_stats(payload)
                except Exception as exc:
                    return self.reply(400, json.dumps({"error": "bad_request",
                                                       "message": str(exc)[:300]}).encode())
                if other:
                    return self.refuse("media", f"only text and base64 image parts; got {other}")
                if images > proxy.max_images or pixels > proxy.max_pixels:
                    return self.refuse("field_of_view", f"one call may hold at most "
                                       f"{proxy.max_images} images and {proxy.max_pixels} pixels "
                                       f"together; this call has {images} and {pixels}")
                with proxy.lock:
                    over = proxy.calls >= proxy.max_calls
                    if not over:
                        proxy.calls += 1
                if over:
                    return self.refuse("call_budget", f"this split may make at most "
                                       f"{proxy.max_calls} backbone calls")
                payload["model"] = proxy.model
                payload["stream"] = False
                payload["n"] = 1
                payload["max_tokens"] = min(int(payload.get("max_tokens") or proxy.max_tokens),
                                            proxy.max_tokens)
                payload.pop("max_completion_tokens", None)
                req = urllib.request.Request(proxy.upstream + "/v1/chat/completions",
                                             data=json.dumps(payload).encode(), method="POST",
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=1800) as resp:
                        body = resp.read()
                    usage = json.loads(body).get("usage") or {}
                    with proxy.lock:
                        proxy.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                        proxy.completion_tokens += int(usage.get("completion_tokens") or 0)
                    return self.reply(200, body)
                except urllib.error.HTTPError as exc:
                    if exc.code >= 500:
                        proxy.refund()
                    return self.reply(exc.code, exc.read())
                except Exception as exc:
                    proxy.refund()
                    return self.reply(502, json.dumps({"error": "unreachable",
                                                       "message": str(exc)[:300]}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def refund(self) -> None:
        """A call the backbone failed to serve does not count against the split's budget."""
        with self.lock:
            self.calls -= 1
