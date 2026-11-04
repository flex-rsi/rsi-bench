"""Calls the fixed backbone (an open VLM served by vLLM) through the evaluator's proxy.

    chat(messages, max_tokens=4096, thinking=False, **params) -> str
        an OpenAI chat-completions call; `messages` in the OpenAI format, user content a string
        or a list of {"type": "text", "text": ...} and image parts from image_part(). Extra
        sampling params (temperature, top_p, seed, ...) pass through. Returns the reply text.
    ask(content, ...) -> str
        one user turn.
    image_part(image, max_side=640) -> dict
        an image part from a path, bytes, a PIL image or an RGB numpy array, shrunk (aspect kept)
        to at most `max_side` pixels on its longer side and sent as JPEG.

Limits: one call may hold at most VLM_MAX_IMAGES images (32) and VLM_MAX_PIXELS pixels together
(32 x 640 x 360); a split may make VLM_MAX_CALLS calls (8 per question on average). The proxy
refuses anything beyond (HTTP 400) and caps max_tokens at 16384.
"""

import base64
import io
import json
import os
import urllib.request

from PIL import Image

URL = os.environ.get("VLM_URL", "")
KEY = os.environ.get("VLM_KEY", "")
MODEL = os.environ.get("VLM_MODEL", "")
MAX_IMAGES = int(os.environ.get("VLM_MAX_IMAGES", 32))
MAX_PIXELS = int(os.environ.get("VLM_MAX_PIXELS", 32 * 640 * 360))
MAX_CALLS = int(os.environ.get("VLM_MAX_CALLS", 0))


def image_part(image, max_side: int = 640) -> dict:
    if isinstance(image, Image.Image):
        im = image
    elif isinstance(image, bytes):
        im = Image.open(io.BytesIO(image))
    elif hasattr(image, "shape"):
        im = Image.fromarray(image)
    else:
        im = Image.open(image)
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def chat(messages: list[dict], max_tokens: int = 4096, thinking: bool = False, **params) -> str:
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": thinking}, **params}
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"] or ""


def ask(content, max_tokens: int = 4096, thinking: bool = False, system: str | None = None,
        **params) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": content}]
    return chat(messages, max_tokens=max_tokens, thinking=thinking, **params)
