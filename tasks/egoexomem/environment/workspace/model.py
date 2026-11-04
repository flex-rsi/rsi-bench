"""Calls the model under test through the gateway the evaluator hands the solver.

    complete(content, max_tokens=16384, system=...) -> str
        one user turn; `content` is a string or a list of parts, in the order the model sees them:
        {"type": "text", "text": ...} and image parts from image_part(). Returns the reply text.
    image_part(image, max_pixels=MAX_PIXELS) -> dict
        an image part from a path, bytes or a PIL image, shrunk (aspect kept) to at most
        `max_pixels` pixels and sent as JPEG.

The frame budget: the images of one call may hold at most MAX_PIXELS pixels together (16 frames of
448 x 448); the evaluator refuses a larger call (HTTP 400, "frame_budget"). A split may make
MAX_CALLS calls (3 per question on average); the evaluator refuses every call after that (HTTP 400,
"call_budget"). On dev every call is also charged to the run's budget.
"""

import base64
import io
import json
import os
import urllib.request
import uuid

from PIL import Image

MAX_PIXELS = int(os.environ.get("EGOEXOMEM_MAX_PIXELS", 16 * 448 * 448))
MAX_CALLS = int(os.environ.get("EGOEXOMEM_MAX_CALLS", 0))  # this split's calls in total (0: unknown)
SYSTEM = "You are a helpful assistant that answers questions about videos."


def image_part(image, max_pixels: int = MAX_PIXELS) -> dict:
    im = image if isinstance(image, Image.Image) else \
        Image.open(io.BytesIO(image) if isinstance(image, bytes) else image)
    w, h = im.size
    scale = min(1.0, (max_pixels / (w * h)) ** 0.5)
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=92)
    return {"type": "image", "mime": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}


def complete(content, max_tokens: int = 16384, system: str = SYSTEM) -> str:
    payload = {"system": system, "messages": [{"role": "user", "content": content}], "tools": [],
               "max_tokens": max_tokens, "session": uuid.uuid4().hex}  # a fresh conversation
    req = urllib.request.Request(
        os.environ["RSIB_GATEWAY_URL"].rstrip("/") + "/v1/complete",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['RSIB_GATEWAY_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())["text"]
