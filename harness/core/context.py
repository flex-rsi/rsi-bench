"""Building model context: message content as ordered parts. Standard library only.

A message content is a string or a list of parts, and the model sees the parts in list order:

    {"type": "text", "text": "..."}
    {"type": "image", "mime": "image/png", "data": "<base64>"}

The gateway checks every call against this format, and every model adapter delivers it in this
order or names each change it has to make (docs/harness.md, "Context fidelity").
"""

from __future__ import annotations

import base64
from pathlib import Path

# 5 MiB: its base64 fits the gateway's limit of 7,000,000 characters per image
MAX_IMAGE_BYTES = 5 * 1024 * 1024
EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def text(s: str) -> dict:
    return {"type": "text", "text": str(s)}


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image(path) -> dict:
    """An image part from a PNG, JPEG, GIF or WebP file of at most 5 MiB.

    The mime type comes from the file's bytes, so a mislabelled file is still sent correctly."""
    p = Path(path)
    if p.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"{p.name}: not an image file ({', '.join(EXTENSIONS)})")
    size = p.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"{p.name}: {size} B is over the 5 MiB image limit")
    data = p.read_bytes()
    mime = _sniff(data)
    if mime is None:
        raise ValueError(f"{p.name}: not a valid PNG, JPEG, GIF or WebP image")
    return {"type": "image", "mime": mime, "data": base64.b64encode(data).decode("ascii")}


def extend(content, more: list[dict]) -> list[dict]:
    """`content` (a string or a list of parts) followed by the parts `more`."""
    return ([text(content)] if isinstance(content, str) else list(content)) + list(more)
