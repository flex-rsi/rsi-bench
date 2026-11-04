"""Starter solution: the direct "look and answer" pipeline. Show the model the whole photo, shrunk
to the field of view, with the question's prompt, and read the option letter it answers with."""

import re

from model import complete, image_part

WORKERS = 8  # questions answered at once


def answer(item: dict) -> str | None:
    reply = complete([image_part(item["image"]), {"type": "text", "text": item["prompt"]}])
    m = re.search(r"\b([A-D])\b", reply)
    return m.group(1) if m else None
