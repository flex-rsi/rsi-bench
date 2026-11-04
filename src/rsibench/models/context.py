"""Canonical message content and the transport shapes made from it. Pure functions, no I/O.

A content is a string or an ordered list of parts, and the order of the parts is the order the
model sees (a string is one text part):

    {"type": "text", "text": "..."}
    {"type": "image", "mime": "image/png" | "image/jpeg" | "image/gif" | "image/webp",
     "data": "<base64>"}

Content appears in user messages and in tool results; schemas/messages.schema.json is the whole
message contract. Every adapter builds its provider's shape here and nowhere else.
"""

from __future__ import annotations

import base64
import hashlib
import json

EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
EMPTY_RESULT = "(no output)"
REPLAY_NOTE = ("[The conversation so far, in order. Your own earlier replies are the "
               "<assistant_turn> blocks.]")


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def parts(content) -> list[dict]:
    """Content as a list of parts (a string becomes one text part)."""
    return [text(content)] if isinstance(content, str) else list(content)


def result_parts(content) -> list[dict]:
    """A tool result's parts for a provider's native tool result: merged, EMPTY_RESULT if empty."""
    return merge(parts(content)) or [text(EMPTY_RESULT)]


def digest(messages: list[dict]) -> str:
    """Fingerprint of a message list, to tell whether a later call kept its earlier messages."""
    blob = json.dumps(messages, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def merge(items: list[dict]) -> list[dict]:
    """Adjacent text parts joined, empty ones dropped; images stay in place."""
    out: list[dict] = []
    for p in items:
        if p["type"] != "text":
            out.append(p)
        elif p["text"]:
            if out and out[-1]["type"] == "text":
                out[-1] = text(out[-1]["text"] + p["text"])
            else:
                out.append(text(p["text"]))
    return out


def _items(msg: dict) -> list[list[dict]]:
    """A user-side message as items of parts: its content, or each tool result wrapped in
    <tool_result id="..."> tags (an empty result leaves the body empty)."""
    if msg["role"] == "user":
        return [parts(msg["content"])]
    if msg["role"] != "tool_results":
        return []
    items = []
    for r in msg["results"]:
        flag = " (error)" if r.get("is_error") else ""
        items.append([text(f'<tool_result id="{r["tool_call_id"]}"{flag}>\n'),
                      *parts(r["content"]), text("\n</tool_result>")])
    return items


def _joined(items: list[list[dict]]) -> list[dict]:
    """Items separated by a blank line, adjacent text merged."""
    joined: list[dict] = []
    for i, item in enumerate(items):
        joined += ([text("\n\n")] if i else []) + item
    return merge(joined)


def delta(messages: list[dict]) -> list[dict]:
    """The user-side messages after the last assistant reply as one ordered list of parts.

    Each tool result is wrapped in <tool_result id="..."> tags (an empty result leaves the body
    empty); items are separated by a blank line.
    """
    start = max((i + 1 for i, m in enumerate(messages) if m["role"] == "assistant"), default=0)
    return _joined([item for msg in messages[start:] for item in _items(msg)])


def replay(messages: list[dict]) -> list[dict]:
    """The whole conversation as one ordered list of parts, for a transport that cannot take
    assistant turns (a CLI session holds only the replies it produced): REPLAY_NOTE, then every
    message in order, user content and tool results as in `delta` and each assistant turn as
    <assistant_turn> text with its tool calls as <tool_call id="..." name="..."> blocks holding
    the arguments as JSON. Without assistant turns this is `delta(messages)`."""
    if not any(m["role"] == "assistant" for m in messages):
        return delta(messages)
    items: list[list[dict]] = [[text(REPLAY_NOTE)]]
    for msg in messages:
        if msg["role"] != "assistant":
            items += _items(msg)
            continue
        lines = [msg["text"]] if msg.get("text") else []
        for call in msg.get("tool_calls") or []:
            args = json.dumps(call.get("arguments", {}), ensure_ascii=False, sort_keys=True)
            lines.append(f'<tool_call id="{call.get("id", "")}" name="{call.get("name", "")}">'
                         f'{args}</tool_call>')
        items.append([text("<assistant_turn>\n" + "\n".join(lines) + "\n</assistant_turn>")])
    return _joined(items)


def images_in(messages: list[dict]) -> list[dict]:
    """Every image part of the user messages and tool results, in order."""
    contents = []
    for msg in messages:
        if msg["role"] == "user":
            contents.append(msg["content"])
        elif msg["role"] == "tool_results":
            contents += [r["content"] for r in msg["results"]]
    return [p for c in contents for p in parts(c) if p["type"] == "image"]


def replace_images(messages: list[dict], fn) -> list[dict]:
    """A copy of `messages` in which fn(part) stands where each image part stood."""
    def swap(content):
        return content if isinstance(content, str) else \
            [fn(p) if p["type"] == "image" else p for p in content]

    out = []
    for msg in messages:
        if msg["role"] == "user":
            msg = {**msg, "content": swap(msg["content"])}
        elif msg["role"] == "tool_results":
            msg = {**msg, "results": [{**r, "content": swap(r["content"])} for r in msg["results"]]}
        out.append(msg)
    return out


def to_blocks(items: list[dict]) -> list[dict]:
    """Anthropic content blocks (also what Claude Code reads as stream-json input)."""
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else
            {"type": "image", "source": {"type": "base64", "media_type": p["mime"],
                                         "data": p["data"]}}
            for p in items]


def to_openai(items: list[dict]) -> list[dict]:
    """OpenAI Chat Completions content parts."""
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else
            {"type": "image_url", "image_url": {"url": f"data:{p['mime']};base64,{p['data']}"}}
            for p in items]


def images_first(items: list[dict]) -> tuple[str, list[dict]]:
    """The text with each image replaced by "[Image #k]" (k = 1..n), and the images in order.

    This is how the codex CLI presents attachments: every image first, labelled, then the prompt.
    """
    texts, found = [], []
    for p in items:
        if p["type"] == "image":
            found.append(p)
            texts.append(f"[Image #{len(found)}]")
        else:
            texts.append(p["text"])
    return "".join(texts), found


def sniff(data: bytes) -> str | None:
    """The image format of `data` from its signature, or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_file(part: dict) -> tuple[str, bytes]:
    """Content-addressed file name (<sha16>.<ext>) and the decoded bytes of an image part."""
    data = base64.b64decode(part["data"])
    return f"{hashlib.sha256(data).hexdigest()[:16]}.{EXTENSIONS[part['mime']]}", data
