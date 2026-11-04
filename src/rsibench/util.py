"""Small shared helpers: time, hashing, JSON / JSONL I/O, slugs, process groups."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Iterable


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def slugify(text: str) -> str:
    """Lower-case, git-ref-safe slug (keeps letters, digits, '.', '_' and '-')."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.").lower()
    slug = re.sub(r"\.{2,}", ".", slug)
    return slug or "x"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """Content hash of a directory: relative paths and file bytes, in sorted order."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel.split("/"):
            continue
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sha256_file(path).encode())
        h.update(b"\n")
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def chunks(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def kill_group(pid: int) -> None:
    """SIGKILL the process group led by `pid` (started with start_new_session=True), including
    whatever its leader left running. The group may already be gone."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def wait_group(pid: int, timeout_s: float = 5.0) -> None:
    """Wait, at most `timeout_s`, until the process group led by `pid` has no member left; a killed
    member counts until it is reaped."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.02)
