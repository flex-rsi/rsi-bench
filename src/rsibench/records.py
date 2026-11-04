"""Schema-checked writers and readers for the files under records/ in a run branch.

Only the orchestrator writes these files (records/notes/ is the exception: it belongs to the
agent). Every JSONL file is append-only; `rsib validate` checks that no commit rewrites history.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Iterable
from functools import lru_cache
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator

from .util import read_json, read_jsonl, write_json

JSONL_FILES = {
    "rounds": "round",
    "experiments": "experiment",
    "usage": "usage",
    "artifacts": "artifact",
    "violations": "violation",
    "final": "final",
    "posthoc": "posthoc",
    "environment": "environment",
}


def latest_posthoc(rows: Iterable[dict]) -> dict[tuple[str, int], dict]:
    """Latest appended posthoc row for each (task_id, round), including failed retries.

    File order determines which row supersedes another; timestamps are evidence, not ordering.
    Raw record readers retain every row so validation and audit history stay append-only.
    """
    return {(row["task_id"], row["round"]): row for row in rows}


@lru_cache(maxsize=None)
def validator(kind: str) -> Draft202012Validator:
    text = resources.files("rsibench.schemas").joinpath(f"{kind}.schema.json").read_text()
    return Draft202012Validator(json.loads(text))


def _clip(message: str) -> str:
    # messages quote the offending value (megabytes for image data); the reason is at the end
    return message if len(message) <= 300 else message[:200] + " ... " + message[-95:]


def validate(kind: str, record: dict | list, limit: int | None = None,
             values: bool = True) -> list[str]:
    """Schema errors as "path: message", at most `limit` of them. Without `values` the message
    names the broken rule instead of quoting the offending value."""
    out = []
    for err in itertools.islice(validator(kind).iter_errors(record), limit):
        message = (err.message if values
                   else f"breaks {err.validator} {json.dumps(err.validator_value)}")
        out.append(f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {_clip(message)}")
    return out


class SchemaError(ValueError):
    pass


class Records:
    """records/ directory of one run worktree."""

    def __init__(self, worktree: Path):
        self.worktree = Path(worktree)
        self.dir = self.worktree / "records"
        self.notes_dir = self.dir / "notes"
        self._lock = threading.Lock()
        # Bytes the orchestrator wrote, per file. Anything else on disk means someone else
        # (the agent) edited a record file; `tampered()` / `repair()` detect and undo that.
        self._cache: dict[Path, bytes] = {}

    def _remember(self, path: Path) -> None:
        if path not in self._cache:
            self._cache[path] = path.read_bytes() if path.exists() else b""

    def owned_paths(self) -> list[str]:
        names = [f"records/{n}.jsonl" for n in JSONL_FILES] + ["records/run.json"]
        return names

    def tampered(self) -> list[str]:
        with self._lock:
            bad = []
            for path, data in self._cache.items():
                current = path.read_bytes() if path.exists() else b""
                if current != data:
                    bad.append(path.relative_to(self.worktree).as_posix())
            return bad

    def repair(self) -> None:
        with self._lock:
            for path, data in self._cache.items():
                if not path.exists() or path.read_bytes() != data:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)

    def track_all(self) -> None:
        """Start tracking every orchestrator-owned record file at its current content."""
        with self._lock:
            self._cache = {}
            for rel in self.owned_paths():
                self._remember(self.worktree / rel)

    # ---- paths -------------------------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / f"{name}.jsonl"

    @property
    def run_path(self) -> Path:
        return self.dir / "run.json"

    # ---- run.json ----------------------------------------------------------------------------
    def write_run(self, run: dict) -> None:
        errors = validate("run", run)
        if errors:
            raise SchemaError("run.json: " + "; ".join(errors))
        write_json(self.run_path, run)
        with self._lock:
            self._cache[self.run_path] = self.run_path.read_bytes()

    def restore_run(self, committed: str | None) -> bool:
        """Put the committed run.json back if the file differs (nothing changes it after
        `rsib run new`; a phase may have, before an interruption kept it from being reverted)."""
        current = self.run_path.read_text() if self.run_path.exists() else None
        if committed is None or current == committed:
            return False
        self.run_path.write_text(committed)
        return True

    def run(self) -> dict:
        data = read_json(self.run_path)
        if data is None:
            raise FileNotFoundError(f"{self.run_path} not found; is this a run worktree?")
        return data

    # ---- JSONL -------------------------------------------------------------------------------
    def append(self, name: str, record: dict) -> None:
        kind = JSONL_FILES[name]
        errors = validate(kind, record)
        if errors:
            raise SchemaError(f"{name}.jsonl: " + "; ".join(errors))
        with self._lock:
            path = self.path(name)
            self._remember(path)
            line = (json.dumps(record, sort_keys=False) + "\n").encode()
            with open(path, "ab") as f:
                f.write(line)
            self._cache[path] += line

    def read(self, name: str) -> list[dict]:
        return read_jsonl(self.path(name))

    def ensure_layout(self) -> None:
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        keep = self.notes_dir / ".gitkeep"
        if not keep.exists():
            keep.write_text("")
        for name in JSONL_FILES:
            self.path(name).touch(exist_ok=True)


def validate_records_dir(records_dir: Path) -> list[str]:
    """Validate every record file in a records/ directory; returns human-readable errors."""
    errors: list[str] = []
    run = read_json(records_dir / "run.json")
    if run is None:
        return ["records/run.json is missing"]
    errors += [f"run.json: {e}" for e in validate("run", run)]
    for name, kind in JSONL_FILES.items():
        path = records_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{name}.jsonl:{lineno}: invalid JSON ({exc})")
                continue
            errors += [f"{name}.jsonl:{lineno}: {e}" for e in validate(kind, record)]
    return errors
