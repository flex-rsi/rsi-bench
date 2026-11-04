"""Tools available to the executor and the improver.

File tools take paths relative to the worktree root (absolute paths are allowed when they fall
inside a readable root, e.g. the dev data or the transcript directory). Writes outside the
writable prefixes are refused here; the orchestrator also reverts them after the phase.
A tool returns text, or a list of content parts (context.py) such as view_image's image.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

from . import context
from .gateway_client import Gateway, GatewayError


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} characters omitted]...\n" + text[-half:]


def _verdict(result: dict) -> dict:
    """An evaluation result without its long fields; the evaluator's files stay in artifact_dir."""
    out = {k: result.get(k) for k in ("status", "score", "artifact_dir")}
    out["detail"] = str(result.get("detail") or "")[:300]
    out["omitted"] = ("metrics, artifacts, info and log_tail, too long for this output; the "
                      "evaluator's files are in artifact_dir")
    return out


SPECS = {
    "bash": {
        "description": "Run a shell command (bash -c) and return its combined stdout/stderr and "
                       "exit code. Use it for inspecting files, quick scripts and short tests. "
                       "Long training runs belong in run_experiment.",
        "input_schema": {"type": "object", "properties": {
            "command": {"type": "string", "description": "the command to run"},
            "timeout_s": {"type": "integer", "description": "optional timeout in seconds"}},
            "required": ["command"]},
    },
    "read_file": {
        "description": "Read a text file. Paths are relative to the worktree root. Use offset and "
                       "limit (in lines) for large files.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "first line to return, 1-based"},
            "limit": {"type": "integer", "description": "number of lines to return"}},
            "required": ["path"]},
    },
    "view_image": {
        "description": "Look at an image file (PNG, JPEG, GIF or WebP, at most 5 MiB), such "
                       "as a plot, a rendered frame or a predicted mask. Paths are relative to "
                       "the worktree root.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]},
    },
    "write_file": {
        "description": "Create or overwrite a text file with the given content.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]},
    },
    "edit_file": {
        "description": "Replace an exact text span in a file. old_text must appear exactly once "
                       "unless replace_all is true.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"},
            "new_text": {"type": "string"}, "replace_all": {"type": "boolean"}},
            "required": ["path", "old_text", "new_text"]},
    },
    "list_dir": {
        "description": "List files under a directory (relative to the worktree root).",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"}, "depth": {"type": "integer"}}, "required": []},
    },
    "run_experiment": {
        "description": "Run a command as a tracked experiment on the compute backend (local or "
                       "cluster), in the task workspace. The job can write a JSON dict of results "
                       "to $RSIB_METRICS_FILE: metric keys, plus optionally \"metrics\" (an "
                       "object, values may nest), \"artifacts\" (a list of {\"path\": ...} "
                       "relative to $RSIB_ARTIFACT_DIR) and \"info\" (an object). Large outputs "
                       "go to $RSIB_ARTIFACT_DIR; dev data is at $RSIB_DEV_DATA. The job may call "
                       "the model: POST $RSIB_GATEWAY_URL/v1/complete with the header "
                       "\"Authorization: Bearer $RSIB_GATEWAY_TOKEN\" and the body that "
                       "harness/core/gateway_client.py sends. GPU hours, wall time and model "
                       "calls are charged to the budget.",
        "input_schema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "short label for this experiment"},
            "command": {"type": "string"},
            "gpus": {"type": "integer", "description": "GPUs for the job (default 0: a CPU-only "
                     "job; anything that uses CUDA needs gpus >= 1)"},
            "timeout_min": {"type": "number"}},
            "required": ["name", "command"]},
    },
    "evaluate_dev": {
        "description": "Score the current workspace with the task's evaluator on the dev split. "
                       "Returns the primary metric (score), all metrics, the files the evaluator "
                       "wrote for diagnosis (artifacts: read them with read_file, look at images "
                       "with view_image), info and the tail of the evaluator's log. The evaluator "
                       "enforces the task's time limit on the whole split; a run that exceeds it "
                       "scores nothing. With candidates, scores each directory instead, "
                       "concurrently, and returns a result per id.",
        "input_schema": {"type": "object", "properties": {
            "candidates": {"type": "array", "description": "complete solutions to compare, each "
                           "a directory in the workspace or the scratch directory", "items": {
                "type": "object", "properties": {
                    "id": {"type": "string", "description": "up to 32 letters, digits, . _ -"},
                    "path": {"type": "string", "description": "relative to the worktree root"}},
                "required": ["id", "path"]}}},
            "required": []},
    },
    "check_budget": {
        "description": "Show the remaining budget (wall time, USD, tokens, GPU hours, val calls).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "finish": {
        "description": "End this phase. Give a concise report of what you did and found.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                         "required": ["summary"]},
    },
}

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list,
          "object": dict}


def _problem(value, spec: dict, name: str) -> str | None:
    """Why `value` does not fit `spec`: its type, then the items or properties it declares."""
    kind = spec["type"]
    if not isinstance(value, _TYPES[kind]) or (kind != "boolean" and isinstance(value, bool)):
        return f"argument '{name}' must be of type {kind}"
    if kind == "array" and "items" in spec:
        for i, item in enumerate(value):
            problem = _problem(item, spec["items"], f"{name}[{i}]")
            if problem:
                return problem
    if kind == "object" and "properties" in spec:
        for key in spec.get("required", []):
            if key not in value:
                return f"argument '{name}' is missing '{key}'"
        for key, item in value.items():
            if key not in spec["properties"]:
                return f"argument '{name}' has an unknown key '{key}'"
            problem = _problem(item, spec["properties"][key], f"{name}.{key}")
            if problem:
                return problem
    return None


class ToolBox:
    def __init__(self, root: Path, writable: list[str], gateway: Gateway | None, cwd: Path,
                 read_roots: list[Path] | None = None, experiments: bool = False,
                 bash_timeout_s: int = 600, max_output_chars: int = 16000):
        self.root = Path(root).resolve()
        self.writable = [(self.root / w).resolve() for w in writable]
        self.read_roots = [self.root] + [Path(r).resolve() for r in (read_roots or [])]
        self.gateway = gateway
        self.cwd = Path(cwd)
        self.bash_timeout_s = int(bash_timeout_s)
        self.max_output = int(max_output_chars)
        names = ["bash", "read_file", "view_image", "write_file", "edit_file", "list_dir"]
        if experiments:
            names += ["run_experiment", "evaluate_dev"]
        self.names = names + ["check_budget", "finish"]
        self.finished = False
        self.finish_summary = ""
        self.call_counts: dict[str, int] = {}

    def specs(self) -> list[dict]:
        return [{"name": n, **SPECS[n]} for n in self.names]

    # ---- argument checks ---------------------------------------------------------------------
    def _check(self, name: str, args: dict) -> str | None:
        if not isinstance(args, dict) or "__invalid_json__" in args:
            return "arguments were not valid JSON; re-issue the call"
        schema = SPECS[name]["input_schema"]
        for key in schema.get("required", []):
            if key not in args:
                return f"missing required argument '{key}'"
        for key, value in args.items():
            spec = schema["properties"].get(key)
            if spec is None:
                return f"unknown argument '{key}'"
            if spec["type"] == "integer" and isinstance(value, float) and value.is_integer():
                args[key] = int(value)
                continue
            problem = _problem(value, spec, key)
            if problem:
                return problem
        return None

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return (p if p.is_absolute() else self.root / p).resolve()

    def _readable(self, p: Path) -> bool:
        return any(p == r or r in p.parents for r in self.read_roots)

    def _writable(self, p: Path) -> bool:
        return any(p == w or w in p.parents for w in self.writable)

    def _rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    # ---- dispatch ----------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> tuple[str | list[dict], bool]:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        if name not in self.names:
            return f"unknown tool '{name}'. Available: {', '.join(self.names)}", True
        problem = self._check(name, args)
        if problem:
            return problem, True
        try:
            return getattr(self, f"_t_{name}")(**args), False
        except ToolError as exc:
            return str(exc), True
        except GatewayError as exc:
            return f"gateway refused: {exc.message} ({exc.code})", True
        except Exception as exc:  # report, never crash the loop
            return f"{type(exc).__name__}: {exc}", True

    # ---- tools -------------------------------------------------------------------------------
    def _t_bash(self, command: str, timeout_s: int | None = None) -> str:
        timeout = min(int(timeout_s or self.bash_timeout_s), self.bash_timeout_s)
        # Own process group, so a timeout also stops the command's children instead of leaving
        # them running; the output produced before the timeout is still returned.
        proc = subprocess.Popen(["bash", "-c", command], cwd=str(self.cwd), env=os.environ.copy(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            out = (stdout or "") + (("\n[stderr]\n" + stderr) if stderr else "")
            return _clip(f"[timed out after {timeout}s; the command and its child processes were "
                         f"stopped. Output so far:]\n{out}\n[run long jobs in the background with "
                         f"output to a log file, e.g. `nohup cmd > log 2>&1 &`, and read the log "
                         f"in a later call]", self.max_output)
        out = stdout + (("\n[stderr]\n" + stderr) if stderr else "")
        return _clip(f"[exit {proc.returncode}]\n{out}", self.max_output)

    def _t_read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        p = self._resolve(path)
        if not self._readable(p):
            raise ToolError(f"{path} is outside the readable area")
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        lines = p.read_text(errors="replace").splitlines()
        start = max(1, int(offset or 1))
        end = len(lines) if not limit else min(len(lines), start - 1 + int(limit))
        body = "\n".join(f"{i:6d}  {lines[i - 1]}" for i in range(start, end + 1))
        header = f"{self._rel(p)} (lines {start}-{end} of {len(lines)})\n"
        return _clip(header + body, self.max_output)

    def _t_view_image(self, path: str) -> list[dict]:
        p = self._resolve(path)
        if not self._readable(p):
            raise ToolError(f"{path} is outside the readable area")
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        try:
            part = context.image(p)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        return [context.text(f"{self._rel(p)} ({p.stat().st_size} B)"), part]

    def _t_write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if not self._writable(p):
            raise ToolError(f"{path} is not writable in this phase; allowed: "
                            + ", ".join(self._rel(w) for w in self.writable))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {self._rel(p)} ({len(content)} chars)"

    def _t_edit_file(self, path: str, old_text: str, new_text: str,
                     replace_all: bool = False) -> str:
        p = self._resolve(path)
        if not self._writable(p):
            raise ToolError(f"{path} is not writable in this phase")
        if not p.is_file():
            raise ToolError(f"{path} does not exist")
        text = p.read_text()
        count = text.count(old_text)
        if count == 0:
            raise ToolError("old_text not found; read the file and copy the exact text")
        if count > 1 and not replace_all:
            raise ToolError(f"old_text appears {count} times; add context or set replace_all")
        p.write_text(text.replace(old_text, new_text) if replace_all else
                     text.replace(old_text, new_text, 1))
        return f"edited {self._rel(p)} ({count if replace_all else 1} replacement(s))"

    def _t_list_dir(self, path: str = ".", depth: int = 2) -> str:
        p = self._resolve(path)
        if not self._readable(p) or not p.is_dir():
            raise ToolError(f"{path} is not a readable directory")
        out = []
        base_depth = len(p.parts)
        for item in sorted(p.rglob("*")):
            rel_parts = item.parts[base_depth:]
            if len(rel_parts) > max(1, int(depth)) or any(part.startswith(".") or part == "__pycache__"
                                                          for part in rel_parts):
                continue
            size = f" ({item.stat().st_size} B)" if item.is_file() else "/"
            out.append(self._rel(item) + size)
            if len(out) >= 400:
                out.append("...[truncated]")
                break
        return "\n".join(out) or "(empty)"

    def _t_run_experiment(self, name: str, command: str, gpus: int = 0,
                          timeout_min: float = 60) -> str:
        res = self.gateway.run_experiment(name, command, gpus, timeout_min)
        res.pop("budget", None)
        return _clip(json.dumps(res, indent=1), self.max_output)

    def _t_evaluate_dev(self, candidates: list | None = None) -> str:
        res = self.gateway.evaluate("dev", candidates)
        res.pop("budget", None)
        results = res.get("results") or {}
        for r in results.values():
            keep = max(200, self.max_output // (4 * len(results)))
            r["log_tail"] = (r.get("log_tail") or "")[-keep:]
        text = json.dumps(res, indent=1)
        # still too long: the largest results shrink to their verdict, so that every candidate
        # stays in the output and the output stays valid JSON
        for cid in sorted(results, key=lambda c: -len(json.dumps(results[c]))):
            if len(text) <= self.max_output:
                break
            results[cid] = _verdict(results[cid])
            text = json.dumps(res, indent=1)
        return _clip(text, self.max_output)

    def _t_check_budget(self) -> str:
        return json.dumps(self.gateway.budget(), indent=1)

    def _t_finish(self, summary: str) -> str:
        self.finished = True
        self.finish_summary = summary
        return "phase finished"


class ToolError(Exception):
    pass
