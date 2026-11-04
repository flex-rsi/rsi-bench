"""OpenAI models through the local Codex CLI (`codex exec`), using the machine's existing login.

The counterpart of the claude-cli adapter for machines signed in to Codex (ChatGPT login) without
an API key. The CLI is used as a plain model transport, isolated from the local Codex setup:

    model_instructions_file=<file>     the harness system prompt replaces Codex's own instructions
    --disable shell_tool, ...          no built-in tools (shell, apps, plugins, browser, ...)
    --ignore-user-config/-rules        no ~/.codex/config.toml, no execpolicy rules
    -c include_*_instructions=false    no injected permission, environment or skills context
    -s read-only, cwd = empty dir      nothing to read or write even if a tool slipped through

Tool calling: the harness tools are described in the instructions and `--output-schema` forces
every reply into {"message", "tool_calls": [{"name", "arguments_json"}]}; the arguments travel as
a JSON string because strict structured outputs cannot hold free-form objects. A neutral
conversation maps to one Codex thread: the first call starts it, later calls `codex exec resume`
it with only the new tool results and user content; a call whose history the thread does not hold
exactly starts a new thread that gets the whole history, its assistant turns as text (cli.turn,
lowering `history_replayed`). The text goes through stdin and each image
as an `--image` file, which Codex shows before the text: a delta with images is lowered to
`images_first`, every image in order with the label `[Image #k]` where it stood in the text.

Usage comes from the CLI's `turn.completed` event, which sums the whole thread: a resumed call
records only what it added (cli.per_call). There is no cost report, so the usd budget uses
the OpenAI prices in pricing.toml when the model is listed there. Login profile: `codex_home`
(default $RSIB_CODEX_HOME from the environment or the .env file) is passed as CODEX_HOME. A short
multi-agent notice that Codex always adds cannot be removed; it carries no task instructions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .. import localenv
from . import cli, context
from .base import ChatModel, ModelError, ModelResponse

TOOL_PROTOCOL = """

# Tool use protocol

You act only through the tools listed below. They are not native function calls: you call a tool
by listing it in "tool_calls" of your reply, and the harness runs it for you. Every listed tool is
available. Every reply is a JSON object with:
- "message": a short note on what you are doing and why (may be empty)
- "tool_calls": the tool calls to run now, in order: [{"name": <tool>, "arguments_json": <string>}]
  where "arguments_json" is the tool's arguments as a JSON object encoded in a string,
  for example "{\\"path\\": \\"a.py\\"}"

The tools run after your reply and their results come back in the next message. To end the
phase, call the `finish` tool. Arguments must follow each tool's JSON schema exactly.

## Tools
"""

DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "multi_agent", "apps", "plugins", "recommended_plugins",
    "browser_use", "computer_use", "image_generation", "sleep_tool", "skill_search",
    "tool_suggest", "hooks", "goals", "memories", "code_mode_host",
)
CONFIG = (
    "project_doc_max_bytes=0", 'web_search="disabled"', "include_permissions_instructions=false",
    "include_apps_instructions=false", "include_environment_context=false",
    "include_collaboration_mode_instructions=false", "skills.include_instructions=false",
    "tools.update_plan.enabled=false", "tools.experimental_request_user_input.enabled=false",
)
# the counters of a `turn.completed` event; they sum the whole thread, so a call takes the increase
USAGE = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens",
         "reasoning_output_tokens")


def _reply_schema(tools: list[dict]) -> dict:
    if not tools:  # nothing to call: a "tool_calls" field would invite calls to tools that do not exist
        return {"type": "object", "additionalProperties": False,
                "properties": {"message": {"type": "string"}}, "required": ["message"]}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "message": {"type": "string"},
            "tool_calls": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": cli.tool_name_schema(tools),
                               "arguments_json": {"type": "string"}},
                "required": ["name", "arguments_json"]}},
        },
        "required": ["message", "tool_calls"],
    }


def _clean_env() -> dict:
    """Drop API-key variables (so the CLI uses the machine's login) and an enclosing Claude Code
    session's variables."""
    drop = {"OPENAI_API_KEY", "CODEX_API_KEY", "CLAUDECODE", "CLAUDE_PID"}
    return {k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith(("CLAUDE_CODE_", "CLAUDE_PLUGIN_"))}


def parse_events(stdout: str) -> dict:
    """The JSONL event stream of one `codex exec` call -> thread id, final message, usage, error."""
    out = {"thread_id": None, "text": None, "usage": {}, "error": None}
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "thread.started":
            out["thread_id"] = ev.get("thread_id")
        elif kind == "item.completed" and (ev.get("item") or {}).get("type") == "agent_message":
            out["text"] = ev["item"].get("text")
        elif kind == "turn.completed":
            out["usage"] = ev.get("usage") or {}
        elif kind == "turn.failed":
            out["error"] = str((ev.get("error") or {}).get("message") or ev)
        elif kind == "error":
            out["error"] = str(ev.get("message") or ev)
    return out


class CodexCLIModel(ChatModel):
    adapter = "codex-cli"

    def __init__(self, name: str, effort: str | None = None, params: dict | None = None):
        super().__init__(name, effort, params)
        self.binary = self.params.get("binary") or shutil.which("codex") or "codex"
        self.timeout_s = float(self.params.get("timeout_s", 1800))
        self.retries = int(self.params.get("retries", 3))
        self.backoff_s = float(self.params.get("backoff_s", 60))
        # a usage limit is waited out (cli.through_outages), up to this long in all
        self.outage_wait_s = float(self.params.get("outage_wait_s", 6 * 3600))
        self.limit_poll_s = float(self.params.get("limit_poll_s", 300))
        self.codex_home = self.params.get("codex_home") or localenv.lookup("RSIB_CODEX_HOME")
        # files (instructions, schema, images) live next to the working directory, which stays
        # empty; by default both sit in one private temporary directory
        workdir = self.params.get("workdir")
        if workdir:
            self.workdir = Path(workdir)
            self.files = self.workdir.parent / f"{self.workdir.name}-files"
        else:
            base = Path(tempfile.mkdtemp(prefix="rsib-codex-"))
            self.workdir, self.files = base / "cwd", base / "files"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.files.mkdir(parents=True, exist_ok=True)
        # neutral session label -> state of its Codex thread (cli.session, cli.remember)
        self._sessions: dict[str, dict] = {}
        self._run = subprocess.run

    def _file(self, kind: str, text: str) -> Path:
        path = self.files / f"{kind}-{hashlib.sha1(text.encode()).hexdigest()[:16]}"
        if not path.exists():
            path.write_text(text)
        return path

    def _image(self, part: dict) -> Path:
        name, data = context.image_file(part)
        path = self.files / f"image-{name}"
        if not path.exists():
            path.write_bytes(data)
        return path

    def _env(self) -> dict:
        env = _clean_env()
        if self.codex_home:
            env["CODEX_HOME"] = str(Path(self.codex_home).expanduser())
        return env

    def _command(self, system: str, tools: list[dict], resume: str | None,
                 images: list[Path]) -> list[str]:
        instructions = self._file("instructions.md", system + cli.reply_protocol(TOOL_PROTOCOL, tools))
        schema = self._file("schema.json", json.dumps(_reply_schema(tools)))
        cmd = [self.binary, "exec"] + (["resume"] if resume else [])
        cmd += ["--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
                "-m", self.name, "--output-schema", str(schema)]
        cmd += [f"--image={path}" for path in images]  # "=": the option takes no further values
        cmd += ["-c", f"model_instructions_file={json.dumps(str(instructions))}"]
        for item in CONFIG:
            cmd += ["-c", item]
        for feature in DISABLED_FEATURES:
            cmd += ["--disable", feature]
        if self.effort:
            cmd += ["-c", f"model_reasoning_effort={json.dumps(self.effort)}"]
        if resume:
            return cmd + [resume, "-"]
        return cmd + ["-s", "read-only", "-C", str(self.workdir), "-"]

    def _invoke(self, cmd: list[str], prompt: str) -> tuple[dict, tuple | None]:
        """The CLI's events, retried through outages (cli.through_outages)."""
        def attempt():
            try:
                proc = self._run(cmd, cwd=str(self.workdir), env=self._env(), input=prompt,
                                 capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return None, f"codex CLI timed out after {self.timeout_s:.0f}s"
            events = parse_events(proc.stdout)
            if events["text"] is not None and not events["error"]:
                return events, None
            last = events["error"] or proc.stderr.strip()[-500:] or "no reply"
            return None, f"codex CLI error (exit {proc.returncode}): {last[:500]}"
        return cli.through_outages(self, attempt, "codex-cli")

    def lowerings(self, system, messages, tools, meta=None) -> list[str]:
        state = cli.session(self._sessions, meta or {}, messages)
        images = context.images_first(cli.turn(messages, state))[1]
        return cli.lowerings(messages, state, ("images_first",) if images else ())

    def complete(self, system, messages, tools, max_tokens, meta) -> ModelResponse:
        state = cli.session(self._sessions, meta, messages)
        prompt, images = context.images_first(cli.turn(messages, state))
        cmd = self._command(system, tools, state["id"], [self._image(p) for p in images])
        events, outage = self._invoke(cmd, prompt)

        try:
            structured = json.loads(events["text"] or "{}")
            if not isinstance(structured, dict):
                raise ValueError
        except ValueError:
            structured = {"message": str(events["text"] or ""), "tool_calls": []}
        calls = []
        for i, call in enumerate(structured.get("tool_calls") or []):
            text = call.get("arguments_json") or "{}"
            try:
                args = json.loads(text)
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"__invalid_json__": text}
            calls.append({"id": f"c{len(messages)}_{i}", "name": str(call.get("name", "")),
                          "arguments": args})
        thread = events["thread_id"] or state["id"]
        totals = {k: int(events["usage"].get(k) or 0) for k in USAGE}
        usage = cli.per_call(state, thread, totals)
        cli.remember(self._sessions, meta, thread, messages,
                     cli.lowerings(messages, state, ("images_first",) if images else ()), totals)

        cached = int(usage.get("cached_input_tokens") or 0)
        written = int(usage.get("cache_write_input_tokens") or 0)
        return ModelResponse(
            text=str(structured.get("message") or ""), tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            raw={"thread_id": events["thread_id"], "structured": structured}, model=self.name,
            # Codex reports input_tokens including the cached and cache-write parts
            input_tokens=max(0, int(usage.get("input_tokens") or 0) - cached - written),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=cached, cache_write_tokens=written,
            extra={**({"outage": list(outage)} if outage else {}),
                   "reasoning_output_tokens": usage["reasoning_output_tokens"]},
        )
