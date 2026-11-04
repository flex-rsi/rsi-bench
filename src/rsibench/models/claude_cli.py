"""Claude through the local Claude Code CLI (`claude -p`), using the machine's existing login.

Use this when the machine is signed in to Claude Code (OAuth) and has no API key. The CLI is used
as a plain model transport, isolated from the local Claude Code setup:

    --system-prompt <harness prompt>   replaces Claude Code's own system prompt
    --tools ""                         no built-in tools; tool calls come back as structured output
    --setting-sources local            no user/project settings, hooks, plugins or CLAUDE.md
    --strict-mcp-config                no MCP servers
    cwd = an empty scratch directory   no project CLAUDE.md or auto-memory

Tool calling: the harness tools are described in the system prompt, and `--json-schema` forces
every reply into {"message", "tool_calls": [{"name", "arguments"}]}. A call without tools (a model
call made by a task's solution) gets neither the tool protocol nor "tool_calls": only {"message"}. A neutral conversation maps to
one Claude Code session: the first call starts it, later calls `--resume` it and send only the
new tool results and user content, so the history stays append-only and prompt caching applies. A
call whose history the session does not hold exactly starts a new session that gets the whole
history, its assistant turns as text (cli.turn, lowering `history_replayed`).

Transport: `--input-format stream-json` reads one user message from stdin whose content blocks are
the delta in order, images included (context.to_blocks); `--output-format stream-json --verbose`
prints JSON events, and the last `result` event is the reply. The only lowerings are the ones every
CLI session has (cli.lowerings).

Login profile: `config_dir` (default $RSIB_CLAUDE_CONFIG_DIR from the environment or the .env file)
is passed as CLAUDE_CONFIG_DIR, so a second Claude Code profile can serve the runs; without it the
CLI uses its default profile.

Cost comes from the CLI's own `total_cost_usd`, which sums the whole session: a resumed call
records only what it added (cli.per_call). One small context block that Claude Code always
adds for signed-in users (the account e-mail) cannot be removed; it carries no instructions.
"""

from __future__ import annotations

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

You act only through the tools listed below. They are not native tool calls: you have no tools of
your own, and calling one natively only fails ("No such tool available"). You call a tool by
listing it in "tool_calls" of your reply, and the harness runs it for you. Every listed tool is
available. Every reply is a JSON object with:
- "message": a short note on what you are doing and why (may be empty)
- "tool_calls": the tool calls to run now, in order: [{"name": <tool>, "arguments": {...}}]

The tools run after your reply and their results come back in the next message. To end the
phase, call the `finish` tool. Arguments must follow each tool's JSON schema exactly.

## Tools
"""


def _reply_schema(tools: list[dict]) -> dict:
    if not tools:  # nothing to call: a "tool_calls" field would invite calls to tools that do not exist
        return {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}
    return {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "tool_calls": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": cli.tool_name_schema(tools),
                               "arguments": {"type": "object"}},
                "required": ["name", "arguments"]}},
        },
        "required": ["message", "tool_calls"],
    }


def parse_result(stdout: str) -> dict | None:
    """The last `result` event of a stream-json output."""
    result = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    return result


def _clean_env() -> dict:
    """Drop the variables of an enclosing Claude Code session (so the CLI does not run as a
    nested child session) and API-key variables (so it uses the machine's login)."""
    drop_prefixes = ("CLAUDE_CODE_", "CLAUDE_PLUGIN_")
    drop = {"CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
    return {k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith(drop_prefixes)}


class ClaudeCLIModel(ChatModel):
    adapter = "claude-cli"

    def __init__(self, name: str, effort: str | None = None, params: dict | None = None):
        super().__init__(name, effort, params)
        self.binary = self.params.get("binary") or shutil.which("claude") or "claude"
        self.timeout_s = float(self.params.get("timeout_s", 1800))
        self.retries = int(self.params.get("retries", 3))
        self.max_turns = int(self.params.get("max_turns", 4))
        self.backoff_s = float(self.params.get("backoff_s", 60))
        # a session or usage limit is waited out (cli.through_outages), up to this long in all
        self.outage_wait_s = float(self.params.get("outage_wait_s", 6 * 3600))
        self.limit_poll_s = float(self.params.get("limit_poll_s", 300))
        self.config_dir = self.params.get("config_dir") or localenv.lookup("RSIB_CLAUDE_CONFIG_DIR")
        workdir = self.params.get("workdir")
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rsib-claude-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        # neutral session label -> state of its Claude Code session (cli.session, cli.remember)
        self._sessions: dict[str, dict] = {}
        self._run = subprocess.run

    def _env(self) -> dict:
        env = _clean_env()
        if self.config_dir:
            env["CLAUDE_CONFIG_DIR"] = str(Path(self.config_dir).expanduser())
        return env

    def _command(self, system: str, tools: list[dict], resume: str | None) -> list[str]:
        cmd = [self.binary, "-p", "--model", self.name, "--input-format", "stream-json",
               "--output-format", "stream-json", "--verbose",
               "--system-prompt", system + cli.reply_protocol(TOOL_PROTOCOL, tools),
               "--json-schema", json.dumps(_reply_schema(tools)),
               "--tools", "", "--setting-sources", "local", "--strict-mcp-config",
               "--max-turns", str(self.max_turns)]
        if self.effort:
            cmd += ["--effort", self.effort]
        if resume:
            cmd += ["--resume", resume]
        return cmd

    def _invoke(self, cmd: list[str], stdin: str) -> tuple[dict, tuple | None]:
        """The CLI's result event, retried through outages (cli.through_outages)."""
        def attempt():
            try:
                proc = self._run(cmd, cwd=str(self.workdir), env=self._env(), input=stdin,
                                 capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return None, f"claude CLI timed out after {self.timeout_s:.0f}s"
            data = parse_result(proc.stdout)
            if data is not None and not data.get("is_error"):
                return data, None
            last = (data or {}).get("result") or proc.stderr.strip() or proc.stdout.strip()[-500:]
            return None, f"claude CLI error (exit {proc.returncode}): {str(last)[:500]}"
        return cli.through_outages(self, attempt, "claude-cli")

    def lowerings(self, system, messages, tools, meta=None) -> list[str]:
        return cli.lowerings(messages, cli.session(self._sessions, meta or {}, messages))

    def complete(self, system, messages, tools, max_tokens, meta) -> ModelResponse:
        state = cli.session(self._sessions, meta, messages)
        line = {"type": "user", "message": {"role": "user",
                                            "content": context.to_blocks(cli.turn(messages, state))}}
        data, outage = self._invoke(self._command(system, tools, state["id"]),
                                    json.dumps(line) + "\n")

        structured = data.get("structured_output")
        if not isinstance(structured, dict):
            try:
                structured = json.loads(data.get("result") or "{}")
            except json.JSONDecodeError:
                structured = {"message": str(data.get("result") or ""), "tool_calls": []}
        calls = []
        for i, call in enumerate(structured.get("tool_calls") or []):
            args = call.get("arguments")
            calls.append({"id": f"c{len(messages)}_{i}", "name": str(call.get("name", "")),
                          "arguments": args if isinstance(args, dict) else {}})
        total = data.get("total_cost_usd")
        totals = {} if total is None else {"cost_usd": float(total)}
        cost = cli.per_call(state, data.get("session_id"), totals).get("cost_usd")
        cli.remember(self._sessions, meta, data.get("session_id"), messages,
                     cli.lowerings(messages, state), totals)

        usage = data.get("usage") or {}
        stop = data.get("stop_reason")
        stop_reason = "refusal" if stop == "refusal" else ("tool_use" if calls else "end_turn")
        model_name = next(iter(data.get("modelUsage") or {}), self.name)
        return ModelResponse(
            text=str(structured.get("message") or ""), tool_calls=calls, stop_reason=stop_reason,
            raw={"session_id": data.get("session_id"), "structured": structured},
            model=model_name,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            extra={**({"outage": list(outage)} if outage else {}),
                   "cost_usd": cost, "duration_ms": data.get("duration_ms"),
                   "num_turns": data.get("num_turns")},
        )
