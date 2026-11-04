"""The claude-cli adapter against a fake `claude` executable (no network, no login)."""

from __future__ import annotations

import base64
import json
import stat

import pytest

from rsibench.models import cli, context, load_model
from rsibench.models.context import delta

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
stdin = sys.stdin.read()
log = os.environ["FAKE_CLAUDE_LOG"]
with open(log, "a") as f:
    f.write(json.dumps({"args": args, "stdin": stdin,
                        "env_claudecode": os.environ.get("CLAUDECODE"),
                        "config_dir": os.environ.get("CLAUDE_CONFIG_DIR")}) + "\n")
resume = args[args.index("--resume") + 1] if "--resume" in args else None
session = resume or "sess-1"
count = log + "." + session  # the CLI reports the session's running cost, not the call's
n = int(open(count).read()) + 1 if resume and os.path.exists(count) else 1
open(count, "w").write(str(n))
calls = [{"name": "finish", "arguments": {"summary": "done"}}] if resume else \
        [{"name": "add", "arguments": {"a": 1, "b": 2}}]
print(json.dumps({"type": "system", "subtype": "init", "session_id": session, "tools": []}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session,
                  "result": "", "structured_output": {"message": "ok", "tool_calls": calls},
                  "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3,
                            "cache_creation_input_tokens": 7},
                  "modelUsage": {"claude-opus-5-5": {}}, "total_cost_usd": 0.01 * n, "num_turns": 2}))
'''
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 8).decode()
TOOLS = [{"name": "add", "description": "add", "input_schema": {"type": "object"}},
         {"name": "finish", "description": "end", "input_schema": {"type": "object"}}]


def fake_claude(tmp_path, monkeypatch, effort=None, **params):
    """A claude-cli model over the fake binary, and the log of its calls."""
    fake = tmp_path / "claude"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    model = load_model("claude-cli:claude-opus-5-5", effort=effort,
                       params={"binary": str(fake), "workdir": str(tmp_path / "wd"), **params})
    return model, log


def answer(reply: dict, content) -> list[dict]:
    """A reply (wire form) as the harness sends it back, and a result for its first tool call."""
    return [{"role": "assistant", "text": reply["text"], "tool_calls": reply["tool_calls"],
             "raw": reply["raw"]},
            {"role": "tool_results", "results": [{"tool_call_id": reply["tool_calls"][0]["id"],
                                                  "content": content, "is_error": False}]}]


def test_claude_cli_adapter_sessions_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")  # an enclosing session must not leak into the child
    model, log = fake_claude(tmp_path, monkeypatch, effort="high",
                             config_dir=str(tmp_path / "profile"))
    meta = {"phase": "executor", "task_id": "t", "round": 1, "session": "s0"}
    msgs = [{"role": "user", "content": "go"}]
    r1 = model.complete("SYS", msgs, TOOLS, 1000, meta)
    assert r1.tool_calls[0]["name"] == "add" and r1.stop_reason == "tool_use"
    assert r1.extra["cost_usd"] == 0.01 and r1.cache_write_tokens == 7
    assert r1.model == "claude-opus-5-5"
    msgs += answer(r1.to_wire(), "3")
    r2 = model.complete("SYS", msgs, TOOLS, 1000, meta)
    assert r2.tool_calls[0]["name"] == "finish"
    image = {"type": "image", "mime": "image/png", "data": PNG}
    msgs += answer(r2.to_wire(), [{"type": "text", "text": "plot.png (16 B)"}, image])
    assert model.lowerings("SYS", msgs, TOOLS, meta) == []
    r3 = model.complete("SYS", msgs, TOOLS, 1000, meta)
    # the session's running cost is 0.01, 0.02, 0.03: each resumed call records what it added
    assert [r.extra["cost_usd"] for r in (r2, r3)] == pytest.approx([0.01, 0.01])
    other = model.complete("SYS", [{"role": "user", "content": "go"}], TOOLS, 1000,
                           {**meta, "session": "s1"})
    assert other.extra["cost_usd"] == pytest.approx(0.01)  # a new session starts from zero

    first, second, third = [json.loads(x) for x in log.read_text().splitlines()][:3]
    for call in (first, second, third):
        args = call["args"]
        assert args[args.index("--input-format") + 1] == "stream-json"
        assert args[args.index("--output-format") + 1] == "stream-json" and "--verbose" in args
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--setting-sources") + 1] == "local"
        assert "--strict-mcp-config" in args and args[args.index("--effort") + 1] == "high"
        assert args[args.index("--system-prompt") + 1].startswith("SYS")
        # the harness tools travel in the reply JSON, never as native tool calls of the CLI
        assert "They are not native tool calls" in args[args.index("--system-prompt") + 1]
        assert call["env_claudecode"] is None
        assert call["config_dir"] == str(tmp_path / "profile")
        assert call["stdin"].endswith("\n") and call["stdin"].count("\n") == 1  # one message
    content = [json.loads(c["stdin"])["message"]["content"] for c in (first, second, third)]
    assert json.loads(first["stdin"])["type"] == "user" and "go" not in first["args"]
    assert content[0] == [{"type": "text", "text": "go"}]
    assert "--resume" not in first["args"]
    assert second["args"][second["args"].index("--resume") + 1] == "sess-1"
    assert content[1] == [{"type": "text", "text": '<tool_result id="c1_0">\n3\n</tool_result>'}]
    assert content[2] == [
        {"type": "text", "text": '<tool_result id="c3_0">\nplot.png (16 B)'},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        {"type": "text", "text": "\n</tool_result>"}]


def test_claude_cli_names_what_a_session_cannot_hold(tmp_path):
    model = load_model("claude-cli:claude-opus-5-5", params={"workdir": str(tmp_path / "wd")})
    seeded = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "text": "4"},
              {"role": "user", "content": "3+3?"}]  # a demonstration the CLI never produced
    assert model.lowerings("SYS", seeded, [], {"session": "s"}) == ["history_replayed"]
    assert model.lowerings("SYS", seeded[:2], [], {"session": "s"}) == \
        ["history_replayed", "continue_prompt"]
    meta = {"session": "s"}
    held = {"id": "sess-1", "seen": 2, "digest": context.digest(seeded[:1]), "lowerings": []}
    sessions = {cli.session_label(meta): held}  # holds [user, its reply]
    assert cli.session(sessions, meta, seeded) is held and cli.lowerings(seeded, held) == []
    lowered = {**held, "lowerings": ["continue_prompt"]}  # what the session received earlier
    assert cli.lowerings(seeded, lowered, ("images_first",)) == ["continue_prompt", "images_first"]
    edited = [{"role": "user", "content": "2 + 2?"}, *seeded[1:]]
    written = [*seeded, {"role": "assistant", "text": "6"}, {"role": "user", "content": "4+4?"}]
    # shorter, without the reply, an earlier message changed, a reply the session did not produce
    for other in (seeded[:1], [seeded[0], seeded[2]], edited, written):
        assert cli.session(sessions, meta, other) == cli.NEW
    assert cli.lowerings(edited, cli.NEW) == ["history_replayed"]


def test_a_new_cli_session_gets_the_whole_history(tmp_path, monkeypatch):
    """A history the session does not hold (a few-shot demonstration, an edit, a new session for
    a conversation that has replies) starts a session that receives all of it, the assistant
    turns as text; a call that continues the session sends only its delta."""
    model, log = fake_claude(tmp_path, monkeypatch)
    meta = {"phase": "executor", "task_id": "t", "round": 1, "session": "few-shot"}
    msgs = [{"role": "user", "content": "DEMO"}, {"role": "assistant", "text": "DEMO answer"},
            {"role": "user", "content": "TASK"}]
    reported = []
    for edit in (None, None, "EDITED DEMO"):
        if edit:
            msgs[0] = {"role": "user", "content": edit}  # in place, under the same session
        reported.append(model.lowerings("SYS", msgs, TOOLS, meta))
        msgs += answer(model.complete("SYS", msgs, TOOLS, 1000, meta).to_wire(), "out")
    assert reported == [["history_replayed"]] * 3
    calls = [json.loads(x) for x in log.read_text().splitlines()]
    sent = [json.loads(c["stdin"])["message"]["content"] for c in calls]
    assert all(len(blocks) == 1 for blocks in sent)
    first, second, third = [blocks[0]["text"] for blocks in sent]
    assert first == (f"{context.REPLAY_NOTE}\n\nDEMO\n\n<assistant_turn>\nDEMO answer\n"
                     "</assistant_turn>\n\nTASK")
    assert "--resume" in calls[1]["args"] and second == '<tool_result id="c3_0">\nout\n</tool_result>'
    # the edit starts a new session, which gets the edited history with every reply and call
    assert "--resume" not in calls[2]["args"] and first not in third
    assert third.startswith(f"{context.REPLAY_NOTE}\n\nEDITED DEMO\n\n<assistant_turn>")
    assert '<tool_call id="c3_0" name="add">{"a": 1, "b": 2}</tool_call>' in third
    assert third.count("<assistant_turn>\n") == 3 and third.endswith("out\n</tool_result>")

    meta = {**meta, "session": "images"}
    image = {"type": "image", "mime": "image/png", "data": PNG}
    a, b = context.text("A"), context.text("B")
    msgs = [{"role": "user", "content": [a, image, b, image]}]
    assert model.lowerings("SYS", msgs, TOOLS, meta) == []
    msgs += answer(model.complete("SYS", msgs, TOOLS, 1000, meta).to_wire(), "out")
    msgs[0] = {"role": "user", "content": [b, image, a]}  # reordered in place
    assert model.lowerings("SYS", msgs, TOOLS, meta) == ["history_replayed"]
    model.complete("SYS", msgs, TOOLS, 1000, meta)
    blocks = json.loads(json.loads(log.read_text().splitlines()[-1])["stdin"])["message"]["content"]
    assert [x["type"] for x in blocks] == ["text", "image", "text"]  # the image stays in place
    assert blocks[0]["text"] == f"{context.REPLAY_NOTE}\n\nB"
    assert blocks[2]["text"].startswith("A\n\n<assistant_turn>\nok\n<tool_call")


def test_a_message_in_place_of_the_last_reply_reaches_the_transcript(tmp_path, monkeypatch):
    from conftest import stub_services
    model, _ = fake_claude(tmp_path, monkeypatch)
    services, token, _ = stub_services(tmp_path, model)

    def complete(messages):
        return services.complete(services.authorize(token), {
            "system": "S", "messages": messages, "tools": TOOLS, "session": "s"})

    msgs = [{"role": "user", "content": "GO"}]
    complete(msgs)
    complete([*msgs, {"role": "user", "content": "SUMMARIZE"}])  # where the reply stood
    entries = [json.loads(line) for line in
               (tmp_path / "transcripts" / "executor.jsonl").read_text().splitlines()]
    assert entries[1].get("history_edited") is True
    assert [m["content"] for m in entries[1]["new_messages"]] == ["GO", "SUMMARIZE"]


def test_exact_context_refuses_an_edit_the_cli_session_would_miss(tmp_path, monkeypatch):
    from conftest import stub_services

    from rsibench.orchestrator.gateway import GatewayRefusal
    model, log = fake_claude(tmp_path, monkeypatch)
    services, token, usage = stub_services(tmp_path, model, exact_context=True)

    def complete(messages):
        return services.complete(services.authorize(token), {
            "system": "S", "messages": messages, "tools": TOOLS, "session": "s"})

    msgs = [{"role": "user", "content": "ORIGINAL"}]
    msgs += answer(complete(msgs), "3")
    with pytest.raises(GatewayRefusal) as refused:
        complete([{"role": "user", "content": "EDITED"}, *msgs[1:]])
    assert refused.value.code == "inexact_context" and "history_replayed" in refused.value.message
    complete(msgs)  # the same history, unedited, goes through
    assert [row["lowerings"] for row in usage] == [[], []]
    assert len(log.read_text().splitlines()) == 2  # the refused call never reached the CLI


def test_concurrent_grants_never_continue_each_other_s_cli_session(tmp_path, monkeypatch):
    from conftest import stub_services
    model, log = fake_claude(tmp_path, monkeypatch)
    services, token, _ = stub_services(tmp_path, model)
    phase = services.authorize(token)
    first, second = (services.authorize(services.grant(
        "experiment", phase.task, 1, phase.budget, ("complete",), tmp_path / f"{name}.jsonl"))
        for name in ("e1", "e2"))  # two experiments of one round, the same session name

    def complete(grant, messages):
        return services.complete(grant, {"system": "S", "messages": messages, "tools": TOOLS,
                                         "session": "s"})

    msgs = [{"role": "user", "content": "go"}]
    msgs += answer(complete(first, msgs), "3")
    complete(second, msgs)  # a history that the first experiment's session could continue
    complete(first, msgs)
    resumed = ["--resume" in json.loads(line)["args"] for line in log.read_text().splitlines()]
    assert resumed == [False, False, True]  # the second started its own; the first went on


def test_delta_marks_errors():
    parts = delta([{"role": "tool_results", "results": [
        {"tool_call_id": "x", "content": "boom", "is_error": True}]},
        {"role": "user", "content": "keep going"}])
    assert len(parts) == 1 and parts[0]["type"] == "text"
    assert '<tool_result id="x" (error)>' in parts[0]["text"]
    assert parts[0]["text"].endswith("keep going")


def test_cli_reply_schemas_without_tools_hold_only_the_message():
    from rsibench.models import claude_cli, codex_cli
    for schema in (claude_cli._reply_schema, codex_cli._reply_schema):
        assert set(schema([])["properties"]) == {"message"} and schema([])["required"] == ["message"]
        name = schema([{"name": "act"}])["properties"]["tool_calls"]["items"]["properties"]["name"]
        assert name == {"type": "string", "enum": ["act"]}


def test_claude_cli_call_without_tools_gets_no_tool_protocol(tmp_path, monkeypatch):
    # A solution's own model call has no tools and must not be told to act through tools: a model
    # that follows the protocol puts its answer into tool calls the solution never reads.
    model, _ = fake_claude(tmp_path, monkeypatch)
    for tools, protocol in (([], False), (TOOLS, True)):
        args = model._command("SYS", tools, None)
        system = args[args.index("--system-prompt") + 1]
        schema = json.loads(args[args.index("--json-schema") + 1])
        assert system.startswith("SYS")
        assert ("# Tool use protocol" in system) is protocol
        assert ("tool_calls" in schema["properties"]) is protocol
