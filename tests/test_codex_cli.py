"""The codex-cli adapter against a fake `codex` executable (no network, no login)."""

from __future__ import annotations

import base64
import json
import stat
from pathlib import Path

from rsibench.models import load_model

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
instr = next(a.split("=", 1)[1] for a in args if a.startswith("model_instructions_file="))
with open(os.environ["FAKE_CODEX_LOG"], "a") as f:
    f.write(json.dumps({"args": args, "stdin": prompt, "instructions": open(json.loads(instr)).read(),
                        "openai_key": os.environ.get("OPENAI_API_KEY"),
                        "codex_home": os.environ.get("CODEX_HOME"), "cwd": os.getcwd()}) + "\n")
resume = "resume" in args
count = os.environ["FAKE_CODEX_LOG"] + ".thread-1"  # turn.completed sums the whole thread
n = int(open(count).read()) + 1 if resume and os.path.exists(count) else 1
open(count, "w").write(str(n))
reply = {"message": "ok", "tool_calls": [{"name": "finish", "arguments_json": '{"summary": "done"}'}]
         if resume else [{"name": "add", "arguments_json": '{"a": 1, "b": 2}'},
                         {"name": "add", "arguments_json": "not json"}]}
for ev in ({"type": "thread.started", "thread_id": "thread-1"},
           {"type": "item.completed", "item": {"type": "error", "message": "a config warning"}},
           {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(reply)}},
           {"type": "turn.completed", "usage": {"input_tokens": 100 * n, "cached_input_tokens": 60 * n,
                                                "cache_write_input_tokens": 10 * n,
                                                "output_tokens": 7 * n}}):
    print(json.dumps(ev))
'''


def test_codex_cli_adapter_threads_isolation_and_usage(tmp_path, monkeypatch):
    fake = tmp_path / "codex"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-reach-the-cli")
    model = load_model("codex-cli:gpt-5.6-sol", effort="high",
                       params={"binary": str(fake), "workdir": str(tmp_path / "wd"),
                               "codex_home": str(tmp_path / "codex-home")})
    tools = [{"name": "add", "description": "add", "input_schema": {"type": "object"}},
             {"name": "finish", "description": "end", "input_schema": {"type": "object"}}]
    meta = {"phase": "executor", "task_id": "t", "round": 1, "session": "s0"}
    msgs = [{"role": "user", "content": "go"}]
    r1 = model.complete("SYS", msgs, tools, 1000, meta)
    assert [c["arguments"] for c in r1.tool_calls] == [{"a": 1, "b": 2},
                                                        {"__invalid_json__": "not json"}]
    assert (r1.input_tokens, r1.cache_read_tokens, r1.cache_write_tokens, r1.output_tokens) == \
        (30, 60, 10, 7)
    msgs += [{"role": "assistant", "text": r1.text, "tool_calls": r1.tool_calls, "raw": r1.raw},
             {"role": "tool_results", "results": [{"tool_call_id": r1.tool_calls[0]["id"],
                                                   "content": "3", "is_error": False}]}]
    r2 = model.complete("SYS", msgs, tools, 1000, meta)
    assert r2.tool_calls == [{"id": "c3_0", "name": "finish", "arguments": {"summary": "done"}}]
    # the thread's counters doubled; the resumed call records only its own share
    assert (r2.input_tokens, r2.cache_read_tokens, r2.cache_write_tokens, r2.output_tokens) == \
        (30, 60, 10, 7)
    first, second = [json.loads(x) for x in log.read_text().splitlines()]
    for call in (first, second):
        args = call["args"]
        assert "--ignore-user-config" in args and "--ignore-rules" in args
        assert args[args.index("-m") + 1] == "gpt-5.6-sol"
        assert 'model_reasoning_effort="high"' in args
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "multi_agent"):
            assert args[args.index(feature) - 1] == "--disable"
        assert call["instructions"].startswith("SYS") and "### add" in call["instructions"]
        assert call["openai_key"] is None and call["codex_home"] == str(tmp_path / "codex-home")
        assert Path(call["cwd"]) == tmp_path / "wd" and args[-1] == "-"
    assert first["args"][:2] == ["exec", "--json"]
    assert first["args"][first["args"].index("-s") + 1] == "read-only" and first["stdin"] == "go"
    assert second["args"][:2] == ["exec", "resume"] and second["args"][-2] == "thread-1"
    assert second["stdin"] == '<tool_result id="c1_0">\n3\n</tool_result>'
    assert not any((tmp_path / "wd").iterdir())  # the working directory stays empty


def test_codex_cli_sends_images_as_files_before_the_text(tmp_path, monkeypatch):
    fake = tmp_path / "codex"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    model = load_model("codex-cli:gpt-5.6-sol", params={"binary": str(fake),
                                                         "workdir": str(tmp_path / "wd")})
    tools = [{"name": "add", "description": "add", "input_schema": {"type": "object"}},
             {"name": "finish", "description": "end", "input_schema": {"type": "object"}}]
    meta = {"phase": "executor", "task_id": "t", "round": 1, "session": "s0"}
    msgs = [{"role": "user", "content": "go"}]
    r1 = model.complete("SYS", msgs, tools, 1000, meta)
    png, gif = b"\x89PNG\r\n\x1a\n" + b"\0" * 8, b"GIF89a" + b"\0" * 8
    image = [{"type": "image", "mime": mime, "data": base64.b64encode(data).decode()}
             for mime, data in (("image/png", png), ("image/gif", gif))]
    msgs += [{"role": "assistant", "text": r1.text, "tool_calls": r1.tool_calls, "raw": r1.raw},
             {"role": "tool_results", "results": [
                 {"tool_call_id": "c1_0", "content": [image[0], {"type": "text", "text": "a"}],
                  "is_error": False},
                 {"tool_call_id": "c1_1", "content": [image[1]], "is_error": True}]}]
    assert model.lowerings("SYS", msgs, tools, meta) == ["images_first"]
    assert model.lowerings("SYS", msgs[:1], tools, meta) == []
    r2 = model.complete("SYS", msgs, tools, 1000, meta)

    second = [json.loads(x) for x in log.read_text().splitlines()][1]
    files = [a.split("=", 1)[1] for a in second["args"] if a.startswith("--image=")]
    assert [Path(f).read_bytes() for f in files] == [png, gif]  # in the order of the context
    assert all(Path(f).name.startswith("image-") for f in files)
    assert second["args"][:2] == ["exec", "resume"] and second["args"][-2] == "thread-1"
    assert second["stdin"] == ('<tool_result id="c1_0">\n[Image #1]a\n</tool_result>\n\n'
                               '<tool_result id="c1_1" (error)>\n[Image #2]\n</tool_result>')
    # the thread keeps the images as it received them, so its later calls are lowered too
    msgs += [{"role": "assistant", "text": r2.text, "tool_calls": r2.tool_calls, "raw": r2.raw},
             {"role": "tool_results", "results": [{"tool_call_id": r2.tool_calls[0]["id"],
                                                   "content": "", "is_error": False}]}]
    assert model.lowerings("SYS", msgs, tools, meta) == ["images_first"]
    model.complete("SYS", msgs, tools, 1000, meta)
    third = [json.loads(x) for x in log.read_text().splitlines()][2]
    assert not any(a.startswith("--image=") for a in third["args"])
    call_id = r2.tool_calls[0]["id"]
    assert third["stdin"] == f'<tool_result id="{call_id}">\n\n</tool_result>'
    # a new thread takes its images the same way
    model.complete("SYS", [{"role": "user", "content": [{"type": "text", "text": "see "},
                                                         image[1]]}],
                   tools, 1000, {**meta, "session": "s1"})
    fourth = [json.loads(x) for x in log.read_text().splitlines()][3]
    assert fourth["args"][:2] == ["exec", "--json"] and fourth["stdin"] == "see [Image #1]"
    assert [a for a in fourth["args"] if a.startswith("--image=")] == [f"--image={files[1]}"]


def test_codex_cli_keeps_its_files_in_one_private_temporary_directory(tmp_path, monkeypatch):
    import tempfile
    fake = tmp_path / "codex"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    model = load_model("codex-cli:gpt-5.6-sol", params={"binary": str(fake),
                                                         "codex_home": str(tmp_path / "home")})
    model.complete("SYS", [{"role": "user", "content": "go"}],
                   [{"name": "finish", "description": "end", "input_schema": {"type": "object"}}],
                   1000, {"phase": "executor", "task_id": "t", "round": 1, "session": "s0"})
    base = model.workdir.parent
    assert base.parent == tmp_path / "tmp" and model.files.parent == base
    assert stat.S_IMODE(base.stat().st_mode) == 0o700  # prompts and images are not shared
    assert any(model.files.iterdir()) and not any(model.workdir.iterdir())


def test_codex_cli_call_without_tools_gets_no_tool_protocol(tmp_path, monkeypatch):
    fake = tmp_path / "codex"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    model = load_model("codex-cli:gpt-5.6-sol", params={"binary": str(fake), "workdir": str(tmp_path / "wd"),
                                                       "codex_home": str(tmp_path / "codex-home")})
    tools = [{"name": "add", "description": "add", "input_schema": {"type": "object"}}]
    for tools, protocol in (([], False), (tools, True)):
        args = model._command("SYS", tools, None, [])
        instr = next(a.split("=", 1)[1] for a in args if a.startswith("model_instructions_file="))
        text = Path(json.loads(instr)).read_text()
        schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
        assert text.startswith("SYS")
        assert ("# Tool use protocol" in text) is protocol
        assert ("tool_calls" in schema["properties"]) is protocol


def test_codex_cli_replays_a_conversation_to_a_new_thread(tmp_path, monkeypatch):
    """A task's solution that keeps the conversation but starts a new session on every call
    (hstar-search's chat()) gets the whole history, its earlier replies as text, every image as
    a file in order."""
    from rsibench.models.context import REPLAY_NOTE
    fake = tmp_path / "codex"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    model = load_model("codex-cli:gpt-6.1-sol", params={"binary": str(fake),
                                                         "workdir": str(tmp_path / "wd")})
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 8
    image = {"type": "image", "mime": "image/png", "data": base64.b64encode(png).decode()}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look "}, image]},
            {"role": "assistant", "text": "rotate(-45,0)", "raw": None},
            {"role": "user", "content": [{"type": "text", "text": "now "}, image]}]
    assert model.lowerings("S", msgs, [], {"session": "a"}) == ["history_replayed", "images_first"]
    model.complete("S", msgs, [], 100, {"session": "b"})
    call = json.loads(log.read_text().splitlines()[-1])
    assert call["args"][:2] == ["exec", "--json"]  # a new thread
    assert call["stdin"] == (f"{REPLAY_NOTE}\n\nlook [Image #1]\n\n<assistant_turn>\n"
                             "rotate(-45,0)\n</assistant_turn>\n\nnow [Image #2]")
    files = [a.split("=", 1)[1] for a in call["args"] if a.startswith("--image=")]
    assert [Path(f).read_bytes() for f in files] == [png, png]
