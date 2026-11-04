"""Unit tests for pure components (no runs, no git)."""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsibench.budget import Budget, Usage
from rsibench.contracts import parse_result, public_view
from rsibench.compute import Job
from rsibench.compute.slurm import SlurmBackend
from rsibench import localenv
from rsibench.envs import scrubbed_env
from rsibench.models import context, load_model, parse_spec, resolve_model
from rsibench.models.anthropic_model import to_api_messages as anthropic_messages
from rsibench.models.openai_model import OpenAIModel, to_api_messages as openai_messages
from rsibench.models.pricing import cost_usd
from rsibench.orchestrator.policy import allowed_scopes, classify, default_accept
from rsibench.protocol import parse_budget_overrides
from rsibench.records import validate
from rsibench.metrics import normalize, step_auc


def test_classify_scopes():
    assert classify("workspace/a/solver.py", "a") == "solution"
    assert classify("workspace/b/solver.py", "a") == "frozen"
    assert classify("harness/improver/accept.py") == "I"
    assert classify("harness/executor/system.md") == "E"
    assert classify("harness/core/agent.py") == "E"
    assert classify("records/notes/a.md") == "notes"
    assert classify("records/rounds.jsonl") == "frozen"
    assert classify("tasks/a/task.toml") == "frozen"
    assert classify("src/rsibench/cli.py") == "frozen"


def test_allowed_scopes_by_mode():
    assert allowed_scopes("executor", "full") == {"solution", "notes"}
    assert allowed_scopes("improver", "full") == {"E", "I", "notes"}
    assert allowed_scopes("improver", "fixed-i") == {"E", "notes"}
    assert allowed_scopes("improver", "base") == {"notes"}
    assert allowed_scopes("accept", "full") == set()


def test_default_accept_directions_and_noise():
    assert default_accept(0.9, 0.8, "maximize", 0.0, 2.0, 0.0)[0]
    assert not default_accept(0.8, 0.8, "maximize", 0.0, 2.0, 0.0)[0]
    assert default_accept(0.1, 0.2, "minimize", 0.0, 2.0, 0.0)[0]
    assert not default_accept(0.85, 0.8, "maximize", 0.03, 2.0, 0.0)[0]  # inside 2 sigma
    assert default_accept(0.5, None, "maximize", 0.0, 2.0, 0.0)[0]


def test_step_auc_and_normalize():
    assert step_auc([(0.0, 0.0), (0.5, 1.0)]) == pytest.approx(0.5)
    assert step_auc([(0.5, 1.0)]) == pytest.approx(1.0)  # first value holds before the first point
    assert step_auc([]) is None
    assert normalize(0.5, {"baseline": 1.0, "reference": 0.0}) == pytest.approx(0.5)
    assert normalize(0.5, {"baseline": None, "reference": 0.0}) is None


def test_budget_exhaustion_and_fractions():
    b = Budget({"wall_hours": 10, "gpu_hours": 2, "usd": 1.0, "tokens": 1000, "val_max_calls": 3})
    assert b.exhausted() is None
    b.charge(usd=0.5, input_tokens=100, output_tokens=100)
    assert b.fraction_used() == pytest.approx(0.5)
    b.charge(gpu_hours=2.0)
    assert b.exhausted() == "budget_gpu"
    carried = Budget({"wall_hours": 10, "gpu_hours": 0, "usd": 1.0, "tokens": 1000,
                      "val_max_calls": 3}, carried=Usage(usd=2.0))
    assert carried.exhausted() == "budget_usd"
    unmetered = Budget({"wall_hours": 10, "gpu_hours": 0, "usd": 1.0, "tokens": 1000,
                        "val_max_calls": 3}, usd_metering=False)
    unmetered.charge(usd=5.0)
    assert unmetered.exhausted() is None


def test_budget_overrides_parse():
    assert parse_budget_overrides(["usd=2", "tokens=10", "val_max_calls=4"]) == \
        {"usd": 2.0, "tokens": 10, "val_max_calls": 4}
    with pytest.raises(ValueError):
        parse_budget_overrides(["usd"])


def test_pricing():
    assert cost_usd("anthropic", "claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("anthropic", "claude-opus-5-5", 0, 1_000_000, 1_000_000) == pytest.approx(20.2)
    assert cost_usd("openai", "no-such-model", 10, 10) is None


def test_model_specs():
    assert parse_spec("anthropic:claude-opus-5-5") == ("anthropic", "claude-opus-5-5", {})
    assert parse_spec("openai:Qwen/Qwen3-32B@http://h:8000/v1") == \
        ("openai", "Qwen/Qwen3-32B", {"base_url": "http://h:8000/v1"})
    assert load_model("mock:finish").name == "mock-finish"
    with pytest.raises(ValueError):
        load_model("nope:x")


def test_bare_model_names_resolve_by_access_mode():
    assert resolve_model("claude-opus-5-5") == ("claude-cli:claude-opus-5-5", "cli", {})
    assert resolve_model("claude-opus-5-5", "api") == ("anthropic:claude-opus-5-5", "api", {})
    assert resolve_model("gpt-5.6-sol") == ("codex-cli:gpt-5.6-sol", "cli", {})
    assert resolve_model("gpt-5.6-sol", "api") == ("openai:gpt-5.6-sol", "api", {})
    spec, auth, params = resolve_model("gemini-3-pro", "api")
    assert spec == "openai:gemini-3-pro" and params["api_key_env"] == "GEMINI_API_KEY"
    assert params["base_url"].startswith("https://generativelanguage.googleapis.com/")
    assert resolve_model("Qwen/Qwen3-32B@http://h:8000/v1", "api") == \
        ("openai:Qwen/Qwen3-32B@http://h:8000/v1", "api", {})
    # an explicit endpoint replaces the provider's URL but keeps its key variable
    assert resolve_model("doubao-x@http://proxy/v1", "api")[2] == {"api_key_env": "ARK_API_KEY"}
    # full specs pass through; their adapter implies the access mode
    assert resolve_model("mock:finish") == ("mock:finish", None, {})
    assert resolve_model("anthropic:claude-opus-5-5")[1] == "api"
    for model, auth in (("Qwen/Qwen3-32B", None), ("claude-cli:claude-opus-5-5", "api"),
                        ("claude-opus-5-5@http://x/v1", "api"), ("claude-opus-5-5", "token")):
        with pytest.raises(ValueError):
            resolve_model(model, auth)


def test_localenv_prefers_the_environment_and_stays_out_of_child_envs(tmp_path, monkeypatch):
    env = tmp_path / "keys.env"
    env.write_text("# keys\nexport A_KEY=\"from-file\" # note\nB_KEY=plain # tail\n"
                   "C_KEY='q w'\nnot a line\nEMPTY=\n")
    monkeypatch.setenv("RSIB_ENV_FILE", str(env))
    for name in ("A_KEY", "B_KEY", "C_KEY", "EMPTY"):
        monkeypatch.delenv(name, raising=False)
    assert localenv.lookup("A_KEY") == "from-file"
    assert localenv.lookup("B_KEY") == "plain"
    assert localenv.lookup("C_KEY") == "q w"
    assert localenv.lookup("EMPTY") is None
    assert localenv.where("B_KEY") == str(env)
    monkeypatch.setenv("A_KEY", "from-shell")
    assert localenv.lookup("A_KEY") == "from-shell"
    assert localenv.where("A_KEY") == "environment"
    assert "B_KEY" not in os.environ and "B_KEY" not in scrubbed_env()
    monkeypatch.setenv("RSIB_ENV_FILE", str(tmp_path / "missing.env"))
    assert localenv.lookup("B_KEY") is None


def test_localenv_reads_the_user_file_before_the_repository_file(tmp_path, monkeypatch):
    home, repo = tmp_path / "home", tmp_path / "repo"
    (home / ".config" / "rsib").mkdir(parents=True)
    repo.mkdir()
    (home / ".config" / "rsib" / ".env").write_text("SHARED=user\nONLY_USER=u\n")
    (repo / ".env").write_text("SHARED=repo\nONLY_REPO=r\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("RSIB_REPO", str(repo))
    monkeypatch.delenv("RSIB_ENV_FILE")
    for name in ("SHARED", "ONLY_USER", "ONLY_REPO"):
        monkeypatch.delenv(name, raising=False)
    assert localenv.lookup("SHARED") == "user"
    assert (localenv.lookup("ONLY_USER"), localenv.lookup("ONLY_REPO")) == ("u", "r")
    assert localenv.where("ONLY_REPO") == str(repo / ".env")


def test_api_adapters_take_keys_from_the_dotenv_file(tmp_path, monkeypatch):
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-test\nARK_API_KEY=ark-test\n")
    monkeypatch.setenv("RSIB_ENV_FILE", str(env))
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ARK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    spec, _, params = resolve_model("claude-opus-5-5", "api")
    assert load_model(spec, params=params).client.api_key == "sk-ant-test"
    spec, _, params = resolve_model("doubao-seed-1-6", "api")
    model = load_model(spec, params=params)
    assert model.client.api_key == "ark-test"
    assert str(model.client.base_url).startswith("https://ark.cn-beijing.volces.com/api/v3")


def test_anthropic_conversion_replays_raw_and_merges_user_turns():
    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    tool_use = {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": [{"id": "t1", "name": "bash",
                                                          "arguments": {"command": "ls"}}],
         "raw": [thinking, tool_use]},
        {"role": "tool_results", "results": [{"tool_call_id": "t1", "content": "a.py",
                                              "is_error": False}]},
        {"role": "user", "content": "continue"},
    ]
    api = anthropic_messages(msgs)
    assert [m["role"] for m in api] == ["user", "assistant", "user"]
    assert api[1]["content"] == [thinking, tool_use]  # replayed unchanged
    assert api[2]["content"][0]["type"] == "tool_result"
    assert api[2]["content"][1] == {"type": "text", "text": "continue"}


def test_openai_conversion():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": [{"id": "c1", "name": "bash",
                                                          "arguments": {"command": "ls"}}],
         "raw": None},
        {"role": "tool_results", "results": [{"tool_call_id": "c1", "content": "boom",
                                              "is_error": True}]},
    ]
    api = openai_messages("sys", msgs)
    assert api[0] == {"role": "system", "content": "sys"}
    assert api[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"command": "ls"})
    assert api[3] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: boom"}


PNG = {"type": "image", "mime": "image/png", "data": "iVBORw0KGgo="}
GIF = {"type": "image", "mime": "image/gif", "data": "R0lGODlh"}


def _text(s):
    return {"type": "text", "text": s}


def test_context_delta_keeps_the_text_format_and_the_order_of_images():
    assert context.parts("hi") == [_text("hi")] and context.parts([PNG]) == [PNG]
    text_only = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "text": "", "tool_calls": [], "raw": None},
        {"role": "tool_results", "results": [
            {"tool_call_id": "a", "content": "out", "is_error": False},
            {"tool_call_id": "b", "content": "boom", "is_error": True}]},
        {"role": "user", "content": "keep going"},
    ]
    # the text format of the CLI transports: tagged results, blank-line separated
    assert context.delta(text_only) == [_text(
        '<tool_result id="a">\nout\n</tool_result>\n\n'
        '<tool_result id="b" (error)>\nboom\n</tool_result>\n\nkeep going')]
    assert context.delta(text_only[:1]) == [_text("task")]
    assert context.delta(text_only[:2]) == []  # nothing after the last reply
    for empty in ("", [], [_text("")]):  # an empty result keeps its tags around an empty body
        assert context.delta([{"role": "tool_results", "results": [
            {"tool_call_id": "a", "content": empty, "is_error": False}]}]) == [
            _text('<tool_result id="a">\n\n</tool_result>')]
    with_images = text_only[:2] + [
        {"role": "tool_results", "results": [
            {"tool_call_id": "a", "content": [_text("plot"), PNG, _text(""), GIF],
             "is_error": False}]},
        {"role": "user", "content": [PNG, _text("compare")]}]
    assert context.delta(with_images) == [
        _text('<tool_result id="a">\nplot'), PNG, GIF, _text("\n</tool_result>\n\n"), PNG,
        _text("compare")]
    assert context.images_in(with_images) == [PNG, GIF, PNG]


def test_context_transport_shapes():
    parts = [_text("a"), PNG, _text("b"), GIF]
    assert context.to_blocks(parts) == [
        {"type": "text", "text": "a"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": PNG["data"]}},
        {"type": "text", "text": "b"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/gif",
                                     "data": GIF["data"]}}]
    assert context.to_openai(parts) == [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG["data"]}},
        {"type": "text", "text": "b"},
        {"type": "image_url", "image_url": {"url": "data:image/gif;base64," + GIF["data"]}}]
    assert context.images_first(parts) == ("a[Image #1]b[Image #2]", [PNG, GIF])
    assert context.images_first([_text("only text")]) == ("only text", [])
    name, data = context.image_file(PNG)
    assert data.startswith(b"\x89PNG") and name.endswith(".png") and len(name) == 20
    assert context.sniff(data) == "image/png" and context.sniff(b"BM") is None


def test_anthropic_conversion_puts_images_in_place():
    tool_use = {"type": "tool_use", "id": "t1", "name": "view_image", "input": {"path": "p.png"}}
    api = anthropic_messages([
        {"role": "user", "content": [_text("look"), PNG]},
        {"role": "assistant", "text": "", "tool_calls": [], "raw": [tool_use]},
        {"role": "tool_results", "results": [
            {"tool_call_id": "t1", "content": [_text("p.png (9 B)"), PNG], "is_error": False},
            {"tool_call_id": "t2", "content": [], "is_error": False},
            {"tool_call_id": "t3", "content": [_text("")], "is_error": False},
            {"tool_call_id": "t4", "content": [_text("p"), _text("ng")], "is_error": False}]},
    ])
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": PNG["data"]}}
    assert api[0] == {"role": "user", "content": [{"type": "text", "text": "look"}, block]}
    assert api[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "is_error": False,
                                    "content": [{"type": "text", "text": "p.png (9 B)"}, block]}
    assert [r["content"] for r in api[2]["content"][1:]] == ["(no output)", "(no output)", "png"]


def test_openai_conversion_lowers_only_tool_results_with_images():
    call = {"role": "assistant", "text": "", "raw": None, "tool_calls": [
        {"id": "c1", "name": "bash", "arguments": {}}, {"id": "c2", "name": "view_image",
                                                         "arguments": {"path": "p.png"}}]}
    text_only = [{"role": "user", "content": [_text("go "), _text("on")]}, call,
                 {"role": "tool_results", "results": [
                     {"tool_call_id": "c1", "content": "ok", "is_error": False},
                     {"tool_call_id": "c2", "content": [_text("p"), _text("ng")],
                      "is_error": True}]}]
    api = openai_messages("sys", text_only)
    assert api[1] == {"role": "user", "content": "go on"}
    assert api[3:] == [{"role": "tool", "tool_call_id": "c1", "content": "ok"},
                       {"role": "tool", "tool_call_id": "c2", "content": "ERROR: png"}]
    model = OpenAIModel.__new__(OpenAIModel)  # lowerings() needs no SDK client
    assert model.lowerings("sys", text_only, []) == []

    empty = [*text_only[:2], {"role": "tool_results", "results": [
        {"tool_call_id": "c1", "content": "", "is_error": False},
        {"tool_call_id": "c2", "content": [], "is_error": True}]}]
    assert [m["content"] for m in openai_messages("sys", empty)[3:]] == \
        ["(no output)", "ERROR: (no output)"]

    with_image = [{"role": "user", "content": [_text("see"), PNG]}, call,
                  {"role": "tool_results", "results": [
                      {"tool_call_id": "c1", "content": "", "is_error": False},
                      {"tool_call_id": "c2", "content": [_text("p.png"), PNG],
                       "is_error": False}]},
                  {"role": "user", "content": "next"}]
    api = openai_messages("sys", with_image)
    url = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG["data"]}}
    assert api[1] == {"role": "user", "content": [{"type": "text", "text": "see"}, url]}
    assert api[3:5] == [{"role": "tool", "tool_call_id": c, "content": "(output in the next user "
                         "message)"} for c in ("c1", "c2")]
    assert api[5] == {"role": "user", "content": [
        {"type": "text", "text": '<tool_result id="c1">\n\n</tool_result>\n\n'
                                 '<tool_result id="c2">\np.png'},
        url, {"type": "text", "text": "\n</tool_result>"}]}
    assert api[6] == {"role": "user", "content": "next"}
    assert model.lowerings("sys", with_image, []) == ["tool_results_as_user_message"]
    assert model.lowerings("sys", with_image[:1], []) == []  # images in user messages are native


def test_message_schema():
    valid = [
        {"role": "user", "content": "plain text, as old harnesses send it"},
        {"role": "assistant", "text": "", "tool_calls": [{"id": "c", "name": "bash",
                                                          "arguments": {}}], "raw": [{"x": 1}]},
        {"role": "tool_results", "results": [{"tool_call_id": "c", "content": [_text("a"), PNG],
                                              "is_error": False}]},
        {"role": "user", "content": [GIF, _text("b")]},
    ]
    assert validate("messages", valid) == []
    for bad, where in (({**PNG, "mime": "image/bmp"}, "0/content/0/mime"),
                       ({"type": "image", "mime": "image/png"}, "'data' is a required property"),
                       ({"type": "text", "text": 5}, "0/content/0/text"),
                       ({**PNG, "file": "p.png"}, "'file' was unexpected"),
                       ({"type": "video", "data": "x"}, "0/content/0/type")):
        errors = validate("messages", [{"role": "user", "content": [bad]}])
        assert errors and where in errors[0], (bad, errors)
    assert validate("messages", [{"role": "system", "content": "x"}])
    assert validate("messages", [{"role": "tool_results", "results": [
        {"tool_call_id": "c", "content": None}]}])
    huge = [{"role": "user", "content": [{**PNG, "data": "A" * 7_000_001}]}]
    error, = validate("messages", huge, limit=3)
    assert len(error) < 400 and error.endswith("is too long")  # the image is not echoed


def test_harness_context_builds_parts_from_image_files(tmp_path):
    import base64
    import importlib.util

    from conftest import REPO
    spec = importlib.util.spec_from_file_location("hctx", REPO / "harness" / "core" / "context.py")
    hctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hctx)  # standard library only: loads without the harness package
    mislabelled = tmp_path / "plot.jpg"
    mislabelled.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(8))
    part = hctx.image(mislabelled)
    assert part["mime"] == "image/png"  # from the bytes, not the extension
    assert base64.b64decode(part["data"]) == mislabelled.read_bytes()
    assert validate("messages", [{"role": "user", "content": [part]}]) == []
    assert hctx.MAX_IMAGE_BYTES == 5 * 2 ** 20
    largest = tmp_path / "largest.png"  # the largest image fits the gateway's limit
    largest.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(hctx.MAX_IMAGE_BYTES - 8))
    assert validate("messages", [{"role": "user", "content": [hctx.image(largest)]}]) == []
    for name, data, reason in (("notes.txt", b"x", "not an image file"),
                               ("fake.png", b"text", "not a valid"),
                               ("big.png", largest.read_bytes() + b"\0", "5 MiB")):
        (tmp_path / name).write_bytes(data)
        with pytest.raises(ValueError, match=reason):
            hctx.image(tmp_path / name)
    assert hctx.extend("a", [hctx.text("b")]) == [_text("a"), _text("b")]
    assert hctx.extend([part], []) == [part]


def test_exact_context_is_a_protocol_override(tmp_path):
    from rsibench.cli import _protocol_overrides, build_parser
    from rsibench.protocol import load_protocol
    args = build_parser().parse_args(["run", "new", "--task", "t", "--model", "m",
                                      "--exact-context"])
    assert _protocol_overrides(args) == {"exact_context": True}
    args = build_parser().parse_args(["run", "new", "--task", "t", "--model", "m"])
    assert _protocol_overrides(args) == {}
    assert load_protocol()["run"]["exact_context"] is False
    older = tmp_path / "old.toml"  # the protocol file of a release without the key
    older.write_text("[run]\nmax_rounds = 3\n[tiers.XS]\nusd = 1.0\n")
    assert load_protocol(older)["run"] == {"max_rounds": 3}  # as frozen by that release


def test_slurm_script_and_polling(tmp_path):
    calls = []

    class Proc:
        def __init__(self, out="", code=0):
            self.stdout, self.stderr, self.returncode = out, "", code

    states = iter(["", "RUNNING|0:0|10", "COMPLETED|0:0|120"])

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "sbatch":
            return Proc("12345")
        if cmd[0] == "sacct":
            return Proc(next(states))
        return Proc()

    backend = SlurmBackend({"account": "acct", "partition": "p1,p2", "poll_s": 0,
                            "container_image": "/img.sqsh", "container_mounts": ["/a:/a"]},
                           runner=fake_run)
    job = Job(name="exp", command="python train.py --lr 1e-3", cwd=tmp_path,
              env={"RSIB_X": "1", "PATH": "/usr/bin", "RSIB_PYTHON": "/host/venv/bin/python"},
              gpus=2, timeout_s=3600, log_path=tmp_path / "job.log")
    script = backend.script(job)
    assert "#SBATCH --account=acct" in script and "#SBATCH --partition=p1,p2" in script
    assert "#SBATCH --gpus-per-node=2" in script and "#SBATCH --export=NONE" in script
    assert "srun --container-image=/img.sqsh" in script and "export RSIB_X=1" in script
    assert "export PYTHONNOUSERSITE=1" in script
    assert "export PATH=" not in script  # container jobs keep the image's own PATH
    assert "export RSIB_PYTHON=python" in script  # the default image's interpreter, too
    bare = SlurmBackend({"account": "acct"}, runner=fake_run).script(job)
    assert "export RSIB_PYTHON=/host/venv/bin/python" in bare and "export PATH=" in bare
    old = tmp_path / "job.sbatch"  # what an earlier job at this path left
    old.write_text("old")
    old.chmod(0o644)
    res = backend.run(job)
    assert res.status == "ok" and res.job_id == "12345"
    assert res.gpu_hours == pytest.approx(2 * 120 / 3600)
    assert "account" not in backend.describe()
    # the script exports the job's variables, its gateway token too: only its owner may read it
    assert old.read_text() == script and old.stat().st_mode & 0o777 == 0o600


def test_slurm_jobs_cancelled_before_they_start_leave_the_queue(tmp_path):
    calls, over = [], threading.Event()

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        if cmd[0] == "sacct":
            over.set()  # cancelled while it is still pending
        return SimpleNamespace(stdout="77" if cmd[0] == "sbatch" else "", stderr="", returncode=0)

    backend = SlurmBackend({"poll_s": 0}, runner=fake_run)
    job = Job(name="dev", command="true", cwd=tmp_path, env={}, gpus=1, cancelled=over.is_set,
              log_path=tmp_path / "job.log")
    res = backend.run(job)
    assert (res.status, res.job_id, res.gpu_hours) == ("cancelled", "77", 0.0)
    assert calls == ["sbatch", "sacct", "scancel"]
    assert backend.run(job).status == "cancelled" and len(calls) == 3  # not even submitted


def test_record_schema_rejects_bad_round():
    errors = validate("round", {"task_id": "x", "round": -1})
    assert errors


def test_results_table_groups_and_gains():
    from rsibench.results import build_table, table_csv, table_markdown

    def row(run, mode, seed, norm, src=None, at="2026-10-02T00:00:00Z", test=0.5):
        return {"run_id": run, "sandboxed": True, "mode": mode, "model": "m", "effort": "high", "seed": seed,
                "base": "harness-v0.2.0", "improver_source": src, "indexed_at": at,
                "tasks": [{"task_id": "t", "test": test, "normalized": {"test": norm},
                           "best_val": 0.5, "val_auc": 0.4, "test_auc": None, "rounds": 3,
                           "accepted": 1, "cost": {"usd": 1.0, "wall_s": 3600},
                           "ended_reason": "budget_usd"}]}

    rows = [row("b1", "base", 1, 0.2), row("b2", "base", 2, 0.4),
            row("f1", "fixed-i", 1, 0.5), row("f2", "fixed-i", 2, 0.7),
            row("p1", "fixed-i", 1, 0.9, src={"ref": "run/x/final", "commit": "c"}),
            row("f1", "fixed-i", 1, 0.1, at="2026-10-01T00:00:00Z", test=None)]  # older, unfinished
    t = build_table(rows)
    by = {e["condition"]: e for e in t["groups"]}
    assert by["base"]["normalized_test"]["mean"] == pytest.approx(0.3)
    assert by["fixed-i"]["normalized_test"]["mean"] == pytest.approx(0.6)  # stale f1 row ignored
    assert by["probe:run/x/final"]["runs"] == 1
    gains = {g["comparison"]: g["gain"] for g in t["gains"]}
    assert gains["fixed-i - base"] == pytest.approx(0.3)
    assert gains["probe:run/x/final - fixed-i"] == pytest.approx(0.3)
    assert "fixed-i - base" in table_markdown(t) and table_csv(t).count("\n") == 4


def test_result_rows_are_validated_and_grouped_by_protocol(capsys):
    from rsibench.results import build_table, valid_rows

    def row(run, proto, norm):
        return {"schema_version": "1.0", "run_id": run, "sandboxed": True, "ref": f"run/{run}", "commit": "a" * 40,
                "indexed_at": "2026-10-02T00:00:00Z", "mode": "base", "model": "m", "effort": None,
                "seed": 1, "base": "harness-v0.3.0", "improver_source": None, "protocol_id": proto,
                "tasks": [{"task_id": "t", "tier": "S", "direction": "maximize", "rounds": 2,
                           "accepted": 1, "rejected": 1, "baseline_val": 0.1, "best_val": 0.5,
                           "first_round_gain": 0.4, "later_gain": 0.0, "test": 0.5,
                           "normalized": {"test": norm}, "val_auc": 0.3, "test_auc": None,
                           "ended_reason": "budget_usd", "cost": {"usd": 1.0, "wall_s": 60},
                           "harness_commits": {"E": 0, "I": 0}, "violations": 0}]}

    bad = row("broken", "0" * 12, 0.5)
    del bad["tasks"]
    rows = valid_rows([row("a", "1" * 12, 0.2), row("b", "2" * 12, 0.8), bad])
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert "skipping invalid result row broken" in capsys.readouterr().err
    groups = build_table(rows)["groups"]
    assert len(groups) == 2  # different budgets/protocols are never averaged together


class _Sent(Exception):
    """Raised by a stub transport once it has the request, so a test reads what would be sent."""


_FINISH = {"name": "finish", "description": "end", "input_schema": {"type": "object",
                                                                   "properties": {}}}


def test_openai_requests_leave_out_an_empty_tool_list(monkeypatch):
    pytest.importorskip("openai")
    from rsibench.models.openai_model import OpenAIModel
    m = OpenAIModel("m", params={"base_url": "http://127.0.0.1:9/v1"})
    sent = []

    def capture(kwargs):
        sent.append(kwargs)
        raise _Sent

    monkeypatch.setattr(m, "_create_through_outages", capture)
    for tools in ([], [_FINISH]):  # vLLM and OpenAI answer 400 to "tools": []
        with pytest.raises(_Sent):
            m.complete("sys", [{"role": "user", "content": "hi"}], tools, 16, {})
    assert "tools" not in sent[0] and sent[1]["tools"][0]["function"]["name"] == "finish"


def test_anthropic_requests_leave_out_an_empty_tool_list(monkeypatch):
    pytest.importorskip("anthropic")
    from rsibench.models.anthropic_model import AnthropicModel
    monkeypatch.setenv("RSIB_TEST_ANTHROPIC_KEY", "k")
    m = AnthropicModel("m", params={"api_key_env": "RSIB_TEST_ANTHROPIC_KEY"})
    sent = []

    def capture(**kwargs):
        sent.append(kwargs)
        raise _Sent

    monkeypatch.setattr(m.client.messages, "stream", capture)
    for tools in ([], [_FINISH]):
        with pytest.raises(_Sent):
            m.complete("sys", [{"role": "user", "content": "hi"}], tools, 16, {})
    assert "tools" not in sent[0] and sent[1]["tools"][0]["name"] == "finish"


def test_a_limit_names_when_it_resets():
    import datetime as dt
    from rsibench.models.cli import is_limit, wait_hint
    now = dt.datetime(2026, 10, 4, 20, 0, tzinfo=dt.timezone.utc)
    msg = "claude CLI error (exit 1): You've hit your session limit · resets 9:20pm"
    assert is_limit(msg) and wait_hint(msg, now) == 80 * 60
    assert wait_hint("usage limit, resets 7pm", now) == 23 * 3600  # already past: tomorrow
    assert wait_hint("limit reached; resets 12:05am", now) == 4 * 3600 + 5 * 60
    assert wait_hint("overloaded", now) is None and not is_limit("overloaded")
    # an unusable login is waited out like a limit (no reset time: polled)
    held = ("claude CLI error (exit 1): Your account is on hold and can't use Claude Code. "
            "View details or appeal: https://claude.ai/restricted")
    assert is_limit(held) and wait_hint(held, now) is None
    assert is_limit("Not logged in · Please run /login") and not is_limit("bad request")
    # a weekly limit, its reset read in the time zone the message names (22:01 in Chicago)
    late = dt.datetime(2026, 10, 8, 3, 1, tzinfo=dt.timezone.utc)
    weekly = "You've hit your weekly limit · resets 10pm (America/Chicago)"
    assert is_limit(weekly) and wait_hint(weekly, late) == 24 * 3600 - 60  # 22:00 passed: tomorrow
    assert wait_hint("resets 11:30pm (America/Chicago)", late) == 89 * 60
    assert wait_hint("hit your weekly limit · resets Oct 10, 10pm (America/Chicago)", late) \
        == 72 * 3600 - 60


def test_cli_calls_wait_out_a_usage_limit(monkeypatch):
    from rsibench.models import cli
    from rsibench.models.base import ChatModel, ModelError

    class Fake(ChatModel):
        retries, backoff_s, outage_wait_s, limit_poll_s = 1, 2, 10_000, 300

        def complete(self, *a):
            raise NotImplementedError

    slept = []
    monkeypatch.setattr(cli.time, "sleep", slept.append)
    monkeypatch.setattr(cli, "wait_hint", lambda message, now=None: 1000)
    model = Fake("m")
    replies = iter([(None, "You've hit your session limit · resets 9:20pm")] * 2 + [("ok", None)])
    result, outage = cli.through_outages(model, lambda: next(replies), "test")
    assert result == "ok" and slept == [1060, 1060]  # until the reset, plus a minute, twice
    assert outage and outage[1] >= outage[0] and model.outages == 1 and not model.blocked_until()
    # an account on hold is polled every limit_poll_s until a usable login answers
    slept.clear()
    monkeypatch.setattr(cli, "wait_hint", lambda message, now=None: None)
    replies = iter([(None, "Your account is on hold and can't use Claude Code.")] * 2
                   + [("ok", None)])
    assert cli.through_outages(model, lambda: next(replies), "test")[0] == "ok"
    assert slept == [300, 300]
    # a transient error gets `retries` backoffs; an unknown error fails at once
    replies = iter([(None, "overloaded")] * 3)
    with pytest.raises(ModelError, match="overloaded"):
        cli.through_outages(model, lambda: next(replies), "test")
    with pytest.raises(ModelError, match="bad request"):
        cli.through_outages(model, lambda: (None, "bad request"), "test")
    # a limit that outlasts outage_wait_s fails the call
    model.outage_wait_s = 500
    with pytest.raises(ModelError, match="outage_wait_s"):
        cli.through_outages(model, lambda: (None, "usage limit"), "test")


def test_openai_outage_wait_retries_connection_errors(monkeypatch):
    openai = pytest.importorskip("openai")
    from rsibench.models.openai_model import OpenAIModel

    m = OpenAIModel("m", params={"base_url": "http://127.0.0.1:9/v1", "outage_wait_s": 5,
                                 "outage_backoff_s": 0, "max_retries": 0})
    calls = []

    def flaky(**kw):
        calls.append(kw)
        if len(calls) < 3:
            raise openai.APIConnectionError(request=None)
        return "ok"

    monkeypatch.setattr(m.client.chat.completions, "create", flaky)
    resp, outage = m._create_through_outages({"model": "m"})
    assert resp == "ok" and len(calls) == 3 and outage and outage[1] >= outage[0]
    calls.clear()
    calls.extend([{}, {}])  # the next call succeeds at once: no outage reported
    assert m._create_through_outages({"model": "m"}) == ("ok", None)
    m.params["outage_wait_s"] = 0
    calls.clear()
    with pytest.raises(openai.APIConnectionError):
        m._create_through_outages({"model": "m"})


def test_gateway_refuses_calls_beyond_the_phase_cap(tmp_path):
    from types import SimpleNamespace

    from rsibench.orchestrator.gateway import GatewayRefusal, Services

    budget = SimpleNamespace(exhausted=lambda: None)
    svc = Services(SimpleNamespace(protocol={"max_model_calls_per_phase": 2, "submit_grace_calls": 3}))
    grant = svc.authorize(svc.grant("executor", SimpleNamespace(id="t"), 1, budget, ("complete",),
                                    tmp_path / "executor.jsonl"))
    for _ in range(2):  # within the cap the request gets past the check (then fails on the empty payload)
        with pytest.raises(GatewayRefusal) as info:
            svc.complete(grant, {})
        assert info.value.code == "bad_request"
    with pytest.raises(GatewayRefusal) as info:
        svc.complete(grant, {})
    assert info.value.status == 402 and info.value.code == "phase_call_limit"


def test_the_sandbox_keeps_the_symlinks_on_the_way_to_the_interpreter(tmp_path):
    from rsibench.sandbox import links_on_the_way
    real = tmp_path / "cpython-3.11.15" / "bin"
    real.mkdir(parents=True)
    (real / "python3.11").write_text("")
    (tmp_path / "cpython-3.11").symlink_to(tmp_path / "cpython-3.11.15")  # as uv links versions
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(tmp_path / "cpython-3.11" / "bin" / "python3.11")
    (venv / "python3").symlink_to("python")
    assert links_on_the_way(str(venv / "python3")) == {
        str(venv / "python3"), str(venv / "python"), str(tmp_path / "cpython-3.11")}



def test_the_sandbox_allows_a_linked_parent_only_on_the_way_to_the_interpreter(tmp_path):
    from conftest import linked_project
    from rsibench.sandbox import interpreter_paths
    alias_venv, venv, base = linked_project(tmp_path)
    paths = interpreter_paths(str(alias_venv / "bin" / "python"), str(alias_venv), str(base))
    assert str(tmp_path / "fsw") not in paths  # would bind the whole project tree
    assert {str(alias_venv), str(venv), str(base)} <= set(paths)
    assert not any(p in (str(tmp_path / "fs12" / "projects"), str(tmp_path / "fs12" / "projects" / "me"))
                   for p in paths)


def test_slurm_sends_cpu_only_jobs_to_the_cpu_partition(tmp_path):
    site = {"account": "a", "partition": "gpu1,gpu2", "cpu_partition": "cpu_short",
            "cpu_cpus": 16, "cpu_mem_gb": 64, "cpu_max_minutes": 240, "cpus_per_gpu": 8}
    cpu = SlurmBackend(site).script(Job(name="feat", command="python f.py", cwd=tmp_path, env={},
                                        timeout_s=6 * 3600))
    assert "--partition=cpu_short" in cpu and "--cpus-per-task=16" in cpu and "--mem=64G" in cpu
    assert "--time=04:00:00" in cpu and "--gpus" not in cpu  # clamped to the partition's limit
    gpu = SlurmBackend(site).script(Job(name="eval", command="true", cwd=tmp_path, env={}, gpus=1,
                                        timeout_s=3600))
    assert "--partition=gpu1,gpu2" in gpu and "--cpus-per-task=8" in gpu and "--time=01:00:00" in gpu
    plain = SlurmBackend({"partition": "gpu1"}).script(Job(name="f", command="true", cwd=tmp_path,
                                                            env={}))
    assert "--partition=gpu1" in plain  # without a cpu_partition nothing changes


def test_phase_limits_pin_renice_and_cap_threads():
    import subprocess

    from rsibench.phase_limits import PhaseLimits
    assert PhaseLimits({}).prefix() == [] and PhaseLimits(None).env() == {}
    limits = PhaseLimits({"phase": {"cpus": 1, "nice": 5, "threads": 3}})
    assert limits.env()["OMP_NUM_THREADS"] == "3" and limits.env()["MKL_NUM_THREADS"] == "3"
    probe = "import os; print(len(os.sched_getaffinity(0)), os.nice(0), os.environ['OMP_NUM_THREADS'])"
    if len(os.sched_getaffinity(0)) > 1:
        out = subprocess.run(limits.prefix() + [sys.executable, "-c", probe], text=True,
                             capture_output=True, env={**os.environ, **limits.env()}).stdout.split()
        assert out[0] == "1" and int(out[1]) >= 5 and out[2] == "3"
        assert limits.describe() == {"phase_cpus": 1, "phase_threads": 3}


def test_slurm_script_uses_the_job_mounts_of_a_sandboxed_job(tmp_path):
    backend = SlurmBackend({"account": "a", "container_mounts": ["/lustre:/lustre"],
                            "container_flags": ["--no-container-remap-root"]})
    job = Job(name="exp", command="python train.py", cwd=tmp_path, env={"HOME": "/runs/x/home"},
              gpus=1, container="/img.sqsh", mounts=["/runs/x/ws:/runs/x/ws", "/data/dev:/data/dev:ro"])
    script = backend.script(job)
    assert "--container-mounts=/runs/x/ws:/runs/x/ws,/data/dev:/data/dev:ro" in script
    assert "/lustre:/lustre" not in script and "--no-container-mount-home" in script
    assert "export HOME=/runs/x/home; python train.py" in script
    plain = backend.script(Job(name="e", command="true", cwd=tmp_path, env={}, container="/img.sqsh"))
    assert "--container-mounts=/lustre:/lustre" in plain and "--no-container-mount-home" not in plain


def test_result_contract_accepts_plain_and_structured_result_files(tmp_path):
    fields, problems = parse_result({"mse": 0.5, "n": 3}, tmp_path)
    assert fields == {"metrics": {"mse": 0.5, "n": 3}, "artifacts": [], "info": {}} and not problems
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "0.png").write_bytes(b"png")
    raw = {"mse": 0.5, "metrics": {"per_class": {"a": 1.0}}, "info": {"note": "x"},
           "artifacts": [{"path": "frames/0.png", "role": "failure"},
                         {"path": "../outside.txt"}, {"path": "missing.txt"}]}
    fields, problems = parse_result(raw, tmp_path)
    assert fields["metrics"] == {"mse": 0.5, "per_class": {"a": 1.0}}
    assert fields["artifacts"] == [{"path": str((tmp_path / "frames/0.png").resolve()),
                                    "role": "failure", "mime": "image/png"}]
    assert fields["info"] == {"note": "x"} and len(problems) == 2
    for bad in ({"artifacts": "x"}, {"metrics": [1]}, [1, 2]):
        fields, problems = parse_result(bad, tmp_path)
        assert problems and fields["metrics"] == {}
    view = public_view({"status": "ok", "score": 0.5, "metrics": raw["metrics"] | {"mse": 0.5},
                        "artifacts": fields["artifacts"], "info": {"x": 1}})
    assert view == {"status": "ok", "score": 0.5, "metrics": {"mse": 0.5}}


def test_a_task_lists_the_public_metrics_of_val_and_test(tmp_path):
    import shutil

    from conftest import FIXTURE_TASKS
    from rsibench.tasks import TaskError, load_task_dir
    root = tmp_path / "toy-regression"
    shutil.copytree(FIXTURE_TASKS / "toy-regression", root,
                    ignore=shutil.ignore_patterns("__pycache__"))
    plain = load_task_dir(root)
    before = plain.version_hash()
    assert plain.public_metrics is None
    toml = root / "task.toml"
    toml.write_text(toml.read_text().replace("secondary = [", 'public = ["n"]\nsecondary = ['))
    task = load_task_dir(root)
    assert task.public_metrics == ["mse", "n"]  # the primary metric is always public
    assert task.version_hash() != before        # part of the task definition
    result = {"status": "ok", "score": 0.5,
              "metrics": {"mse": 0.5, "max_abs_err": 0.9, "n": 3, "per_half": {"left": 0.4}}}
    assert public_view(result, task.public_metrics)["metrics"] == {"mse": 0.5, "n": 3}
    assert public_view(result)["metrics"] == {"mse": 0.5, "max_abs_err": 0.9, "n": 3}
    toml.write_text(toml.read_text().replace('public = ["n"]', 'public = "n"'))
    with pytest.raises(TaskError, match="metric.public must be a list of metric names"):
        load_task_dir(root)


def test_result_problems_of_experiment_files_and_hidden_splits(tmp_path):
    raw = {"loss": 0.5, "info": "lr=0.1", "metrics": [0.9, 0.7], "artifacts": [{"path": "S.png"}]}
    fields, problems = parse_result(raw, tmp_path, strict=False)  # an experiment's file
    assert fields == {"metrics": {"loss": 0.5, "info": "lr=0.1", "metrics": [0.9, 0.7]},
                      "artifacts": [], "info": {}}  # malformed reserved keys stay plain metrics
    assert len(problems) == 3 and any("'lr=0.1' is not of type 'object'" in p for p in problems)
    fields, problems = parse_result(raw, tmp_path, hidden=True)  # a val or test reward.json
    assert fields["metrics"] == {}
    assert sorted(problems) == ['info: breaks type "object"', 'metrics: breaks type "object"']
    for bad, problem in (({"loss": 0.5, "artifacts": [{"path": "S.png", "note": 7}]},
                          'artifacts/0/note: breaks type "string"'),
                         ({"loss": 0.5, "artifacts": [{"path": "S.png", "extra": "S"}]},
                          "artifacts/0: breaks additionalProperties false"),
                         ({"loss": 0.5, "artifacts": [{"path": "S.png"}]},
                          "artifacts/0: is not a file inside the output directory")):
        assert parse_result(bad, tmp_path, hidden=True)[1] == [problem]  # no value is quoted


def test_an_evaluator_run_is_charged_whatever_reward_json_it_leaves(tmp_path):
    from rsibench.compute.base import JobResult
    from rsibench.orchestrator.evaluator import run_eval
    task = SimpleNamespace(id="t", primary_metric="mse", eval_gpus=1, eval_timeout_s=60.0,
                           evaluator=tmp_path / "test.sh", root=tmp_path,
                           split_dir=lambda split, root=None: tmp_path / split)

    def backend(reward: bytes, status: str):
        def run(job):
            (job.cwd / "reward.json").write_bytes(reward)
            return JobResult(status, 0 if status == "ok" else None, 36.0, 0.01)
        return SimpleNamespace(name="fake", run=run)

    for reward, status, detail in ((b'{"mse": 0.', "ok", "reward.json is not valid JSON"),
                                   (b"\xff", "ok", "reward.json is not valid JSON"),
                                   (b'{"mse": 0.', "timeout", "evaluator timeout (exit None)"),
                                   (b'{"mse": 0.5}', "ok", "")):
        for split in ("dev", "val"):
            result = run_eval(task, tmp_path, split, tmp_path / "out" / split,
                              backend(reward, status))
            assert (result["status"], result["detail"]) == ("error" if detail else "ok", detail)
            assert result["gpu_hours"] == 0.01


def test_local_jobs_end_with_the_processes_they_leave_running(tmp_path):
    from conftest import running
    from rsibench.compute.local import LocalBackend
    pid_file = tmp_path / "pid"
    job = Job(name="bg", command=f"sleep 60 > /dev/null 2>&1 & echo $! > {pid_file}",
              cwd=tmp_path, env={"PATH": os.environ["PATH"]}, timeout_s=30)
    assert LocalBackend({}).run(job).status == "ok"
    assert not running(int(pid_file.read_text()))


def test_run_constants_take_their_default_when_a_release_lacks_them():
    from rsibench.protocol import RUN_DEFAULTS, load_protocol, run_constant
    run = load_protocol()["run"]
    assert set(RUN_DEFAULTS) <= set(run)  # this release states them all
    assert run_constant({}, "max_eval_candidates") == 8
    assert run_constant({}, "submit_min_gain") == 0.0  # an older release submits on any dev gain
    assert run_constant({}, "unfinished_submit") == "workspace"  # ... and a cut-off workspace
    assert run_constant({"max_eval_candidates": 2}, "max_eval_candidates") == 2
    with pytest.raises(KeyError):
        run_constant({"max_rounds": 3}, "max_rounds")  # only keys that have a default


def _clocked_jobs(tmp_path, label: str, gpus: int, n: int = 2) -> list[Job]:
    """Jobs that log their CUDA_VISIBLE_DEVICES, then the times they start and end."""
    clock = f"{shlex.quote(sys.executable)} -c 'import time; print(time.time())'"
    return [Job(name=f"{label}{i}", cwd=tmp_path, env={"PATH": os.environ["PATH"]}, gpus=gpus,
                command=f"echo gpus=$CUDA_VISIBLE_DEVICES; {clock}; sleep 0.5; {clock}",
                timeout_s=30, log_path=tmp_path / f"{label}{i}.log") for i in range(n)]


def _run_together(backend, jobs) -> tuple[list, list]:
    """Results of jobs started at once, and the (devices, start, end) each one logged."""
    with ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(backend.run, jobs))
    assert [r.status for r in results] == ["ok"] * len(jobs)
    logged = []
    for job in jobs:
        devices, start, end = job.log_path.read_text().split()
        logged.append((devices.removeprefix("gpus="), float(start), float(end)))
    return results, logged


def _overlap(logged) -> bool:
    (_, start0, end0), (_, start1, end1) = logged
    return min(end0, end1) > max(start0, start1)


def test_local_backend_leases_gpus_and_bounds_the_jobs_running_at_once(tmp_path):
    from rsibench.compute.local import LocalBackend
    _, logged = _run_together(LocalBackend({"gpus": "0,1"}), _clocked_jobs(tmp_path, "two", 1))
    assert sorted(d for d, _, _ in logged) == ["0", "1"] and _overlap(logged)
    results, logged = _run_together(LocalBackend({"gpus": "3"}), _clocked_jobs(tmp_path, "one", 1))
    assert [d for d, _, _ in logged] == ["3", "3"] and not _overlap(logged)
    walls = sorted(r.wall_s for r in results)  # the job that waited is charged its own run only
    assert walls[1] < 1.5 * walls[0]
    assert all(r.gpu_hours == pytest.approx(r.wall_s / 3600) for r in results)
    _, logged = _run_together(LocalBackend({"max_parallel": 1}), _clocked_jobs(tmp_path, "cpu", 0))
    assert [d for d, _, _ in logged] == ["", ""] and not _overlap(logged)
    _, logged = _run_together(LocalBackend({}), _clocked_jobs(tmp_path, "default", 0))
    assert _overlap(logged)  # up to 4 at once by default


def test_local_backend_gives_devices_back_after_timeouts_and_errors(tmp_path):
    from rsibench.compute.local import LocalBackend
    backend = LocalBackend({"gpus": "0", "max_parallel": 1})
    env = {"PATH": os.environ["PATH"]}
    jobs = [Job(name="slow", command="sleep 30", cwd=tmp_path, env=env, gpus=1, timeout_s=0.3),
            Job(name="error", command="true", cwd=tmp_path / "missing", env=env, gpus=1),
            Job(name="big", command="true", cwd=tmp_path, env=env, gpus=2),
            Job(name="quick", command="echo $CUDA_VISIBLE_DEVICES", cwd=tmp_path, env=env,
                gpus=1, log_path=tmp_path / "quick.log")]
    statuses = []
    worker = threading.Thread(target=lambda: statuses.extend(backend.run(j).status for j in jobs),
                              daemon=True)
    worker.start()
    worker.join(30)
    assert statuses == ["timeout", "error", "rejected", "ok"]  # a kept device would block "quick"
    assert (tmp_path / "quick.log").read_text() == "0\n"


def test_local_backend_never_starts_a_job_cancelled_while_it_waits(tmp_path):
    from rsibench.compute.local import LocalBackend
    backend = LocalBackend({"max_parallel": 1})
    env = {"PATH": os.environ["PATH"]}
    over = threading.Event()
    first = Job(name="first", command="touch first; sleep 1", cwd=tmp_path, env=env)
    queued = Job(name="queued", command="touch queued", cwd=tmp_path, env=env,
                 cancelled=over.is_set)
    with ThreadPoolExecutor(2) as pool:
        running = pool.submit(backend.run, first)
        while not (tmp_path / "first").exists():
            time.sleep(0.01)
        waiting = pool.submit(backend.run, queued)
        time.sleep(0.2)  # queued waits for the slot that first holds
        over.set()
        assert (running.result().status, waiting.result().status) == ("ok", "cancelled")
    assert not (tmp_path / "queued").exists()
    queued.cancelled = lambda: False  # the slot came back
    assert backend.run(queued).status == "ok" and (tmp_path / "queued").exists()


def test_gateway_refuses_malformed_candidates(tmp_path):
    from conftest import eval_services
    from rsibench.orchestrator.gateway import GatewayRefusal
    services, token, rows, _ = eval_services(tmp_path, lambda snapshot: {"score": 1.0})
    grant, wt = services.authorize(token), services.engine.wt
    for name in ("a", "b"):
        (wt / ".rsib" / "scratch" / name).mkdir()
    (tmp_path / "outside").mkdir()
    (wt / ".rsib" / "scratch" / "link").symlink_to(tmp_path / "outside")
    (wt / ".rsib" / "scratch" / "loop").symlink_to(wt / ".rsib" / "scratch" / "loop")
    (wt / "workspace" / "t" / "solver.py").write_text("")
    a, b = ({"id": c, "path": f".rsib/scratch/{c}"} for c in "ab")
    for bad in ([{"id": "a", "path": str(tmp_path / "outside")}],  # outside workspace and scratch
                [{"id": "a", "path": ".rsib/scratch/link"}],  # resolves outside them
                [{"id": "a", "path": ".rsib/scratch/loop"}],  # resolves nowhere
                [{"id": "a", "path": ".rsib/scratch/loop/a"}],
                [{"id": "a", "path": "workspace"}],  # around the task's workspace
                [{"id": "a", "path": "workspace/t/solver.py"}],  # not a directory
                [a, {**b, "id": "a"}], [a, {**b, "id": "A"}],  # ids repeat, also ignoring case
                [{"id": f"c{i}", "path": a["path"]} for i in range(9)],  # over the default limit
                [{**a, "id": "-a"}], [{**a, "id": "x" * 33}], [{"id": "a"}], [{**a, "x": 1}], [],
                "a"):
        with pytest.raises(GatewayRefusal) as refused:
            services.evaluate(grant, {"split": "dev", "candidates": bad})
        assert (refused.value.status, refused.value.code) == (400, "bad_request"), bad
    assert rows == []
    services.engine.protocol["max_eval_candidates"] = 1  # the run's own value, where it has one
    with pytest.raises(GatewayRefusal, match="1 to 1"):
        services.evaluate(grant, {"split": "dev", "candidates": [a, b]})
    reply = services.evaluate(grant, {"split": "dev", "candidates": [
        {"id": "ws", "path": str(wt / "workspace" / "t")}]})
    assert list(reply["results"]) == ["ws"] and rows[-1]["exp_id"] == "t-r001-dev001-ws"
    reply = services.evaluate(grant, {"split": "dev"})  # a single evaluation's reply
    assert set(reply) == {"status", "score", "metric", "direction", "metrics", "artifacts", "info",
                          "detail", "log_tail", "artifact_dir", "budget"}
    assert rows[-1]["exp_id"] == "t-r001-dev002" and rows[-1]["name"] == "evaluate dev"


def test_gateway_scores_candidates_concurrently_and_submits_the_best(tmp_path):
    from conftest import eval_services
    from rsibench.orchestrator.gateway import GatewayRefusal

    def evaluate(snapshot):
        time.sleep(0.5)
        return {"score": float((snapshot / "score.txt").read_text()), "gpu_hours": 0.25}

    services, token, rows, budget = eval_services(tmp_path, evaluate, dev_bar=0.5)
    grant, scratch = services.authorize(token), services.engine.scratch_dir()
    for cid, score in (("a", "0.7"), ("b", "0.2"), ("c", "0.2")):  # b and c beat the bar equally
        (scratch / cid).mkdir()
        (scratch / cid / "score.txt").write_text(score)
    start = time.monotonic()
    reply = services.evaluate(grant, {"split": "dev", "candidates": [
        {"id": c, "path": f".rsib/scratch/{c}"} for c in "abc"]})
    assert time.monotonic() - start < 1.0  # three evaluations of 0.5 s each, at once
    assert budget.used().gpu_hours == pytest.approx(0.75)  # each one charged
    assert [r["exp_id"] for r in rows] == [f"t-r001-dev001-{c}" for c in "abc"]
    assert [r["name"] for r in rows] == [f"evaluate dev {c}" for c in "abc"]
    for row in rows:  # each scored on its own snapshot, kept in its own output directory
        snapshot = Path(row["artifact_dir"]) / "snapshot"
        assert float((snapshot / "score.txt").read_text()) == row["score"] and row["snapshot_sha"]
    assert list(reply["results"]) == ["a", "b", "c"]
    assert [r["score"] for r in reply["results"].values()] == [0.7, 0.2, 0.2]
    assert set(reply["results"]["a"]) == {"status", "score", "metric", "direction", "metrics",
                                          "artifacts", "info", "detail", "log_tail",
                                          "artifact_dir"}
    assert reply["submit_required"] and "candidate b" in reply["message"]  # the first of equals
    assert grant.submit == {"snapshot": str(Path(rows[1]["artifact_dir"]) / "snapshot"),
                            "score": 0.2, "bar": 0.5, "exp_id": "t-r001-dev001-b"}
    with pytest.raises(GatewayRefusal) as refused:
        services.evaluate(grant, {"split": "dev"})
    assert (refused.value.status, refused.value.code) == (409, "submit_required")


def test_the_submit_line_asks_for_a_share_of_the_anchor_span():
    from rsibench.orchestrator.gateway import auto_submit_at, best_improvement, submit_margin
    task = SimpleNamespace(baseline=0.2, reference=0.8, direction="maximize",
                           better=lambda a, b: a > b)
    assert submit_margin(task, {"submit_min_gain": 0.05}) == pytest.approx(0.03)
    assert submit_margin(task, {}) == 0.0  # a release without the constant: any gain submits
    assert submit_margin(SimpleNamespace(baseline=None, reference=None),
                         {"submit_min_gain": 0.05}) == 0.0  # a task without anchors
    assert auto_submit_at(task, 0.4, 0.03) == pytest.approx(0.43)
    assert auto_submit_at(SimpleNamespace(direction="minimize"), 0.4, 0.03) == pytest.approx(0.37)
    assert auto_submit_at(task, None, 0.03) is None
    rows = [{"status": "ok", "score": s} for s in (0.42, 0.45, 0.44)]
    assert best_improvement(task, rows, 0.4, 0.03) == 1
    assert best_improvement(task, rows[:1], 0.4, 0.03) is None  # beats the bar, not the line
    assert best_improvement(task, rows[:1], 0.4) == 0  # no margin: any gain


def test_a_dev_gain_below_the_submit_line_leaves_the_round_running(tmp_path):
    from conftest import eval_services
    scores = iter([0.48, 0.3])  # mse: bar 0.5, line 0.5 - 0.1
    services, token, rows, _ = eval_services(tmp_path, lambda snapshot: {"score": next(scores)},
                                             dev_bar=0.5, dev_margin=0.1)
    grant = services.authorize(token)
    reply = services.evaluate(grant, {"split": "dev"})
    assert "submit_required" not in reply and grant.submit is None
    assert reply["submit_rule"] == {"accepted_dev": 0.5, "auto_submit_at": pytest.approx(0.4)}
    reply = services.evaluate(grant, {"split": "dev"})  # not refused: nothing was submitted
    assert reply["submit_required"] and "submit_rule" not in reply
    assert grant.submit["score"] == 0.3 and grant.submit["bar"] == 0.5


def test_a_candidate_that_cannot_be_scored_leaves_the_others_their_results(tmp_path):
    from conftest import eval_services
    services, token, rows, budget = eval_services(
        tmp_path, lambda snapshot: {"score": 0.1, "gpu_hours": 0.25}, dev_bar=0.5)
    grant, scratch = services.authorize(token), services.engine.scratch_dir()
    for cid in ("bad", "good"):
        (scratch / cid).mkdir()
    (scratch / "bad" / "big.ckpt").symlink_to(tmp_path / "missing.ckpt")  # cannot be copied
    reply = services.evaluate(grant, {"split": "dev", "candidates": [
        {"id": c, "path": f".rsib/scratch/{c}"} for c in ("bad", "good")]})
    bad, good = reply["results"]["bad"], reply["results"]["good"]
    assert (bad["status"], bad["score"]) == ("error", None) and "not scored" in bad["detail"]
    assert set(bad) == set(good) and good["score"] == 0.1
    assert [(r["exp_id"], r["status"], r["gpu_hours"]) for r in rows] == [
        ("t-r001-dev001-bad", "error", 0.0), ("t-r001-dev001-good", "ok", 0.25)]
    assert [validate("experiment", r) for r in rows] == [[], []]
    assert budget.used().gpu_hours == pytest.approx(0.25)
    assert reply["submit_required"] and grant.submit["exp_id"] == "t-r001-dev001-good"


def test_gateway_jobs_are_cancelled_when_their_phase_ends(tmp_path):
    from conftest import eval_services
    services, token, _, _ = eval_services(tmp_path, lambda snapshot: {"score": 1.0})
    stub, seen = services.engine.run_split_eval, []

    def run_split_eval(task, snapshot, split, out_dir, rnd, budget=None, cancelled=None):
        seen.append((threading.current_thread().daemon, cancelled))
        return stub(task, snapshot, split, out_dir, rnd)

    services.engine.run_split_eval = run_split_eval
    for cid in "ab":
        (services.engine.scratch_dir() / cid).mkdir()
    services.evaluate(services.authorize(token), {"split": "dev", "candidates": [
        {"id": c, "path": f".rsib/scratch/{c}"} for c in "ab"]})
    assert [daemon for daemon, _ in seen] == [True, True]  # they never hold up an exit
    assert not any(cancelled() for _, cancelled in seen)
    services.revoke(token)  # the phase ends: what has not started never starts
    assert all(cancelled() for _, cancelled in seen)


def _post(url: str, token: str | None, body: dict | None = None) -> tuple[int, dict]:
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_gateway_serves_each_token_its_own_grant(tmp_path):
    from conftest import stub_services
    from rsibench.models.mock import FinishModel
    from rsibench.orchestrator.gateway import Gateway
    services, token, usage = stub_services(tmp_path, FinishModel())
    executor = services.authorize(token)
    improver = services.grant("improver", executor.task, 1, executor.budget,
                              ("complete", "budget"), tmp_path / "improver.jsonl")
    gateway = Gateway(services).start()
    refused = {"error": "unauthorized", "message": "unknown or revoked token"}
    try:
        assert _post(f"{gateway.url}/v1/budget", token)[0] == 200
        for bad in (None, "", "not-a-token", token[:-1], improver + token):
            assert _post(f"{gateway.url}/v1/budget", bad) == (401, refused)
        assert _post(f"{gateway.url}/v1/nowhere", "not-a-token")[0] == 401  # before routing
        assert _post(f"{gateway.url}/v1/nowhere", improver)[0] == 404
        for path in ("experiment", "evaluate"):  # outside the improver's grant
            status, body = _post(f"{gateway.url}/v1/{path}", improver, {"command": "true"})
            assert (status, body["error"]) == (403, "forbidden")
        assert _post(f"{gateway.url}/v1/budget", improver)[0] == 200
        assert services.revoke(improver).revoked and services.revoke(improver) is None
        assert _post(f"{gateway.url}/v1/budget", improver) == (401, refused)
        assert _post(f"{gateway.url}/v1/budget", token)[0] == 200  # other grants live on
    finally:
        gateway.stop()
    assert usage == []


def test_concurrent_grants_keep_their_own_call_counters_and_sessions(tmp_path):
    from conftest import stub_services
    from rsibench.models import ChatModel, ModelResponse, cli
    from rsibench.orchestrator.gateway import GatewayRefusal
    metas = []

    class Slow(ChatModel):
        adapter = "fake"

        def complete(self, system, messages, tools, max_tokens, meta):
            metas.append(dict(meta))
            time.sleep(0.05)
            return ModelResponse(text="ok", tool_calls=[], stop_reason="end_turn", raw=None,
                                 model="fake")

    services, token, usage = stub_services(tmp_path, Slow("fake"), max_model_calls_per_phase=3)
    phase = services.authorize(token)
    # two experiments of the same round: alike in all but their grant
    experiments = [services.authorize(services.grant(
        "experiment", phase.task, 1, phase.budget, ("complete",),
        tmp_path / name / "transcripts" / "experiment.jsonl")) for name in ("e1", "e2")]
    grants = [phase, *experiments]
    payload = {"system": "s", "messages": [{"role": "user", "content": "x"}], "tools": [],
               "session": "0"}
    with ThreadPoolExecutor(9) as pool:  # three calls of each grant at once
        list(pool.map(lambda grant: services.complete(grant, payload), grants * 3))
    for grant in grants:  # each one reached its own cap of 3, not a shared one
        with pytest.raises(GatewayRefusal, match=f"the {grant.consumer} grant reached"):
            services.complete(grant, payload)
    assert sorted(row["phase"] for row in usage) == ["executor"] * 3 + ["experiment"] * 6
    assert len({m["grant"] for m in metas}) == 3
    assert len({cli.session_label(m) for m in metas}) == 3  # one CLI session per grant
    for grant in grants:  # and a transcript of its own, whose lines follow one another
        entries = [json.loads(line) for line in grant.transcript.read_text().splitlines()]
        assert ["system" in entry for entry in entries] == [True, False, False]


def test_model_access_is_part_of_the_task_definition(tmp_path):
    import shutil

    from conftest import FIXTURE_TASKS
    from rsibench.tasks import TaskError, load_task_dir
    root = tmp_path / "toy-regression"
    shutil.copytree(FIXTURE_TASKS / "toy-regression", root,
                    ignore=shutil.ignore_patterns("__pycache__"))
    plain = load_task_dir(root)
    before = plain.version_hash()
    assert plain.model_access is False
    toml = root / "task.toml"
    text = toml.read_text()
    toml.write_text(text.replace("eval_gpus = 0", "eval_gpus = 0\nmodel_access = true"))
    task = load_task_dir(root)
    assert task.model_access is True and task.version_hash() != before
    for bad in ('"yes"', "1"):
        toml.write_text(text.replace("eval_gpus = 0", f"eval_gpus = 0\nmodel_access = {bad}"))
        with pytest.raises(TaskError, match="resources.model_access must be true or false"):
            load_task_dir(root)


def test_a_task_can_set_the_model_call_cap_of_its_evaluations(tmp_path):
    import shutil

    from conftest import FIXTURE_TASKS
    from rsibench.tasks import TaskError, load_task_dir
    root = tmp_path / "toy-regression"
    shutil.copytree(FIXTURE_TASKS / "toy-regression", root,
                    ignore=shutil.ignore_patterns("__pycache__"))
    assert load_task_dir(root).eval_model_calls is None  # the protocol's cap applies
    toml = root / "task.toml"
    text = toml.read_text()
    toml.write_text(text.replace("eval_gpus = 0", "eval_gpus = 0\nmodel_access = true\n"
                                                  "eval_model_calls_max = 500"))
    assert load_task_dir(root).eval_model_calls == 500
    for extra, problem in (("model_access = true\neval_model_calls_max = 0", "a positive integer"),
                           ("model_access = true\neval_model_calls_max = true", "a positive integer"),
                           ("eval_model_calls_max = 500", "needs resources.model_access = true")):
        toml.write_text(text.replace("eval_gpus = 0", f"eval_gpus = 0\n{extra}"))
        with pytest.raises(TaskError, match=problem):
            load_task_dir(root)


def test_an_evaluation_gets_the_call_cap_of_its_task(tmp_path, monkeypatch):
    from rsibench.orchestrator import engine as engine_mod
    from rsibench.orchestrator.gateway import GatewayRefusal, Services

    seen = {}

    def run_eval(task, workspace, split, out_dir, backend, **kw):
        grant = services.authorize(kw["gateway"][1])
        seen["cap"] = grant.call_cap
        for _ in range(2):  # within the cap a call gets past the check, then fails on its payload
            with pytest.raises(GatewayRefusal, match="missing 'system'"):
                services.complete(grant, {})
        with pytest.raises(GatewayRefusal) as refused:
            services.complete(grant, {})
        seen["third"] = refused.value
        return {"status": "ok"}

    monkeypatch.setattr(engine_mod, "run_eval", run_eval)
    services = Services(SimpleNamespace(protocol={"max_model_calls_per_phase": 80,
                                                  "submit_grace_calls": 3}))
    engine = SimpleNamespace(backend=None, cpu_backend=None, data_root=None, sandbox=None,
                             gateway=SimpleNamespace(url="http://gateway"), services=services,
                             container_for=lambda task, backend: None,
                             live_budget=lambda task_id: None)
    task = SimpleNamespace(id="t", eval_gpus=0, model_access=True, eval_model_calls=2)
    # one evaluation's grant (run_split_eval adds waiting for the model and retries around it)
    engine_mod.RunEngine._split_eval_once(engine, task, tmp_path, "val", tmp_path / "out", 3,
                                          None, None, None)
    assert seen["cap"] == 2
    assert (seen["third"].status, seen["third"].code) == (402, "phase_call_limit")
    assert "the task's limit of 2 model calls" in seen["third"].message
    assert services._grants == {}  # revoked when the evaluation ended


def test_tool_arguments_follow_array_and_object_schemas(tmp_path, monkeypatch):
    from conftest import REPO
    monkeypatch.syspath_prepend(str(REPO))
    from harness.core.tools import ToolBox
    sent = []
    gateway = SimpleNamespace(evaluate=lambda split, candidates=None: sent.append(candidates)
                              or {"results": {"a": {"score": 1.0, "log_tail": "x" * 5000}}})
    toolbox = ToolBox(root=tmp_path, writable=[".rsib/scratch"], gateway=gateway, cwd=tmp_path,
                      experiments=True, max_output_chars=4000)
    for args, problem in (({"candidates": "a"}, "'candidates' must be of type array"),
                          ({"candidates": ["a"]}, "'candidates[0]' must be of type object"),
                          ({"candidates": [{"id": "a"}]}, "'candidates[0]' is missing 'path'"),
                          ({"candidates": [{"id": 1, "path": "p"}]},
                           "'candidates[0].id' must be of type string"),
                          ({"candidates": [{"id": "a", "path": "p", "x": 1}]}, "unknown key 'x'")):
        content, is_error = toolbox.call("evaluate_dev", args)
        assert is_error and problem in content, content
    assert sent == []
    candidates = [{"id": "a", "path": ".rsib/scratch/a"}]
    content, is_error = toolbox.call("evaluate_dev", {"candidates": candidates})
    assert not is_error and sent == [candidates]
    assert json.loads(content)["results"]["a"]["log_tail"] == "x" * 1000  # fits the output
    content, is_error = toolbox.call("write_file", {"path": ".rsib/scratch/a/s.py", "content": ""})
    assert not is_error and (tmp_path / ".rsib" / "scratch" / "a" / "s.py").exists()


def test_evaluate_dev_keeps_every_candidate_in_its_output(tmp_path, monkeypatch):
    from conftest import REPO
    monkeypatch.syspath_prepend(str(REPO))
    from harness.core.tools import ToolBox

    def result(i: int) -> dict:  # per-class metrics and failure images, more for later ones
        return {"status": "ok", "score": i / 10, "metric": "miou", "direction": "maximize",
                "metrics": {f"iou_class_{k}": 0.5 for k in range(20 * i)},
                "artifacts": [{"path": f"/artifacts/c{i}/fail_{k}.png"} for k in range(2 * i)],
                "info": {}, "detail": "", "log_tail": "x" * 3000, "artifact_dir": f"/a/c{i}"}

    results = {f"c{i}": result(i) for i in range(8)}
    gateway = SimpleNamespace(evaluate=lambda split, candidates=None: {"results": results})
    toolbox = ToolBox(root=tmp_path, writable=[], gateway=gateway, cwd=tmp_path, experiments=True)
    content, is_error = toolbox.call("evaluate_dev", {"candidates": [{"id": "c0", "path": "p"}]})
    assert not is_error and len(content) <= 16000
    out = json.loads(content)["results"]  # valid JSON, every candidate with its score
    assert [(cid, r["score"]) for cid, r in out.items()] == [(f"c{i}", i / 10) for i in range(8)]
    shrunk = [cid for cid, r in out.items() if "omitted" in r]
    assert shrunk and shrunk == [f"c{i}" for i in range(8 - len(shrunk), 8)]  # the largest
    assert all(out[cid]["artifact_dir"] == f"/a/{cid}" for cid in shrunk)
    assert len(out["c1"]["metrics"]) == 20  # the others stay whole


def test_budget_leaves_waits_out_of_wall_time_and_counts_overlaps_once():
    now = time.monotonic()
    b = Budget({"wall_hours": 1, "gpu_hours": 1, "usd": 1, "tokens": 10, "val_max_calls": 1})
    b._start = now - 100
    b.add_wait(now - 90, now - 60)
    b.add_wait(now - 70, now - 50)    # overlaps the first: 40 s in total
    b.add_wait(now - 200, now - 95)   # before the budget started: only 5 s count
    b.add_wait(now - 10, now - 20)    # empty
    used = b.used()
    assert used.queue_s == pytest.approx(45, abs=1)
    assert used.wall_s == pytest.approx(55, abs=1)
    assert Budget({"wall_hours": 1, "gpu_hours": 1, "usd": 1, "tokens": 10, "val_max_calls": 1},
                  carried=Usage(wall_s=10, queue_s=3)).used().queue_s == pytest.approx(3)


def test_slurm_job_result_carries_its_queue_time(tmp_path):
    class Proc:
        def __init__(self, out="", code=0):
            self.stdout, self.stderr, self.returncode = out, "", code

    states = iter(["PENDING|0:0|0", "RUNNING|0:0|5", "COMPLETED|0:0|30"])

    def fake_run(cmd, **kw):
        if cmd[0] == "sbatch":
            return Proc("777")
        if cmd[0] == "sacct" and "Submit,Start" in cmd:
            return Proc("2026-10-03T01:00:00|2026-10-03T01:02:30")
        if cmd[0] == "sacct":
            return Proc(next(states))
        return Proc()

    backend = SlurmBackend({"account": "a", "poll_s": 0}, runner=fake_run)
    res = backend.run(Job(name="e", command="true", cwd=tmp_path, env={}, gpus=1,
                          log_path=tmp_path / "e.log"))
    assert res.status == "ok" and res.queue_s == pytest.approx(150)
    assert res.gpu_hours == pytest.approx(30 / 3600)


def test_a_wait_still_going_on_counts_until_it_ends():
    b = Budget({"wall_hours": 1, "gpu_hours": 1, "usd": 1, "tokens": 10, "val_max_calls": 1})
    b._start = time.monotonic() - 100
    with pytest.raises(RuntimeError):
        with b.waiting() as on_queue:
            on_queue(time.monotonic() - 30)          # pending for 30 s so far
            assert b.used().queue_s == pytest.approx(30, abs=1)
            raise RuntimeError("the job never returned")
    # a wait still open when the block ends is closed there, and the wall clock runs again
    assert not b._live and b.used().queue_s == pytest.approx(30, abs=1)
    with b.waiting() as on_queue:
        on_queue(time.monotonic() - 60, time.monotonic() - 45)   # a wait that has ended
    assert b.used().queue_s == pytest.approx(45, abs=1)


def test_a_pending_slurm_job_is_a_wait_while_it_is_pending(tmp_path):
    """The deadline of a phase leaves queue time out: it must see the wait during the wait, not
    only once the job is over (a phase whose job was still pending at its limit was killed)."""
    class Proc:
        def __init__(self, out="", code=0):
            self.stdout, self.stderr, self.returncode = out, "", code

    budget = Budget({"wall_hours": 1, "gpu_hours": 1, "usd": 1, "tokens": 10, "val_max_calls": 1})
    states = iter(["PENDING|0:0|0", "PENDING|0:0|0", "RUNNING|0:0|5", "COMPLETED|0:0|30"])
    seen = []

    def fake_run(cmd, **kw):
        if cmd[0] == "sbatch":
            return Proc("778")
        if cmd[0] == "sacct" and "Submit,Start" in cmd:
            return Proc("2026-10-03T01:00:00|2026-10-03T01:02:30")
        if cmd[0] == "sacct":
            time.sleep(0.2)
            seen.append(budget.used().queue_s)  # what a phase deadline sees at this poll
            return Proc(next(states))
        return Proc()

    backend = SlurmBackend({"account": "a", "poll_s": 0}, runner=fake_run)
    with budget.waiting() as on_queue:
        res = backend.run(Job(name="e", command="true", cwd=tmp_path, env={}, gpus=1,
                              log_path=tmp_path / "e.log", on_queue=on_queue))
    assert seen[0] >= 0.2 and seen[1] >= seen[0] + 0.2   # growing while the job is pending
    assert seen[3] <= seen[2] + 0.05                      # and no longer once it runs
    assert res.status == "ok" and res.queue_s == pytest.approx(150) and not budget._live
