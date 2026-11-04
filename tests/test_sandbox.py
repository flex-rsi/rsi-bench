"""The filesystem sandbox: the launcher itself, and a run whose agent and solution try to read
held-out data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import call, linked_project, write_script

from rsibench.sandbox import LAUNCHER, probe

pytestmark = pytest.mark.skipif(probe() is not None,
                                reason=f"unprivileged user namespaces unavailable: {probe()}")

CHECK = r"""
import json, os, sys
root, allowed, ro = sys.argv[1:4]
res = {"listing": sorted(os.listdir(root)), "secret": os.path.exists(os.path.join(root, "secret.txt")),
       "cwd": os.getcwd()}
res["read_allowed"] = open(os.path.join(allowed, "a.txt")).read()
open(os.path.join(allowed, "new.txt"), "w").write("x")
try:
    open(os.path.join(ro, "c.txt"), "w").write("x")
    res["ro_write"] = "allowed"
except OSError:
    res["ro_write"] = "refused"
print(json.dumps(res))
"""


def test_launcher_shows_only_allowed_paths(tmp_path):
    root = tmp_path / "root"
    (root / "allowed").mkdir(parents=True)
    (root / "ro").mkdir()
    (root / "other").mkdir()
    (root / "secret.txt").write_text("labels")
    (root / "allowed" / "a.txt").write_text("hello")
    proc = subprocess.run(
        [sys.executable, str(LAUNCHER), "--hide", str(root), "--rw", str(root / "allowed"),
         "--ro", str(root / "ro"), "--ro", sys.prefix, "--ro", sys.base_prefix, "--",
         sys.executable, "-c", CHECK, str(root), str(root / "allowed"), str(root / "ro")],
        cwd=root / "other", capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout)
    assert res["listing"] == ["allowed", "ro"] and not res["secret"]
    assert res["cwd"] == "/"  # started in a hidden directory, which no longer exists inside
    assert res["read_allowed"] == "hello" and res["ro_write"] == "refused"
    assert (root / "allowed" / "new.txt").exists()  # read-write paths write through


FORK = r"""
import os, sys, time
kids = []
try:
    for _ in range(200):
        pid = os.fork()
        if pid == 0:
            os.setsid()  # leaving the process group does not escape the limit
            time.sleep(10)
            os._exit(0)
        kids.append(pid)
except OSError:
    pass
print(len(kids))
for p in kids:
    os.kill(p, 9)
    os.waitpid(p, 0)
"""


def test_launcher_caps_the_processes_of_one_launch(tmp_path):
    from rsibench.sandbox import Sandbox
    base = [sys.executable, str(LAUNCHER), "--hide", str(tmp_path), "--ro", sys.prefix,
            "--ro", sys.base_prefix]
    capped = subprocess.run(base + ["--max-procs", "40", "--", sys.executable, "-c", FORK],
                            capture_output=True, text=True, timeout=120)
    assert capped.returncode == 0, capped.stderr
    assert 30 <= int(capped.stdout) < 40  # counted in the launch's own namespace, not per user
    free = subprocess.run(base + ["--", sys.executable, "-c", FORK],
                          capture_output=True, text=True, timeout=120)
    assert int(free.stdout) == 200
    # the run's sandbox passes the site's limit (default DEFAULT_MAX_PROCS; 0 turns it off)
    sb = Sandbox({"enabled": True}, tmp_path / "s", [tmp_path], max_procs=123)
    assert sb.prefix([], [])[-3:] == ["--max-procs", "123", "--"]
    assert "--max-procs" not in Sandbox({"enabled": True}, tmp_path / "s", [tmp_path],
                                        max_procs=0).prefix([], [])


def test_launcher_follows_links_from_one_hidden_root_into_another(tmp_path):
    # a venv in one hidden root whose interpreter links, through a version link, into another
    # (a uv venv under /data whose python lives under /home): python_roots() allows all four.
    # The venv's link is the shorter path, so it comes before the version link it leads through.
    a, b = tmp_path / "a", tmp_path / "b"
    full, minor = b / "uv" / "cpython-3.11.15-linux-x86_64", b / "uv" / "cpython-3.11-linux-x86_64"
    (full / "bin").mkdir(parents=True)
    (full / "bin" / "python3.11").write_text("interpreter\n")
    minor.symlink_to(full)
    (a / "venv" / "bin").mkdir(parents=True)
    (a / "venv" / "bin" / "python").symlink_to(minor / "bin" / "python3.11")
    (a / "secret.txt").write_text("labels")
    allowed = [a / "venv", a / "venv" / "bin" / "python", minor, full]
    args = [sys.executable, str(LAUNCHER), "--hide", str(a), "--hide", str(b)]
    for p in allowed + [Path(sys.prefix), Path(sys.base_prefix)]:
        args += ["--ro", str(p)]
    proc = subprocess.run(args + ["--", "/bin/sh", "-c", f"cat {a}/venv/bin/python; ls {a}"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["interpreter", "venv"]



def test_launcher_reaches_a_venv_through_a_linked_parent_without_exposing_the_rest(tmp_path):
    from rsibench.sandbox import interpreter_paths
    alias_venv, venv, base = linked_project(tmp_path)
    args = [sys.executable, str(LAUNCHER), "--hide", str(tmp_path)]
    for p in interpreter_paths(str(alias_venv / "bin" / "python"), str(alias_venv), str(base)):
        args += ["--ro", p]
    for p in (sys.prefix, sys.base_prefix):
        args += ["--ro", p]
    proc = subprocess.run(args + ["--", "/bin/sh", "-c",
                                  f"cat {alias_venv}/bin/python; ls {tmp_path}/fsw/me; "
                                  f"cat {tmp_path}/fs12/projects/me/val/labels.json || echo HIDDEN"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["interpreter", "venv", "HIDDEN"]


def test_sandboxed_run_hides_held_out_data_from_agent_and_solution(sandbox):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run

    val_labels = sandbox.tmp / "data" / "toy-regression" / "val" / "labels.json"
    assert val_labels.exists()
    other_run = sandbox.tmp / "runs" / "someone-elses-run"
    other_run.mkdir(parents=True)
    probe_cmd = (f"{{ cat {val_labels} || echo VAL_HIDDEN; ls {other_run} || echo RUN_HIDDEN; "
                 f"find {sandbox.tmp} -path '*eval-val*' 2>/dev/null | head -3; echo FIND_DONE; "
                 f"echo HOME=$HOME CACHE=$RSIB_CACHE; }} > ../../records/notes/toy-regression.md 2>&1")
    cheat = (f"import os\nif os.path.exists({str(val_labels)!r}):\n"
             f"    raise RuntimeError('labels are visible to the solution')\n"
             "def fit_predict(train_x, train_y, query_x):\n"
             "    m = sum(train_y) / len(train_y)\n    return [m for _ in query_x]\n")
    script = {"executor": {"1": [
        {"tool_calls": [call("bash", command=probe_cmd)]},
        {"tool_calls": [call("write_file", path="workspace/toy-regression/solver.py", content=cheat)]},
        {"tool_calls": [call("finish", summary="probe")]},
    ]}}
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="base", seed=21,
                         model_params={"script": str(write_script(sandbox, script, "sb.json"))},
                         protocol_overrides={"max_rounds": 1}, sandbox={"enabled": True},
                         repo=sandbox.repo, log=lambda *a: None)
    RunEngine(wt, log=lambda *a: None).start()
    notes = (Path(wt) / "records" / "notes" / "toy-regression.md").read_text()
    assert "VAL_HIDDEN" in notes and "RUN_HIDDEN" in notes
    assert "eval-val" not in notes.split("FIND_DONE")[0]  # val evaluation outputs are not visible
    assert "/_scratch/" in notes and "/home" in notes and "/cache" in notes
    rounds = Records(wt).read("rounds")
    assert rounds[1]["val"]["status"] == "ok"  # the solution ran, and could not see the labels
    assert Records(wt).run()["sandbox"]["enabled"] is True
    # at the end the scratch was inventoried and moved to the archive, out of every agent's reach
    slug = Path(wt).name
    assert not (sandbox.tmp / "runs" / "_scratch" / slug).exists()
    archived = list((sandbox.tmp / "runs" / "_archive").glob(f"{slug}__*"))
    assert len(archived) == 1 and (archived[0] / "ARCHIVE.json").exists()
    assert (archived[0] / "scratch" / "home").is_dir()
    env = Records(wt).read("environment")
    assert len(env) == 1 and env[0]["archive"] == archived[0].name
    assert [p["round"] for p in Records(wt).read("posthoc")] == [0]
