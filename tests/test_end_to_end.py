"""End-to-end runs with the scripted mock model: the full round lifecycle, git lineage, records,
scope enforcement, modes, streams, budgets, crash rollback, validation, results and push."""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import CONSTANT_SOLVER, POLY_SOLVER, call, poly, run_to_end, write_script

from rsibench import gitops
from rsibench.posthoc import posthoc
from rsibench.report import summarize
from rsibench.results import append as results_append
from rsibench.validate import validate_ref

SOLVER = "workspace/toy-regression/solver.py"
NOTES = "records/notes/toy-regression.md"
METRICS_CMD = ("python3 -c \"import json,os; json.dump({'ok': 1, 'info': 'smoke'}, "
               "open(os.environ['RSIB_METRICS_FILE'], 'w'))\"")

FULL_SCRIPT = {
    "executor": {
        "1": [
            {"tool_calls": [call("read_file", path=SOLVER)]},
            {"tool_calls": [call("write_file", path=SOLVER, content=poly(3)),
                            call("write_file", path=NOTES, content="r1: cubic fit\n")]},
            {"tool_calls": [call("run_experiment", name="smoke", command=METRICS_CMD)]},
            {"tool_calls": [call("evaluate_dev")]},
            {"tool_calls": [call("finish", summary="cubic least squares")]},
        ],
        "2": [
            {"tool_calls": [call("write_file", path=SOLVER, content=CONSTANT_SOLVER),
                            call("bash", command="echo tampered >> ../../tasks/toy-regression/instruction.md")]},
            {"tool_calls": [call("finish", summary="constant predictor")]},
        ],
        "3": [
            {"tool_calls": [call("write_file", path=SOLVER, content=poly(7))]},
            {"tool_calls": [call("finish", summary="degree 7")]},
        ],
    },
    "improver": {
        "1": [
            {"tool_calls": [call("edit_file", path="harness/executor/system.md",
                                 old_text="## How to work",
                                 new_text="## How to work\n\n- Fit a model on the training set first.")]},
            {"tool_calls": [call("write_file", path="harness/improver/memory/lessons.md",
                                 content="# Lessons\n\n- Polynomial features help smooth 1-D regression.\n")]},
            {"tool_calls": [call("finish", summary="tightened executor prompt")]},
        ],
    },
    "meta": {
        "2": [
            {"tool_calls": [call("edit_file", path="harness/improver/config.toml",
                                 old_text="sigma_k = 2.0", new_text="sigma_k = 1.0")]},
            {"tool_calls": [call("finish", summary="relaxed acceptance")]},
        ],
    },
}


def tool_results(sandbox, phase: str, rnd: int = 1) -> list[dict]:
    """The tool results the model received in one phase of a round, from its transcript."""
    path = next((sandbox.tmp / "artifacts").rglob(f"r{rnd:03d}-{phase}.jsonl"))
    return [res for line in path.read_text().splitlines()
            for msg in json.loads(line)["new_messages"] for res in msg.get("results", [])]


def test_full_run_lifecycle(sandbox, tmp_path):
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], FULL_SCRIPT, mode="full")
    rounds = rec.read("rounds")
    assert [r["decision"] for r in rounds] == ["baseline", "accept", "reject", "accept"]
    base, r1, r2, r3 = (r["val"]["score"] for r in rounds)
    assert r1 < base and r2 > r1 and r3 < r1                     # mse: lower is better
    assert rounds[3]["best_val_after"] == r3
    assert rounds[1]["decided_by"] == "harness"

    # the rejected candidate was reverted; the accepted degree-7 solver is in place
    assert "deg=7" in (wt / SOLVER).read_text()
    # out-of-scope edit in round 2 was reverted and logged
    assert "tampered" not in (wt / "tasks/toy-regression/instruction.md").read_text()
    viol = rec.read("violations")
    assert any(v["kind"] == "path" and "tasks/toy-regression/instruction.md" in v["paths"]
               and v["round"] == 2 for v in viol)

    # harness self-modification: E in round 1 (improver), I in round 1 (lessons) and round 2 (meta)
    assert "Fit a model on the training set first." in (wt / "harness/executor/system.md").read_text()
    assert "sigma_k = 1.0" in (wt / "harness/improver/config.toml").read_text()
    assert rounds[1]["harness_changes"]["E"]["commit"] and rounds[1]["harness_changes"]["I"]["commit"]
    assert rounds[2]["phases"]["meta"]["status"] == "finished"
    assert (wt / NOTES).read_text().startswith("r1: cubic fit")

    # experiments and evaluations were recorded through the gateway
    exps = rec.read("experiments")
    kinds = [e["kind"] for e in exps]
    assert kinds.count("eval-val") == 4 and "eval-dev" in kinds and "eval-test" in kinds
    # val/test evaluation outputs (per-example scores, logs) stay out of the agent-visible artifacts
    art = sandbox.tmp / "artifacts"
    assert not [p for p in art.rglob("*") if p.name in ("eval-val", "final-test", "posthoc")]
    smoke = next(e for e in exps if e["kind"] == "experiment")
    # "info" is not an object: an experiment file keeps it as a plain metric, and says so
    assert smoke["status"] == "ok" and smoke["metrics"] == {"ok": 1, "info": "smoke"}
    reply = next(json.loads(r["content"]) for r in tool_results(sandbox, "executor")
                 if '"exp_id"' in r["content"])
    assert reply["metrics"] == {"ok": 1, "info": "smoke"} and reply["detail"].startswith("info:")
    usage = rec.read("usage")
    assert usage and all(u["usd"] is not None for u in usage)

    # final test score, normalized against the task anchors
    final = rec.read("final")
    assert len(final) == 1 and final[0]["test"]["status"] == "ok"
    assert final[0]["test"]["score"] < base and final[0]["normalized"]["test"] > 0.9
    assert final[0]["ended_reason"] == "max_rounds" and final[0]["final_round"] == 3

    # git lineage: trailers, a revert, tags
    commits = gitops.log_commits(wt, f"{rec.run()['base']['commit']}..HEAD")
    scopes = [c["trailers"].get("Rsib-Scope") for c in commits]
    assert {"init", "solution", "revert", "E", "I", "records", "final", "posthoc",
            "environment"} <= set(scopes)
    # the end of the run archived the (here empty) scratch and recorded its inventory
    env = rec.read("environment")
    assert len(env) == 1 and env[0]["inventory"]["bytes"] == 0 and env[0]["delete_after"]
    tags = gitops.git(["tag", "--list", f"{branch}/*"], wt).split()
    assert {f"{branch}/toy-regression/r{i}" for i in (0, 1, 3)} <= set(tags)
    assert f"{branch}/final" in tags and f"{branch}/toy-regression/r2" not in tags

    # CI integrity checks pass on the untouched branch
    assert validate_ref(sandbox.repo, branch) == []

    # post-hoc test evaluation of accepted versions ran at the end of the run; again is a no-op
    assert [p["round"] for p in rec.read("posthoc")] == [0, 1, 3]
    assert posthoc(wt, log=lambda *a: None) == []
    assert "posthoc" in scopes
    summary = summarize(wt / "records")
    task = summary["tasks"][0]
    assert task["accepted"] == 2 and task["rejected"] == 1
    assert task["false_acceptance"]["judged"] == 2 and task["test_auc"] is not None
    assert task["harness_commits"]["E"] >= 1 and task["harness_commits"]["I"] >= 2
    # round 1 (cubic) did most of the work; round 3 (degree 7) added the rest
    assert task["first_round_gain"] > 0.8 and task["later_gain"] > 0.0
    assert abs(task["first_round_gain"] + task["later_gain"]
               - (task["normalized"]["val"] - 0.0)) < 0.05
    assert validate_ref(sandbox.repo, branch) == []

    # results index on its own branch, idempotent per commit
    results_append(sandbox.repo, branch)
    results_append(sandbox.repo, branch)
    index = gitops.show(sandbox.repo, "refs/heads/results", "runs.jsonl").splitlines()
    assert len(index) == 1 and json.loads(index[0])["run_id"] == branch

    # cross-run table, from the index and straight from the run branches
    from rsibench.results import build_table, load_index, rows_from_branches
    for rows in (load_index(sandbox.repo), rows_from_branches(sandbox.repo)):
        assert build_table(rows)["runs"] == 0  # this test run is unsandboxed: left out by default
        table = build_table(rows, include_unsandboxed=True)
        assert table["runs"] == 1 and table["groups"][0]["condition"] == "full"
        assert table["groups"][0]["normalized_test"]["mean"] > 0.9

    # a reviewed unsandboxed run can count as standard: a newer row carries the waiver
    results_append(sandbox.repo, branch, accept_unsandboxed="reviewed transcripts")
    results_append(sandbox.repo, branch, accept_unsandboxed="reviewed transcripts")  # no-op
    rows = load_index(sandbox.repo)
    assert len(rows) == 2 and rows[-1]["sandbox_waiver"]["reason"] == "reviewed transcripts"
    assert build_table(rows)["runs"] == 1

    # one markdown report of every run, read from the run branches; committed to `results`
    from rsibench.cli import main
    report = tmp_path / "RESULTS.md"
    assert main(["results", "report", "--repo", str(sandbox.repo), "--output", str(report),
                 "--commit-to", "results"]) == 0
    text = report.read_text()
    assert "## toy-regression" in text and "| waiver |" in text and "reviewed transcripts" in text
    assert "r0 " in text and "r1 " in text and "r3 " in text  # the post-hoc curve
    assert gitops.show(sandbox.repo, "refs/heads/results", "RESULTS.md") == text

    # push branch + tags to a remote
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    gitops.git(["remote", "add", "test-remote", str(remote)], sandbox.repo)
    gitops.push(wt, "test-remote", branch)
    remote_refs = subprocess.run(["git", "ls-remote", str(remote)], capture_output=True,
                                 text=True, check=True).stdout
    assert f"refs/heads/{branch}" in remote_refs and f"refs/tags/{branch}/final" in remote_refs


def test_validate_detects_tampering(sandbox):
    script = {"executor": {"1": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base")
    assert validate_ref(sandbox.repo, branch) == []
    rounds_path = wt / "records" / "rounds.jsonl"
    lines = rounds_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["val"]["score"] = 0.0
    rounds_path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    (wt / "tasks" / "toy-regression" / "instruction.md").write_text("edited\n")
    gitops.commit(wt, "tamper", {"Rsib-Run": branch, "Rsib-Scope": "records"},
                  ["records", "tasks"])
    errors = validate_ref(sandbox.repo, branch)
    assert any("append-only" in e for e in errors)
    assert any("frozen path changed: tasks/toy-regression/instruction.md" in e for e in errors)


def test_fixed_i_reverts_improver_edits_and_base_skips_improver(sandbox):
    script = {"improver": {"*": [
        {"tool_calls": [call("bash", command="echo '# hacked' >> harness/improver/accept.py"),
                        call("edit_file", path="harness/executor/round_prompt.md",
                             old_text="## Budget", new_text="## Budget (shared)")]},
        {"tool_calls": [call("finish", summary="tried to edit I")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="fixed-i")
    assert "# hacked" not in (wt / "harness/improver/accept.py").read_text()
    assert "## Budget (shared)" in (wt / "harness/executor/round_prompt.md").read_text()
    assert any(v["kind"] == "path" and "harness/improver/accept.py" in v["paths"]
               for v in rec.read("violations"))
    rounds = rec.read("rounds")
    assert all(r["harness_changes"]["I"]["commit"] is None for r in rounds[1:])

    branch_b, wt_b, rec_b = run_to_end(sandbox, ["toy-regression"], script, mode="base", seed=2)
    assert all("improver" not in r["phases"] for r in rec_b.read("rounds")[1:])
    assert all(r["decided_by"] in ("orchestrator", "default") for r in rec_b.read("rounds"))


def test_probe_imports_improver_from_another_run(sandbox):
    branch, wt, _ = run_to_end(sandbox, ["toy-regression"], FULL_SCRIPT, mode="full")
    probe_branch, probe_wt, rec = run_to_end(
        sandbox, ["toy-segmentation"], None, mode="fixed-i", model="mock:finish",
        improver_from=f"{branch}/final", seed=3)
    run = rec.run()
    assert run["improver_source"]["ref"] == f"{branch}/final"
    assert (probe_wt / "harness/improver/config.toml").read_text() == \
        (wt / "harness/improver/config.toml").read_text()
    assert "sigma_k = 1.0" in (probe_wt / "harness/improver/config.toml").read_text()
    assert validate_ref(sandbox.repo, probe_branch) == []



def test_executor_from_imports_e_but_keeps_the_improver(sandbox):
    branch, wt, _ = run_to_end(sandbox, ["toy-regression"], FULL_SCRIPT, mode="full")
    e_branch, e_wt, rec = run_to_end(
        sandbox, ["toy-regression"], None, mode="base", model="mock:finish",
        executor_from=f"{branch}/final", seed=4)
    run = rec.run()
    assert run["executor_source"]["ref"] == f"{branch}/final"
    assert run["improver_source"] is None
    # the source run's E edit comes along, its I edit does not
    assert "Fit a model on the training set first." in \
        (e_wt / "harness/executor/system.md").read_text()
    assert "sigma_k = 2.0" in (e_wt / "harness/improver/config.toml").read_text()
    for f in (wt / "harness").rglob("*.py"):
        rel = f.relative_to(wt)
        if "improver" not in rel.parts and "__pycache__" not in rel.parts:
            assert (e_wt / rel).read_text() == f.read_text(), rel
    assert validate_ref(sandbox.repo, e_branch) == []


def test_stream_runs_tasks_in_order_with_meta_at_boundary(sandbox):
    branch, wt, rec = run_to_end(sandbox, ["toy-regression", "toy-segmentation"], None,
                                 mode="full", model="mock:finish")
    rounds = rec.read("rounds")
    assert [r["task_id"] for r in rounds] == ["toy-regression"] * 4 + ["toy-segmentation"] * 4
    first_seg = next(r for r in rounds if r["task_id"] == "toy-segmentation" and r["round"] == 1)
    assert "meta" in first_seg["phases"]
    assert {f["task_id"] for f in rec.read("final")} == {"toy-regression", "toy-segmentation"}
    # test scores are produced only after the whole stream finished
    assert rec.read("final")[0]["test"]["status"] == "ok"
    assert validate_ref(sandbox.repo, branch) == []


def test_usd_budget_stops_the_task(sandbox):
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], None, mode="full",
                                 model="mock:finish", budget={"usd": 0.0001})
    final = rec.read("final")[0]
    assert final["ended_reason"] == "budget_usd"
    assert len(rec.read("rounds")) == 2  # baseline + the round in which the budget ran out


def test_queue_time_of_the_baseline_validation_is_not_charged(sandbox, monkeypatch):
    import time

    from rsibench.orchestrator.engine import RunEngine
    real = RunEngine.run_split_eval

    def queued_val(self, task, workspace, split, out_dir, rnd, **kw):
        start = time.monotonic()
        if split == "val" and rnd == 0:
            time.sleep(1.5)  # stands in for a job pending in the queue
        result = real(self, task, workspace, split, out_dir, rnd, **kw)
        if split == "val" and rnd == 0:
            result["queued"] = (start, start + 1.5)
        return result

    monkeypatch.setattr(RunEngine, "run_split_eval", queued_val)
    _, _, rec = run_to_end(sandbox, ["toy-regression"], None, mode="full", model="mock:finish")
    baseline = rec.read("rounds")[0]["cumulative"]
    assert baseline["queue_s"] >= 1.5


def test_a_phase_is_not_timed_out_while_its_job_is_pending(sandbox, monkeypatch):
    """A job pending in the queue past the phase's limit: the deadline moves on during the wait,
    so the phase gets the job's result and finishes."""
    import time

    from rsibench.compute.local import LocalBackend
    from rsibench.orchestrator.engine import RunEngine
    run, run_phase = LocalBackend.run, RunEngine.run_phase

    def queued_first(self, job):
        if job.name.startswith("eval-"):
            return run(self, job)
        submitted = time.monotonic()
        job.report_queue(submitted)
        time.sleep(12)                       # pending, well past the phase's 5 s
        res = run(self, job)
        res.queued = job.report_queue(submitted, submitted + 12)
        return res

    def short_executor(self, phase, *args, **kw):
        if phase == "executor":
            kw["timeout_s"] = 5
        return run_phase(self, phase, *args, **kw)

    monkeypatch.setattr(LocalBackend, "run", queued_first)
    monkeypatch.setattr(RunEngine, "run_phase", short_executor)
    script = {"executor": {"1": [
        {"tool_calls": [call("run_experiment", name="queued", command=METRICS_CMD)]},
        {"tool_calls": [call("finish", summary="ran one queued job")]}]}}
    _, _, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                           protocol={"max_rounds": 1})
    assert rec.read("rounds")[1]["phases"]["executor"]["status"] == "finished"
    exp = next(e for e in rec.read("experiments") if e["kind"] == "experiment")
    assert exp["status"] == "ok" and exp["queue_s"] >= 12


def test_crash_streak_rolls_back_the_harness(sandbox):
    broken = {"executor": {}, "improver": {"1": [
        {"tool_calls": [call("write_file", path="harness/executor/phase.py",
                             content="raise RuntimeError('broken by improver')\n")]},
        {"tool_calls": [call("finish", summary="broke the executor")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], broken, mode="full")
    rounds = rec.read("rounds")
    assert rounds[2]["phases"]["executor"]["status"] == "crash"
    assert any(v["kind"] == "crash_rollback" for v in rec.read("violations"))
    assert "broken by improver" not in (wt / "harness/executor/phase.py").read_text()
    assert validate_ref(sandbox.repo, branch) == []


def test_agent_cannot_edit_records(sandbox):
    script = {"executor": {"1": [
        {"tool_calls": [call("bash", command="echo '{\"forged\": 1}' >> ../../records/rounds.jsonl"),
                        call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("finish", summary="tried to forge a record")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base")
    assert all("forged" not in r for r in rec.read("rounds"))
    assert any("records/rounds.jsonl" in v["paths"] for v in rec.read("violations"))
    assert validate_ref(sandbox.repo, branch) == []


def test_resume_after_kill_continues_from_last_complete_round(sandbox):
    import os
    import signal
    import sys
    import time

    from conftest import write_script
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run

    slow = {"executor": {
        "1": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]}],
        "2": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(5)),
                              call("bash", command="sleep 4")]}]}}
    script_path = write_script(sandbox, slow, "resume.json")
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="base", seed=7,
                         model_params={"script": str(script_path)}, repo=sandbox.repo,
                         log=lambda *a: None)
    proc = subprocess.Popen([sys.executable, "-m", "rsibench.cli", "run", "start", str(wt)],
                            start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    rounds_file = wt / "records" / "rounds.jsonl"
    deadline = time.time() + 120
    while time.time() < deadline:
        if rounds_file.exists() and len(rounds_file.read_text().splitlines()) >= 2:
            break
        time.sleep(0.2)
    time.sleep(1.0)  # round 2 executor is now inside `sleep 4`
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    time.sleep(4.5)  # let the orphaned harness phase exit
    # resume with a fast script for the remaining rounds
    script_path.write_text(json.dumps({"executor": {
        "1": [], "2": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(5))]}]}}))
    RunEngine(wt, log=lambda *a: None).start()
    rec = Records(wt)
    rounds = [r["round"] for r in rec.read("rounds")]
    assert rounds == [0, 1, 2, 3]
    assert any(v["kind"] == "resume_discard" for v in rec.read("violations"))
    assert rec.read("final")[0]["test"]["status"] == "ok"
    assert validate_ref(sandbox.repo, branch) == []


def test_idle_rounds_end_the_task(sandbox):
    proto = sandbox.repo / "protocol" / "default.toml"
    proto.write_text(proto.read_text().replace("max_rounds = 3", "max_rounds = 50")
                     .replace("max_idle_rounds = 5", "max_idle_rounds = 2"))
    gitops.git(["commit", "-qam", "protocol for idle test"], sandbox.repo,
               env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    gitops.git(["tag", "harness-v0.0.2"], sandbox.repo)
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], None, mode="base", model="mock:finish")
    assert rec.read("final")[0]["ended_reason"] == "idle"
    assert len(rec.read("rounds")) == 3  # baseline + two idle rounds


def test_phase_call_limit_stops_a_looping_phase(sandbox):
    loop = {"executor": {"*": [{"tool_calls": [call("check_budget")]}] * 50}}
    from conftest import write_script
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="base", seed=9,
                         model_params={"script": str(write_script(sandbox, loop, "loop.json"))},
                         protocol_overrides={"max_model_calls_per_phase": 5, "max_rounds": 1},
                         repo=sandbox.repo, log=lambda *a: None)
    RunEngine(wt, log=lambda *a: None).start()
    rec = Records(wt)
    r1 = rec.read("rounds")[1]
    # The starter harness reads the cap from the phase context and stops by itself at 5 calls
    # (the gateway's own refusal of a sixth call is tested in test_units).
    assert r1["phases"]["executor"]["status"] == "stopped"
    assert r1["phases"]["executor"]["steps"] == 5
    assert len([u for u in rec.read("usage") if u["round"] == 1]) == 5
    assert rec.run()["protocol"]["max_model_calls_per_phase"] == 5


def test_processes_a_phase_leaves_running_end_with_the_phase(sandbox):
    from conftest import running
    pid_file = sandbox.tmp / "leftover.pid"
    script = {"executor": {"1": [{"tool_calls": [
        call("bash", command=f"sleep 60 > /dev/null 2>&1 & echo $! > {pid_file}")]}]}}
    run_to_end(sandbox, ["toy-regression"], script, mode="base", protocol={"max_rounds": 1})
    assert not running(int(pid_file.read_text()))


def test_finalize_retries_a_failed_test_evaluation(sandbox):
    from rsibench.orchestrator.engine import RunEngine
    script = {"executor": {"1": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base")
    good = rec.read("final")[0]
    # simulate an infrastructure failure recorded as the latest test result
    failed = dict(good, test={"status": "error", "score": None, "metrics": {}},
                  normalized={"val": good["normalized"]["val"], "test": None})
    rec.append("final", failed)
    gitops.commit(wt, "simulated failed test evaluation", {"Rsib-Run": branch, "Rsib-Scope": "final"},
                  ["records"])
    assert summarize(wt / "records")["tasks"][0]["test"] is None
    engine = RunEngine(wt, log=lambda *a: None)
    engine.records.track_all()
    engine.finalize(retry_failed=True)
    finals = rec.read("final")
    assert [f["test"]["status"] for f in finals] == ["ok", "error", "ok"]
    assert summarize(wt / "records")["tasks"][0]["test"] == good["test"]["score"]
    assert validate_ref(sandbox.repo, branch) == []


def test_val_ignores_files_that_cannot_be_committed(sandbox):
    """A solution that only works with a file too large to commit must fail val too, not just test."""
    big_solver = POLY_SOLVER.replace("DEG", "3").replace(
        "def fit_predict(", "import pathlib\n_W = (pathlib.Path(__file__).parent / 'weights.bin').read_bytes()\n\ndef fit_predict(")
    script = {"executor": {"1": [{"tool_calls": [
        call("write_file", path=SOLVER, content=big_solver),
        call("bash", command="head -c 6000000 /dev/zero > weights.bin")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base")
    r1 = rec.read("rounds")[1]
    assert r1["decision"] == "reject" and r1["val"]["status"] == "error"
    assert any(v["kind"] == "large_file" for v in rec.read("violations"))


def test_a_phase_cut_short_by_a_usage_limit_runs_again_when_the_model_is_back(sandbox,
                                                                              monkeypatch):
    import time as _time
    from rsibench.models.base import ModelError
    from rsibench.models.mock import ScriptedModel
    complete, hit = ScriptedModel.complete, []

    def limited_once(self, system, messages, tools, max_tokens, meta):
        if meta.get("phase") == "executor" and not hit:  # the first executor call: limit hit
            hit.append(1)
            self.note_unavailable(_time.monotonic() + 2)
            raise ModelError("You've hit your session limit · resets 9:20pm")
        return complete(self, system, messages, tools, max_tokens, meta)

    monkeypatch.setattr(ScriptedModel, "complete", limited_once)
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("finish", summary="cubic")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"max_rounds": 1})
    r1 = rec.read("rounds")[1]
    assert hit and r1["phases"]["executor"]["status"] == "finished"  # run again, not lost
    assert r1["decision"] == "accept" and "deg=3" in (wt / SOLVER).read_text()
    assert r1["cumulative"]["queue_s"] >= 1.5  # the wait for the model is not charged as wall time


def test_a_round_whose_phases_all_fail_on_the_model_pauses_the_run(sandbox, monkeypatch):
    from rsibench.models.base import ModelError
    from rsibench.models.mock import ScriptedModel
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run
    complete, down = ScriptedModel.complete, [True]

    def refusing(self, system, messages, tools, max_tokens, meta):
        if down[0]:  # a provider message the adapter does not know as a limit
            raise ModelError("claude CLI error (exit 1): something unexpected")
        return complete(self, system, messages, tools, max_tokens, meta)

    monkeypatch.setattr(ScriptedModel, "complete", refusing)
    script = {"executor": {"1": [{"tool_calls": [call("finish", summary="nothing")]}]}}
    params = {"script": str(write_script(sandbox, script, "script-down.json"))}
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="fixed-i", seed=7,
                         model_params=params, protocol_overrides={"max_rounds": 3},
                         repo=sandbox.repo, log=lambda *a: None)
    result = RunEngine(wt, log=lambda *a: None).start()
    rec = Records(wt)
    assert "round 1" in result["paused"] and "something unexpected" in result["paused"]
    assert [r["round"] for r in rec.read("rounds")] == [0, 1] and not rec.read("final")
    down[0] = False  # the model is back: the run goes on from round 2 and ends normally
    result = RunEngine(wt, log=lambda *a: None).start()
    assert not result.get("paused") and rec.read("final")[0]["ended_reason"] == "max_rounds"
    assert validate_ref(sandbox.repo, branch) == []


def test_dev_improvement_forces_submission_of_that_state(sandbox):
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("evaluate_dev")]},                       # beats the starter -> submit
        {"tool_calls": [call("run_experiment", name="more", command="true"),
                        call("write_file", path=SOLVER, content=CONSTANT_SOLVER)]},
        {"tool_calls": [call("finish", summary="done")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base")
    rounds = rec.read("rounds")
    assert rounds[0]["candidate"]["dev_score"] is not None           # bar from the starter
    r1 = rounds[1]
    assert r1["decision"] == "accept" and r1["candidate"]["dev_score"] < rounds[0]["candidate"]["dev_score"]
    assert r1["phases"]["executor"]["summary"].startswith("[submitted after dev improvement")
    assert "deg=3" in (wt / SOLVER).read_text()                     # not the later constant solver
    assert not [e for e in rec.read("experiments") if e["round"] == 1 and e["kind"] == "experiment"]
    assert validate_ref(sandbox.repo, branch) == []


def test_a_dev_gain_below_the_submit_line_does_not_end_the_round(sandbox):
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("evaluate_dev")]},                    # a gain, short of the line
        {"tool_calls": [call("run_experiment", name="more", command="true")]},  # still allowed
        {"tool_calls": [call("finish", summary="cubic")]}]}}
    # the line: dev mse of the starter minus 1.05 x the anchor span, which the cubic falls short of
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"submit_min_gain": 1.05, "max_rounds": 1})
    rounds = rec.read("rounds")
    r1 = rounds[1]
    assert r1["decision"] == "accept"  # submitted by finishing, not by the rule
    assert not r1["phases"]["executor"]["summary"].startswith("[submitted")
    assert [e["status"] for e in rec.read("experiments")
            if e["round"] == 1 and e["kind"] == "experiment"] == ["ok"]
    reply = next(json.loads(r["content"]) for r in tool_results(sandbox, "executor")
                 if '"submit_rule"' in r["content"])
    bar, line = reply["submit_rule"]["accepted_dev"], reply["submit_rule"]["auto_submit_at"]
    assert bar == rounds[0]["candidate"]["dev_score"] and line < reply["score"] < bar  # mse
    # the round prompt tells the executor what a submission costs and where the line is
    first = json.loads(next((sandbox.tmp / "artifacts").rglob("r001-executor.jsonl"))
                       .read_text().splitlines()[0])["new_messages"][0]["content"]
    assert "## Submissions" in first and "Validation submissions used: 0 of 30" in first
    assert "round 1 of at most 1" in first and f"scoring below {line:.6g}" in first
    assert validate_ref(sandbox.repo, branch) == []


def test_a_round_cut_off_with_nothing_measured_spends_no_validation(sandbox):
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},  # never measured
        {"tool_calls": [call("check_budget")]}]}}                          # the call limit ends it
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"max_model_calls_per_phase": 2, "max_rounds": 1})
    r1 = rec.read("rounds")[1]
    assert r1["phases"]["executor"]["status"] == "stopped"
    assert (r1["decision"], r1["val"]["status"]) == ("no-change", "skipped")
    assert "discarded" in r1["decision_reason"] and not r1["candidate"]["changed"]
    assert "deg=3" not in (wt / SOLVER).read_text()  # the starter again
    assert not [e for e in rec.read("experiments") if e["kind"] == "eval-val" and e["round"] == 1]
    assert validate_ref(sandbox.repo, branch) == []


def test_a_round_cut_off_submits_its_best_dev_evaluation(sandbox):
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("evaluate_dev")]},                          # better, short of the line
        {"tool_calls": [call("write_file", path=SOLVER, content=CONSTANT_SOLVER)]}]}}  # cut off
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"submit_min_gain": 1.05, "max_model_calls_per_phase": 3,
                                           "max_rounds": 1})
    r1 = rec.read("rounds")[1]
    assert r1["phases"]["executor"]["status"] == "stopped"
    assert r1["phases"]["executor"]["summary"].startswith(
        "[stopped before finishing; its best dev evaluation is submitted")
    assert r1["decision"] == "accept" and r1["candidate"]["dev_score"] is not None
    assert "deg=3" in (wt / SOLVER).read_text()  # the measured cubic, not the later constant
    assert validate_ref(sandbox.repo, branch) == []


def test_a_candidate_keeps_the_dev_score_of_its_evaluated_snapshot(sandbox):
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("evaluate_dev")]},
        {"tool_calls": [call("finish", summary="cubic")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"submit_on_dev_improvement": False, "max_rounds": 1})
    dev = [e for e in rec.read("experiments") if e["kind"] == "eval-dev"]
    assert len(dev) == 1 and dev[0]["snapshot_sha"] and dev[0]["score"] is not None
    assert rec.read("rounds")[1]["candidate"]["dev_score"] == dev[0]["score"]  # same snapshot


def test_close_partial_round_submits_the_improving_snapshot_and_counts_its_cost(sandbox):
    import os
    import signal
    import sys
    import time

    from conftest import write_script
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run

    slow = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("evaluate_dev")]},
        {"tool_calls": [call("bash", command="sleep 8")]}]}}
    script_path = write_script(sandbox, slow, "close.json")
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="base", seed=11,
                         model_params={"script": str(script_path)}, repo=sandbox.repo,
                         protocol_overrides={"max_rounds": 1, "submit_grace_calls": 5},
                         log=lambda *a: None)
    proc = subprocess.Popen([sys.executable, "-m", "rsibench.cli", "run", "start", str(wt)],
                            start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    exp_file = wt / "records" / "experiments.jsonl"
    deadline = time.time() + 120
    while time.time() < deadline:
        if exp_file.exists() and '"evaluate dev"' in exp_file.read_text():
            break
        time.sleep(0.2)
    time.sleep(1.5)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    time.sleep(8)
    partial_usd = sum(u["usd"] for u in Records(wt).read("usage") if u["round"] == 1)
    assert partial_usd > 0
    RunEngine(wt, log=lambda *a: None).start(close_partial_round=True)
    rec = Records(wt)
    r1 = rec.read("rounds")[1]
    assert r1["decision"] == "accept" and r1["phases"]["executor"]["status"] == "finished"
    assert "deg=3" in (wt / SOLVER).read_text()
    assert r1["cost"]["usd"] >= partial_usd - 1e-9                    # the interrupted spend counts
    assert not any(v["kind"] == "resume_discard" for v in rec.read("violations"))
    assert validate_ref(sandbox.repo, branch) == []


def test_pause_before_job_end_and_resume(sandbox, monkeypatch):
    import time as _time
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run
    branch, wt = new_run(["toy-regression"], "mock:finish", mode="base", seed=12,
                         protocol_overrides={"max_rounds": 2}, repo=sandbox.repo, log=lambda *a: None)
    monkeypatch.setenv("RSIB_STOP_AT", str(_time.time() + 10))       # job "ends" in 10 s
    monkeypatch.setenv("RSIB_STOP_MARGIN_S", "3600")                  # but a round needs an hour
    out = RunEngine(wt, log=lambda *a: None).start()
    assert out["paused"] and [r["round"] for r in Records(wt).read("rounds")] == [0]
    monkeypatch.delenv("RSIB_STOP_AT")
    RunEngine(wt, log=lambda *a: None).start()
    rec = Records(wt)
    assert [r["round"] for r in rec.read("rounds")] == [0, 1, 2] and rec.read("final")
    assert validate_ref(sandbox.repo, branch) == []


CUBIC, DEG7 = ".rsib/scratch/cubic", ".rsib/scratch/deg7"
WRITE_CANDIDATES = {"tool_calls": [call("write_file", path=f"{CUBIC}/solver.py", content=poly(3)),
                                   call("write_file", path=f"{DEG7}/solver.py", content=poly(7))]}
EVALUATE_BOTH = {"tool_calls": [call("evaluate_dev", candidates=[
    {"id": "cubic", "path": CUBIC}, {"id": "deg7", "path": DEG7}])]}


def batch_reply(sandbox) -> dict:
    return next(json.loads(r["content"]) for r in tool_results(sandbox, "executor")
                if '"results"' in r["content"])


def test_candidates_in_the_scratch_directory_are_scored_in_one_call(sandbox):
    script = {"executor": {"1": [
        WRITE_CANDIDATES, EVALUATE_BOTH,
        {"tool_calls": [call("bash", command=f"cp ../../{DEG7}/solver.py solver.py")]},
        {"tool_calls": [call("finish", summary="kept the degree-7 candidate")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"submit_on_dev_improvement": False, "max_rounds": 1})
    cubic, deg7 = [e for e in rec.read("experiments") if e["kind"] == "eval-dev"]
    assert [cubic["exp_id"], deg7["exp_id"]] == ["toy-regression-r001-dev001-cubic",
                                                 "toy-regression-r001-dev001-deg7"]
    assert deg7["score"] < cubic["score"]
    reply = batch_reply(sandbox)
    assert {c: r["score"] for c, r in reply["results"].items()} == \
        {"cubic": cubic["score"], "deg7": deg7["score"]}
    assert "submit_required" not in reply
    r1 = rec.read("rounds")[1]
    assert r1["decision"] == "accept"
    assert r1["candidate"]["dev_score"] == deg7["score"]  # the workspace is that snapshot
    assert validate_ref(sandbox.repo, branch) == []


def test_the_best_of_the_improving_candidates_is_submitted(sandbox):
    script = {"executor": {"1": [WRITE_CANDIDATES, EVALUATE_BOTH,  # both beat the starter
                                 {"tool_calls": [call("finish", summary="two candidates")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"max_rounds": 1})
    rounds = rec.read("rounds")
    cubic, deg7 = [e for e in rec.read("experiments")
                   if e["kind"] == "eval-dev" and e["phase"] == "executor"]
    assert deg7["score"] < cubic["score"] < rounds[0]["candidate"]["dev_score"]
    reply = batch_reply(sandbox)
    assert reply["submit_required"] and "candidate deg7" in reply["message"]
    r1 = rounds[1]
    assert r1["phases"]["executor"]["summary"].startswith("[submitted after dev improvement")
    assert r1["decision"] == "accept" and r1["candidate"]["dev_score"] == deg7["score"]
    assert (wt / SOLVER).read_text() == poly(7)  # the candidate replaced the starter
    assert validate_ref(sandbox.repo, branch) == []


def interrupted_before_submit(sandbox, monkeypatch, script: dict, seed: int, **protocol):
    """A new base-mode run, started until the orchestrator would submit an improving dev
    evaluation: its branch, its worktree and the exp_id the live submit rule picked."""
    import pytest

    from conftest import write_script
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.runs import new_run

    class Interrupted(Exception):
        pass

    picked = []

    def interrupted(self, task, snapshot):  # the orchestrator stops before it submits
        picked.append(snapshot.parent.name)  # <artifact_dir>/snapshot of the submitted exp_id
        raise Interrupted

    path = write_script(sandbox, script, f"interrupted-{seed}.json")
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="base", seed=seed,
                         model_params={"script": str(path)}, protocol_overrides=protocol,
                         repo=sandbox.repo, log=lambda *a: None)
    with monkeypatch.context() as patch:
        patch.setattr(RunEngine, "restore_workspace", interrupted)
        with pytest.raises(Interrupted):
            RunEngine(wt, log=lambda *a: None).start()
    assert len(picked) == 1
    return branch, wt, picked[0]


def test_close_partial_round_submits_the_candidate_the_live_rule_picked(sandbox, monkeypatch):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    script = {"executor": {"1": [WRITE_CANDIDATES, EVALUATE_BOTH,
                                 {"tool_calls": [call("finish", summary="two candidates")]}]}}
    branch, wt, picked = interrupted_before_submit(sandbox, monkeypatch, script, seed=12,
                                                   max_rounds=1)
    assert picked == "toy-regression-r001-dev001-deg7"
    assert "deg=" not in (wt / SOLVER).read_text()  # the workspace still holds the starter
    RunEngine(wt, log=lambda *a: None).start(close_partial_round=True)
    rec = Records(wt)
    deg7 = next(e for e in rec.read("experiments") if e["exp_id"] == picked)
    r1 = rec.read("rounds")[1]
    assert r1["decision"] == "accept" and r1["phases"]["executor"]["status"] == "finished"
    assert r1["candidate"]["dev_score"] == deg7["score"]
    assert (wt / SOLVER).read_text() == poly(7)
    assert validate_ref(sandbox.repo, branch) == []


def test_close_partial_round_takes_its_bar_from_the_reference_row_only(sandbox, monkeypatch):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    variant = ".rsib/scratch/devref"
    named_devref = [{"id": "devref", "path": variant}]
    script = {"executor": {
        "1": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},  # no dev score
              {"tool_calls": [call("finish", summary="cubic")]}],
        "2": [{"tool_calls": [call("write_file", path=f"{variant}/solver.py", content=poly(7))]},
              {"tool_calls": [call("evaluate_dev", candidates=named_devref)]},
              {"tool_calls": [call("finish", summary="degree 7")]}]}}
    branch, wt, picked = interrupted_before_submit(sandbox, monkeypatch, script, seed=13,
                                                   max_rounds=2)
    rows = {e["exp_id"]: e for e in Records(wt).read("experiments")}
    ref, cand = rows["toy-regression-r002-devref"], rows[picked]  # the bar of round 2, measured
    assert picked.endswith("-devref") and cand["score"] < ref["score"]
    RunEngine(wt, log=lambda *a: None).start(close_partial_round=True)
    r2 = Records(wt).read("rounds")[2]
    assert r2["phases"]["executor"]["summary"].startswith(
        f"[submitted after dev improvement: {cand['score']:.6g} > {ref['score']:.6g}]")
    assert r2["decision"] == "accept" and r2["candidate"]["dev_score"] == cand["score"]
    assert (wt / SOLVER).read_text() == poly(7)
    assert validate_ref(sandbox.repo, branch) == []


def test_scratch_content_never_reaches_git_nor_counts_as_a_change(sandbox):
    script = {"executor": {
        "1": [{"tool_calls": [call("write_file", path=".rsib/scratch/a/solver.py", content=poly(3)),
                              call("bash", command="head -c 6000000 /dev/zero > "
                                                   "../../.rsib/scratch/big.bin")]}],
        "2": [{"tool_calls": [call("bash", command="ls -A ../../.rsib/scratch")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"submit_on_dev_improvement": False, "max_rounds": 2})
    rounds = rec.read("rounds")[1:]
    assert [r["decision"] for r in rounds] == ["no-change", "no-change"]
    assert not any(r["candidate"]["changed"] for r in rounds)
    assert rec.read("violations") == []  # not a path violation, not a large file
    committed = gitops.git(["log", "--name-only", "--format=",
                            f"{rec.run()['base']['commit']}..HEAD"], wt)
    assert committed.strip() and ".rsib" not in committed
    assert not tool_results(sandbox, "executor")[0]["is_error"]  # the executor may write there
    assert tool_results(sandbox, "executor", rnd=2)[0]["content"] == "[exit 0]\n"  # emptied
    assert validate_ref(sandbox.repo, branch) == []


def test_bare_model_names_default_to_cli_access(sandbox, monkeypatch):
    from rsibench.records import Records
    from rsibench.runs import new_run
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    logs = []
    _, wt = new_run(["toy-regression"], "claude-opus-5-5", repo=sandbox.repo, log=logs.append)
    model = Records(wt).run()["model"]
    assert (model["spec"], model["auth"], model["usd_metering"]) == \
        ("claude-cli:claude-opus-5-5", "cli", True)
    _, wt = new_run(["toy-regression"], "claude-opus-5-5", auth="api", seed=1, repo=sandbox.repo,
                    log=logs.append)
    model = Records(wt).run()["model"]
    assert (model["adapter"], model["auth"]) == ("anthropic", "api")
    assert any("ANTHROPIC_API_KEY is not in the environment" in m for m in logs)


def test_runs_branch_from_the_newest_release_of_the_checked_out_branch(sandbox):
    from rsibench.records import Records
    from rsibench.runs import new_run
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    gitops.git(["switch", "-q", "-c", "lab"], sandbox.repo)
    note = sandbox.repo / "tasks" / "toy-regression" / "instruction.md"
    note.write_text(note.read_text() + "\nlab change\n")
    gitops.git(["commit", "-qam", "lab task change"], sandbox.repo, env=env)

    def base(seed, **kw):
        logs = []
        _, wt = new_run(["toy-regression"], "mock:finish", seed=seed, repo=sandbox.repo,
                        log=logs.append, **kw)
        return Records(wt).run()["base"]["ref"], " ".join(logs)

    ref, logs = base(1)
    assert ref == "harness-v0.0.1" and "1 commit(s) after harness-v0.0.1" in logs
    gitops.git(["tag", "harness-v0.0.1-lab.1"], sandbox.repo)
    ref, logs = base(2)
    assert ref == "harness-v0.0.1-lab.1" and "commit(s) after" not in logs
    ref, logs = base(3, base_ref="lab")
    assert ref == "lab" and "is a branch" in logs
    gitops.git(["switch", "-q", "main"], sandbox.repo)
    assert base(4)[0] == "harness-v0.0.1"  # the lab release is not part of main


def test_a_release_without_newer_run_constants_keeps_its_protocol_id(sandbox):
    import tomllib

    from rsibench.records import Records
    from rsibench.report import protocol_id
    from rsibench.runs import new_run
    proto = sandbox.repo / "protocol" / "default.toml"
    proto.write_text("".join(line for line in proto.read_text().splitlines(keepends=True)
                             if not line.startswith(("exact_context", "max_eval_candidates"))))
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    gitops.git(["commit", "-qam", "a release without the newer [run] keys"], sandbox.repo, env=env)
    gitops.git(["tag", "harness-v0.0.2"], sandbox.repo)

    def run(seed, **overrides):
        _, wt = new_run(["toy-regression"], "mock:finish", seed=seed, repo=sandbox.repo,
                        protocol_overrides=overrides, log=lambda *a: None)
        return Records(wt).run()

    plain, exact = run(1), run(2, exact_context=True)
    with open(proto, "rb") as f:
        assert plain["protocol"] == tomllib.load(f)["run"]  # as that release freezes it
    assert exact["protocol"] == {**plain["protocol"], "exact_context": True}
    assert protocol_id(exact) != protocol_id(plain)


def test_run_branches_can_live_in_a_namespace(sandbox, monkeypatch):
    import pytest

    from rsibench.orchestrator.engine import RunEngine
    from rsibench.runs import new_run
    branch, wt = new_run(["toy-regression"], "mock:finish", mode="base", namespace="robo",
                         protocol_overrides={"max_rounds": 1}, repo=sandbox.repo,
                         log=lambda *a: None)
    assert branch.startswith("robo/run/toy-regression/mock-finish/base/")
    RunEngine(wt, log=lambda *a: None).start()
    assert gitops.ref_exists(sandbox.repo, f"refs/tags/{branch}/final")
    assert validate_ref(sandbox.repo, branch) == []
    monkeypatch.setenv("RSIB_RUN_NAMESPACE", "lab")
    branch, _ = new_run(["toy-regression"], "mock:finish", seed=2, repo=sandbox.repo,
                        log=lambda *a: None)
    assert branch.startswith("lab/run/")
    gitops.git(["branch", "team"], sandbox.repo)  # a branch named like the namespace blocks it
    with pytest.raises(ValueError, match="rename it"):
        new_run(["toy-regression"], "mock:finish", seed=3, namespace="team", repo=sandbox.repo,
                log=lambda *a: None)


DIAGNOSTIC_EVALUATOR = """
reward["metrics"] = {"by_half": {"left": mse, "right": mse}}
reward["info"] = {"split": os.environ["RSIB_SPLIT"]}
(out_dir / "residuals.txt").write_text(os.environ["RSIB_SPLIT"] + " " + " ".join(f"{e:.3f}" for e in errors[:5]))
reward["artifacts"] = [{"path": "residuals.txt", "role": "residuals"}]
(out_dir / "reward.json").write_text(json.dumps(reward))
"""


def test_dev_results_reach_the_agent_and_hidden_results_stay_private(sandbox):
    from conftest import write_script
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run
    from rsibench.util import slugify
    evaluator = sandbox.repo / "tasks" / "toy-regression" / "tests" / "evaluate.py"
    evaluator.write_text(evaluator.read_text() + DIAGNOSTIC_EVALUATOR)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    gitops.git(["commit", "-qam", "diagnostic evaluator"], sandbox.repo, env=env)
    gitops.git(["tag", "harness-v0.0.2"], sandbox.repo)
    script = sandbox.tmp / "diagnostic.json"
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="full",
                         model_params={"script": str(script)}, protocol_overrides={"max_rounds": 1},
                         repo=sandbox.repo, log=lambda *a: None)
    # the first dev evaluation of round 1 writes it, under paths.artifacts of the context
    residuals = (sandbox.tmp / "artifacts" / slugify(branch.replace("/", "--")) / "toy-regression"
                 / "r001" / "toy-regression-r001-dev001" / "residuals.txt")
    read = call("read_file", path=str(residuals))
    write_script(sandbox, {
        "executor": {"1": [{"tool_calls": [call("evaluate_dev")]}, {"tool_calls": [read]},
                           {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
                           {"tool_calls": [call("evaluate_dev")]},  # beats the starter -> submit
                           {"tool_calls": [call("finish", summary="cubic")]}]},
        "improver": {"1": [{"tool_calls": [read]}]}}, script.name)
    RunEngine(wt, log=lambda *a: None).start()
    rec = Records(wt)

    # dev: the full Result, including nested metrics and a readable artifact, reaches the agent
    transcript = next((sandbox.tmp / "artifacts").rglob("r001-executor.jsonl")).read_text()
    assert "by_half" in transcript and "residuals.txt" in transcript
    for phase in ("executor", "improver"):  # both read the artifact with their file tools
        got = [r for r in tool_results(sandbox, phase) if "residuals.txt (lines" in r["content"]]
        assert got and not got[0]["is_error"] and "1  dev " in got[0]["content"]
    # the reply of the improving evaluation holds the whole Result and the submit fields
    reply = next(json.loads(r["content"]) for r in tool_results(sandbox, "executor")
                 if '"submit_required"' in r["content"])
    assert reply["submit_required"] is True and "beats the accepted solution" in reply["message"]
    assert set(reply["metrics"]["by_half"]) == {"left", "right"}
    assert reply["info"] == {"split": "dev"} and reply["artifacts"][0]["role"] == "residuals"
    artifact = Path(reply["artifacts"][0]["path"])
    assert artifact.parent == Path(reply["artifact_dir"]).resolve()
    assert artifact.read_text().startswith("dev")
    assert rec.read("rounds")[1]["candidate"]["dev_score"] == reply["score"]  # submitted state
    dev = next(e for e in rec.read("experiments") if e["kind"] == "eval-dev")
    assert dev["artifact_dir"] and "by_half" not in dev["metrics"]  # records keep scalars
    assert (Path(dev["artifact_dir"]) / "residuals.txt").read_text().startswith("dev")

    # val and test: records hold the public view; the files sit in the private store
    for e in rec.read("experiments"):
        if e["kind"] in ("eval-val", "eval-test"):
            assert e["artifact_dir"] is None and e["log_path"] is None
            assert set(e["metrics"]) == {"mse", "max_abs_err", "n"}
    assert all(set(r["val"]["metrics"]) <= {"mse", "max_abs_err", "n"} for r in rec.read("rounds"))
    private = list((sandbox.tmp / "runs" / "_private").rglob("residuals.txt"))
    assert {p.read_text().split()[0] for p in private} == {"val", "test"}
    visible = [p for p in (sandbox.tmp / "artifacts").rglob("residuals.txt")]
    assert visible and all(p.read_text().startswith("dev") for p in visible)
    assert validate_ref(sandbox.repo, branch) == []


def test_public_metrics_limit_what_val_and_test_records_keep(sandbox):
    toml = sandbox.repo / "tasks" / "toy-regression" / "task.toml"
    toml.write_text(toml.read_text().replace("secondary = [", 'public = ["mse"]\nsecondary = ['))
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    gitops.git(["commit", "-qam", "only mse is public"], sandbox.repo, env=env)
    gitops.git(["tag", "harness-v0.0.2"], sandbox.repo)
    script = {"executor": {"1": [
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]},
        {"tool_calls": [call("evaluate_dev")]},
        {"tool_calls": [call("finish", summary="cubic")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"max_rounds": 1, "posthoc_at_end": False})
    assert len(posthoc(wt, log=lambda *a: None)) == 2
    exps = rec.read("experiments")
    hidden = ([r["val"] for r in rec.read("rounds")] + [f["test"] for f in rec.read("final")]
              + [e for e in exps if e["kind"] in ("eval-val", "eval-test")] + rec.read("posthoc"))
    assert len(hidden) == 8 and all(h["metrics"] == {"mse": h["score"]} for h in hidden)
    dev = [e for e in exps if e["kind"] == "eval-dev"]  # dev results keep every scalar metric
    assert dev and all(set(e["metrics"]) == {"mse", "max_abs_err", "n"} for e in dev)
    assert validate_ref(sandbox.repo, branch) == []


LEAKY_EVALUATOR = """
if os.environ["RSIB_SPLIT"] != "dev":  # a malformed reserved key holding labels of the split
    reward["info"] = [f"label={y:.4f}" for y in labels[:5]]
    (out_dir / "reward.json").write_text(json.dumps(reward))
"""


def test_a_malformed_hidden_result_keeps_its_values_out_of_the_records(sandbox):
    evaluator = sandbox.repo / "tasks" / "toy-regression" / "tests" / "evaluate.py"
    evaluator.write_text(evaluator.read_text() + LEAKY_EVALUATOR)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    gitops.git(["commit", "-qam", "malformed val and test results"], sandbox.repo, env=env)
    gitops.git(["tag", "harness-v0.0.2"], sandbox.repo)
    script = {"executor": {"1": [{"tool_calls": [call("write_file", path=SOLVER, content=poly(3))]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="full",
                                 protocol={"max_rounds": 1})
    problem = 'invalid reward.json: info: breaks type "object"'  # where, not what
    assert [r["decision_reason"] for r in rec.read("rounds")] == \
        [f"baseline evaluation failed: {problem}", f"validation failed: {problem}"]
    ctx = wt / ".rsib" / "ctx-improver.json"
    assert problem in json.loads(ctx.read_text())["last_round"]["decision_reason"]
    seen = [wt / "records" / "rounds.jsonl", ctx,
            next((sandbox.tmp / "artifacts").rglob("r001-improver.jsonl"))]
    assert not any("label=" in path.read_text() for path in seen)
    assert rec.read("final")[0]["test"]["status"] == "error"
    assert validate_ref(sandbox.repo, branch) == []


# writes a valid 1x1 PNG; no base64 in the command, so none can reach the transcript through it
PNG_CMD = ("python3 -c 'import struct,zlib; c=lambda t,d: struct.pack(\">I\",len(d))+t+d"
           "+struct.pack(\">I\",zlib.crc32(t+d)); open(\"plot.png\",\"wb\").write("
           "b\"\\x89PNG\\r\\n\\x1a\\n\"+c(b\"IHDR\",struct.pack(\">IIBBBBB\",1,1,8,2,0,0,0))"
           "+c(b\"IDAT\",zlib.compress(b\"\\x00\\xff\\x00\\x00\"))+c(b\"IEND\",b\"\"))'")


def test_images_reach_the_model_and_transcripts_keep_them_as_files(sandbox, monkeypatch):
    import base64
    import hashlib

    from rsibench.models.mock import ScriptedModel
    received, complete = [], ScriptedModel.complete

    def recording(self, system, messages, tools, max_tokens, meta):
        received.append(((meta["phase"], meta["round"]), messages))
        return complete(self, system, messages, tools, max_tokens, meta)

    monkeypatch.setattr(ScriptedModel, "complete", recording)
    script = {"executor": {"1": [
        {"tool_calls": [call("bash", command=PNG_CMD)]},
        {"tool_calls": [call("view_image", path="workspace/toy-regression/plot.png"),
                        call("view_image", path="/etc/hostname")]},
        {"tool_calls": [call("finish", summary="looked at the plot")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"exact_context": True})  # the mock never lowers
    assert rec.run()["protocol"]["exact_context"] is True
    assert rec.read("rounds")[1]["phases"]["executor"]["status"] == "finished"
    path = next((sandbox.tmp / "artifacts").rglob("r001-executor.jsonl"))
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    results = entries[2]["new_messages"][-1]["results"]
    image = results[0]["content"][1]
    assert results[0]["content"][0]["text"].startswith("workspace/toy-regression/plot.png (")
    assert results[1]["is_error"] and "outside the readable area" in results[1]["content"]
    media = path.parent / image["file"]
    data = media.read_bytes()
    assert image == {"type": "image", "mime": "image/png", "bytes": len(data),
                     "file": f"media/{hashlib.sha256(data).hexdigest()[:16]}.png"}
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    text = path.read_text()
    assert base64.b64encode(data).decode() not in text and "iVBORw0KGgo" not in text
    executor = [messages for where, messages in received if where == ("executor", 1)]
    seen = executor[2][-1]["results"][0]["content"]  # the model got the image inline, in place
    assert seen[1] == {"type": "image", "mime": "image/png",
                       "data": base64.b64encode(data).decode()}
    assert seen[0] == results[0]["content"][0]
    assert all(e["lowerings"] == [] for e in entries)
    usage = rec.read("usage")
    assert usage and all(u["lowerings"] == [] for u in usage)
    assert validate_ref(sandbox.repo, branch) == []


def test_gateway_checks_messages_and_refuses_inexact_context(tmp_path):
    import base64
    import urllib.error
    import urllib.request

    from conftest import stub_services

    from rsibench.models import ChatModel, ModelResponse, context
    from rsibench.orchestrator.gateway import Gateway

    received = []

    class ImagesFirst(ChatModel):  # an adapter that lowers every context holding an image
        adapter = "fake"

        def lowerings(self, system, messages, tools, meta=None):
            return ["images_first"] if context.images_in(messages) else []

        def complete(self, system, messages, tools, max_tokens, meta):
            received.append(messages)
            return ModelResponse(text="ok", tool_calls=[], stop_reason="end_turn", raw=None,
                                 model="fake")

    services, token, usage = stub_services(tmp_path, ImagesFirst("fake"), exact_context=True)
    gateway = Gateway(services).start()

    def post(messages, session):
        body = {"system": "s", "messages": messages, "tools": [], "session": session}
        req = urllib.request.Request(gateway.url + "/v1/complete", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(8)).decode()
    image = {"type": "image", "mime": "image/png", "data": png}
    reply = {"role": "assistant", "text": "ok", "tool_calls": [], "raw": None}
    first = [{"role": "user", "content": "text needs no lowering"}]
    grown = [*first, reply, {"role": "user", "content": "more"}]
    edited = [{"role": "user", "content": "edited text"}, *grown[1:], reply,
              {"role": "user", "content": "again"}]  # longer, but its first message changed
    try:
        for bad in ({"role": "user", "content": "a message, not a list"},
                    [{"role": "user", "content": [{"type": "image", "mime": "image/png"}]}],
                    [{"role": "user", "content": [{**image, "mime": "image/jpeg"}]}],
                    [{"role": "user", "content": [{**image, "data": "not base64!"}]}]):
            status, body = post(bad, "bad")
            assert (status, body["error"]) == (400, "bad_request"), body
        status, body = post([{"role": "user", "content": [image]}], "exact")
        assert (status, body["error"]) == (400, "inexact_context")
        assert "images_first" in body["message"] and usage == []  # refused before the model
        assert [post(m, "text")[0] for m in (first, grown, edited)] == [200, 200, 200]
        services.engine.protocol["exact_context"] = False
        assert post([{"role": "user", "content": [image]}], "lowered")[0] == 200
    finally:
        gateway.stop()
    # the adapter gets the messages as sent: text unchanged, the image inline and in place
    assert received == [first, grown, edited, [{"role": "user", "content": [image]}]]
    assert [row["lowerings"] for row in usage] == [[], [], [], ["images_first"]]
    transcript = (tmp_path / "transcripts" / "executor.jsonl").read_text()
    assert png not in transcript and '"lowerings": ["images_first"]' in transcript
    assert len(list((tmp_path / "transcripts" / "media").iterdir())) == 1
    entries = [json.loads(line) for line in transcript.splitlines()][:3]
    assert [e["new_messages"] for e in entries] == [first, grown[2:], edited]
    assert [e.get("history_edited", False) for e in entries] == [False, False, True]


# The status a token gets from /v1/budget. `--watch OUT`: leave the process group (setsid), then
# note each change of status of this process's token in OUT until it stops working. Otherwise:
# print the status of each token given.
PROBE = '''
import os, sys, time, urllib.error, urllib.request


def status(token):
    req = urllib.request.Request(os.environ["RSIB_GATEWAY_URL"] + "/v1/budget", data=b"{}",
                                 headers={"Authorization": "Bearer " + token})
    try:
        return urllib.request.urlopen(req, timeout=10).status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return 0


if sys.argv[1] == "--watch":
    os.setsid()
    last, deadline = None, time.time() + 60
    while last in (None, 200) and time.time() < deadline:
        got = status(os.environ["RSIB_GATEWAY_TOKEN"])
        if got != last:
            with open(sys.argv[2], "a") as f:
                f.write(f"{got} ")
        last = got
        time.sleep(0.05)
else:
    print(*(status(token) for token in sys.argv[1:]))
'''


def probe(sandbox) -> str:
    """The command that runs PROBE."""
    path = sandbox.tmp / "probe.py"
    path.write_text(PROBE)
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(path))}"


def test_a_process_that_outlives_its_phase_holds_a_dead_token(sandbox):
    from conftest import running
    seen, plain = sandbox.tmp / "statuses.txt", sandbox.tmp / "plain.pid"
    out, saved = shlex.quote(str(seen)), shlex.quote(str(sandbox.tmp / "executor-token"))
    # a plain leftover keeps the phase's marker and is stopped with the phase; the watcher drops
    # the marker, outlives the phase, and only its revoked token stops it
    spawn = (f"printf %s \"$RSIB_GATEWAY_TOKEN\" > {saved}; "
             f"sleep 300 > /dev/null 2>&1 < /dev/null & echo $! > {shlex.quote(str(plain))}; "
             f"env -u RSIB_SANDBOX_TOKEN {probe(sandbox)} --watch {out} > /dev/null 2>&1 "
             f"< /dev/null & for i in $(seq 200); do [ -s {out} ] && break; sleep 0.05; done")
    # the next phase tries the executor's token, then keeps the gateway up until the escaped
    # process noted its refusal
    check = (f"{probe(sandbox)} \"$(cat {saved})\" \"$RSIB_GATEWAY_TOKEN\"; "
             f"for i in $(seq 200); do grep -q 401 {out} && break; sleep 0.05; done")
    script = {"executor": {"1": [{"tool_calls": [call("bash", command=spawn)]}]},
              "improver": {"1": [{"tool_calls": [call("bash", command=check)]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="full",
                                 protocol={"max_rounds": 1})
    # refused as soon as the phase ended, while the next phase's own token works
    assert tool_results(sandbox, "improver")[0]["content"].split()[-2:] == ["401", "200"]
    assert seen.read_text().split() == ["200", "401"]  # served during the phase, refused after
    assert not running(int(plain.read_text()))


def test_an_experiment_calls_the_model_with_a_grant_of_its_own(sandbox):
    tokens = shlex.quote(str(sandbox.tmp / "experiment-token"))
    ask = (f"printf %s \"$RSIB_GATEWAY_TOKEN\" > {tokens} && "
           "python3 -c \"import json, os, urllib.request as r; "
           "body = json.dumps({'system': 's', 'messages': [{'role': 'user', 'content': 'hi'}], "
           "'tools': []}).encode(); "
           "req = r.Request(os.environ['RSIB_GATEWAY_URL'] + '/v1/complete', data=body, "
           "headers={'Authorization': 'Bearer ' + os.environ['RSIB_GATEWAY_TOKEN']}); "
           "reply = json.loads(r.urlopen(req, timeout=60).read()); "
           "json.dump({'model': reply['model']}, open(os.environ['RSIB_METRICS_FILE'], 'w'))\"")
    check = f"{probe(sandbox)} \"$(cat {tokens})\" \"$RSIB_GATEWAY_TOKEN\""
    script = {"executor": {"1": [
        {"tool_calls": [call("run_experiment", name="ask", command=ask)]},
        {"tool_calls": [call("bash", command=check)]},
        {"tool_calls": [call("finish", summary="asked the model")]}]}}
    branch, wt, rec = run_to_end(sandbox, ["toy-regression"], script, mode="base",
                                 protocol={"max_rounds": 1})
    exp = next(e for e in rec.read("experiments") if e["kind"] == "experiment")
    assert exp["status"] == "ok" and exp["metrics"] == {"model": "mock-scripted"}
    usage = rec.read("usage")
    asked = [u for u in usage if u["phase"] == "experiment"]
    assert len(asked) == 1 and asked[0]["round"] == 1 and "charged" not in asked[0]
    # the round's cost holds every call made in it, the job's too
    assert rec.read("rounds")[1]["cost"]["usd"] == pytest.approx(
        sum(u["usd"] for u in usage if u["round"] == 1), abs=1e-5)
    transcript = Path(exp["artifact_dir"]) / "transcripts" / "experiment.jsonl"
    assert len(transcript.read_text().splitlines()) == 1
    # the job's token ended with the job, while the phase's own token still worked
    assert tool_results(sandbox, "executor")[1]["content"].split()[-2:] == ["401", "200"]
    assert validate_ref(sandbox.repo, branch) == []


# appended to the starter solver and to the executor's: one model call per evaluation, when the
# evaluator hands the solver the gateway; the solver keeps its token in the file TOKENS
CALLS_THE_MODEL = '''

import json as _json
import os as _os
import urllib.request as _request

_fit_predict = fit_predict


def fit_predict(train_x, train_y, query_x):
    if "RSIB_GATEWAY_URL" in _os.environ:
        body = {"system": "s", "messages": [{"role": "user", "content": "a hint?"}], "tools": []}
        token = _os.environ["RSIB_GATEWAY_TOKEN"]
        with open(TOKENS, "a") as kept:
            kept.write(token + " ")
        req = _request.Request(_os.environ["RSIB_GATEWAY_URL"] + "/v1/complete",
                               data=_json.dumps(body).encode(),
                               headers={"Authorization": "Bearer " + token})
        _request.urlopen(req, timeout=60).read()
    return _fit_predict(train_x, train_y, query_x)
'''
# the evaluator passes the gateway variables it gets on to the solver, and notes their names
GATEWAY_TO_SOLVER = ('"PYTHONDONTWRITEBYTECODE": "1", **{k: v for k, v in os.environ.items() '
                     'if k.startswith("RSIB_GATEWAY_")}}')
GATEWAY_SEEN = '''
(out_dir / "gateway.txt").write_text(" ".join(sorted(k for k in os.environ
                                                     if k.startswith("RSIB_GATEWAY_"))))
'''


def model_access_run(sandbox, access: bool):
    """A one-round run of toy-regression whose solvers call the model when their evaluator hands
    them the gateway; the task declares model access if `access`. After its dev evaluation, the
    executor tries the tokens the solvers kept so far."""
    task = sandbox.repo / "tasks" / "toy-regression"
    evaluator = task / "tests" / "evaluate.py"
    evaluator.write_text(evaluator.read_text().replace('"PYTHONDONTWRITEBYTECODE": "1"}',
                                                       GATEWAY_TO_SOLVER) + GATEWAY_SEEN)
    tokens = sandbox.tmp / "eval-tokens"
    calls = CALLS_THE_MODEL.replace("TOKENS", repr(str(tokens)))
    starter = task / "environment" / "workspace" / "solver.py"
    starter.write_text(starter.read_text() + calls)
    if access:
        toml = task / "task.toml"
        toml.write_text(toml.read_text().replace("eval_gpus = 0", "eval_gpus = 0\n"
                                                                  "model_access = true"))
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    gitops.git(["commit", "-qam", "solvers that call the model"], sandbox.repo, env=env)
    gitops.git(["tag", "harness-v0.0.2"], sandbox.repo)
    kept = f"{probe(sandbox)} $(cat {shlex.quote(str(tokens))} 2> /dev/null)"
    script = {"executor": {"1": [
        {"tool_calls": [call("evaluate_dev")]},
        {"tool_calls": [call("bash", command=kept)]},
        {"tool_calls": [call("write_file", path=SOLVER, content=poly(3) + calls)]},
        {"tool_calls": [call("finish", summary="cubic")]}]}}
    return run_to_end(sandbox, ["toy-regression"], script, mode="base",
                      protocol={"max_rounds": 1, "posthoc_at_end": False})


def test_evaluations_of_a_task_with_model_access_are_metered_and_only_dev_is_charged(sandbox):
    from rsibench.cli import main
    from rsibench.orchestrator.engine import RunEngine
    branch, wt, rec = model_access_run(sandbox, access=True)
    rounds, usage = rec.read("rounds"), rec.read("usage")
    assert [r["decision"] for r in rounds] == ["baseline", "accept"]
    # every evaluation made its call; val and test are metered, not charged
    assert sorted((u["phase"], u["round"], u.get("charged", True)) for u in usage
                  if u["phase"] != "executor") == [
        ("eval-dev", 0, True), ("eval-dev", 1, True),  # the reference, the executor's
        ("eval-test", 1, False), ("eval-val", 0, False), ("eval-val", 1, False)]
    for r in rounds:  # a round's cost holds exactly its charged calls
        charged = [u["usd"] for u in usage if u["round"] == r["round"] and u.get("charged", True)]
        assert r["cost"]["usd"] == pytest.approx(sum(charged), abs=1e-5)
    # resuming rebuilds the spend of an interrupted round from the charged calls only
    engine = RunEngine(wt, log=lambda *a: None)
    after_baseline = engine.partial_usage(engine.tasks[0], 0, rounds[0]["ended_at"])
    assert after_baseline.usd == pytest.approx(sum(
        u["usd"] for u in usage if u["round"] > 0 and u.get("charged", True)), abs=1e-5)
    assert summarize(wt / "records")["tasks"][0]["eval_model"]["calls"] == 3
    # each evaluator got the gateway; dev transcripts sit with the agent's outputs, the val and
    # test ones in the private store
    private, artifacts = sandbox.tmp / "runs" / "_private", sandbox.tmp / "artifacts"
    seen = [p.read_text() for p in sandbox.tmp.rglob("gateway.txt")]
    assert len(seen) == 5 and set(seen) == {"RSIB_GATEWAY_TOKEN RSIB_GATEWAY_URL"}
    assert sorted(p.name for p in private.rglob("transcripts/*.jsonl")) == \
        ["eval-test.jsonl", "eval-val.jsonl", "eval-val.jsonl"]
    assert sorted(p.name for p in artifacts.rglob("transcripts/eval-*.jsonl")) == \
        ["eval-dev.jsonl", "eval-dev.jsonl"]
    # a token a solver kept is dead once its evaluation ended: those of the baseline val, the
    # accepted solution's dev and the executor's dev evaluation, tried while the gateway served
    assert tool_results(sandbox, "executor")[1]["content"].split()[2:] == ["401"] * 3
    assert validate_ref(sandbox.repo, branch) == []

    # rsib posthoc and rsib run finalize serve the gateway themselves
    site = sandbox.tmp / "site.toml"
    site.write_text("")
    assert main(["posthoc", str(wt), "--site", str(site)]) == 0
    assert [(p["round"], p["status"]) for p in rec.read("posthoc")] == [(0, "ok"), (1, "ok")]
    assert [(u["round"], u["charged"]) for u in rec.read("usage") if u["phase"] == "posthoc"] \
        == [(0, False), (1, False)]
    assert len(list(private.rglob("transcripts/posthoc.jsonl"))) == 2
    good = rec.read("final")[0]
    rec.append("final", dict(good, test={"status": "error", "score": None, "metrics": {}}))
    gitops.commit(wt, "simulated failed test evaluation",
                  {"Rsib-Run": branch, "Rsib-Scope": "final"}, ["records"])
    assert main(["run", "finalize", str(wt), "--site", str(site)]) == 0
    assert [f["test"]["status"] for f in rec.read("final")] == ["ok", "error", "ok"]
    assert [u["phase"] for u in rec.read("usage")].count("eval-test") == 2
    assert validate_ref(sandbox.repo, branch) == []


def test_a_task_without_model_access_gives_its_evaluator_no_gateway(sandbox):
    branch, wt, rec = model_access_run(sandbox, access=False)
    assert [r["decision"] for r in rec.read("rounds")] == ["baseline", "accept"]
    seen = [p.read_text() for p in sandbox.tmp.rglob("gateway.txt")]
    assert len(seen) == 5 and set(seen) == {""}
    assert {u["phase"] for u in rec.read("usage")} == {"executor"}
    assert summarize(wt / "records")["tasks"][0]["eval_model"] == \
        {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}


def test_cpu_jobs_and_phase_limits_follow_the_site(sandbox):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.runs import new_run
    _, wt = new_run(["toy-regression"], "mock:finish", mode="base", repo=sandbox.repo,
                    log=lambda *a: None)
    site = {"slurm": {"cpu_partition": "cpu_short"}, "phase": {"cpus": 1, "threads": 2}}
    engine = RunEngine(wt, backend="slurm", site=site, log=lambda *a: None)
    assert engine.cpu_jobs == "slurm" and engine.cpu_backend is engine.backend
    plain = RunEngine(wt, backend="slurm", site={"slurm": {}}, log=lambda *a: None)
    assert plain.cpu_jobs == "local" and plain.cpu_backend is not plain.backend
    assert RunEngine(wt, log=lambda *a: None).cpu_jobs == "local"
    sys.path.insert(0, str(wt / "harness"))
    try:
        from executor.phase import _environment_note
    finally:
        sys.path.remove(str(wt / "harness"))
    note = _environment_note({"sandboxed": False, "cpu_jobs": "slurm", "phase_cpus": 1})
    assert "run_experiment" in note and "1 CPU cores" in note and "CPU node" in note
    assert _environment_note({"sandboxed": False, "cpu_jobs": "local"}) == ""


def test_the_gateway_serves_where_the_site_says_and_only_when_needed(sandbox):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.runs import new_run
    _, wt = new_run(["toy-regression"], "mock:finish", mode="base", repo=sandbox.repo,
                    log=lambda *a: None)
    engine = RunEngine(wt, site={"gateway": {"advertise": "localhost"}}, log=lambda *a: None)
    with engine.serving(engine.tasks):  # no task gives its evaluator model access
        assert engine.gateway is None
    with engine.serving():
        gateway = engine.gateway
        assert gateway.server.server_address[0] == "127.0.0.1"  # the default bind address
        assert gateway.url.startswith("http://localhost:")
        with engine.serving(engine.tasks), engine.serving():  # re-entrant
            assert engine.gateway is gateway
        assert engine.gateway is gateway
        reply = subprocess.run([*shlex.split(probe(sandbox)), "no-such-token"], text=True,
                               capture_output=True, env={"RSIB_GATEWAY_URL": gateway.url})
        assert reply.stdout.split() == ["401"]  # reachable at the advertised name
    assert engine.gateway is None
    wide = RunEngine(wt, site={"gateway": {"host": "0.0.0.0"}}, log=lambda *a: None)
    with wide.serving():  # on every interface, the URL names this machine
        host, port = wide.gateway.server.server_address[:2]
        assert (host, wide.gateway.url) == ("0.0.0.0", f"http://{socket.gethostname()}:{port}")
