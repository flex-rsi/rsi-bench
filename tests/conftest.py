"""Shared fixtures: a throw-away copy of this repository with its own git history, data, runs
and artifact roots, so tests never touch the real checkout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
# small tasks the tests run end to end; the sandbox repository gets them under tasks/
FIXTURE_TASKS = Path(__file__).resolve().parent / "fixtures" / "tasks"
IGNORE = shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache", "*.egg-info",
                                ".env")


@pytest.fixture(autouse=True)
def no_dotenv(tmp_path, monkeypatch):
    """Tests never read API keys from the user's .env files."""
    monkeypatch.setenv("RSIB_ENV_FILE", str(tmp_path / "no-such.env"))


def _git(args, cwd):
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    shutil.copytree(REPO, repo, ignore=IGNORE)
    for task in FIXTURE_TASKS.iterdir():
        shutil.copytree(task, repo / "tasks" / task.name, ignore=IGNORE)
    proto = repo / "protocol" / "default.toml"
    proto.write_text(proto.read_text().replace("max_rounds = 200", "max_rounds = 3"))
    imp = repo / "harness" / "improver" / "config.toml"
    imp.write_text(imp.read_text().replace("meta_every_rounds = 4", "meta_every_rounds = 2"))
    _git(["init", "-q", "-b", "main"], repo)
    # no automatic gc: a background `git gc --auto` after a commit prunes loose-object
    # directories while a test walks the tree (seen in CI as FileNotFoundError on .git/objects/xx)
    _git(["config", "gc.auto", "0"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "test base"], repo)
    _git(["tag", "harness-v0.0.1"], repo)
    for key, sub in (("RSIB_REPO", "repo"), ("RSIB_RUNS_ROOT", "runs"),
                     ("RSIB_DATA_ROOT", "data"), ("RSIB_ARTIFACT_ROOT", "artifacts")):
        monkeypatch.setenv(key, str(tmp_path / sub))
    monkeypatch.delenv("RSIB_SITE", raising=False)
    for task in ("toy-regression", "toy-segmentation"):
        subprocess.run([sys.executable, str(repo / "tasks" / task / "prepare.py"), "--out",
                        str(tmp_path / "data" / task)], check=True, capture_output=True)
    return SimpleNamespace(repo=repo, tmp=tmp_path)


def write_script(sandbox, script: dict, name: str = "script.json") -> Path:
    path = sandbox.tmp / name
    path.write_text(json.dumps(script))
    return path


def run_to_end(sandbox, tasks, script=None, mode="full", model="mock:scripted", budget=None,
               improver_from=None, executor_from=None, seed=1, protocol=None):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run
    params = {"script": str(write_script(sandbox, script, f"script-{mode}-{seed}.json"))} \
        if script is not None else {}
    branch, wt = new_run(tasks, model, mode=mode, seed=seed, model_params=params,
                         budget_overrides=budget, improver_from=improver_from,
                         executor_from=executor_from,
                         protocol_overrides=protocol, repo=sandbox.repo, log=lambda *a: None)
    RunEngine(wt, log=lambda *a: None).start()
    return branch, wt, Records(wt)


def stub_services(tmp_path, model, **protocol):
    """Gateway services over a stub engine (no run) with the grant of an executor phase (task `t`,
    round 1): the services, the grant's token and the usage rows they record. The grant's
    transcript is tmp_path/transcripts/executor.jsonl."""
    from rsibench.budget import Budget
    from rsibench.orchestrator.engine import PHASE_ENDPOINTS
    from rsibench.orchestrator.gateway import Services
    usage = []
    engine = SimpleNamespace(
        protocol={"max_tokens_per_call": 1000, **protocol}, model=model,
        records=SimpleNamespace(append=lambda name, row: usage.append(row)))
    services = Services(engine)
    token = services.grant("executor", SimpleNamespace(id="t"), 1, Budget(
        {"wall_hours": 1, "gpu_hours": 0, "usd": 1.0, "tokens": 10 ** 6, "val_max_calls": 1}),
        PHASE_ENDPOINTS["executor"], tmp_path / "transcripts" / "executor.jsonl")
    return services, token, usage


def eval_services(tmp_path, evaluate, dev_bar=None, dev_margin=0.0, **protocol):
    """Gateway services over a stub engine whose dev evaluation of a snapshot returns
    `evaluate(snapshot)` (run_eval fields), with an executor grant (task `t`, mse, round 1, bar
    `dev_bar`, submit margin `dev_margin`): the services, the grant's token, the experiments rows
    they record and the budget."""
    from rsibench.budget import Budget
    from rsibench.orchestrator.engine import PHASE_ENDPOINTS
    from rsibench.orchestrator.gateway import Services
    wt = (tmp_path / "wt").resolve()
    (wt / "workspace" / "t").mkdir(parents=True)
    (wt / ".rsib" / "scratch").mkdir(parents=True)
    rows = []

    def run_split_eval(task, snapshot, split, out_dir, rnd, budget=None, cancelled=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "ok", "score": None, "metrics": {}, "artifacts": [], "info": {},
                "detail": "", "wall_s": 0.1, "gpu_hours": 0.0, "exit_code": 0, "job_id": "1",
                "backend": "stub", "log_path": str(out_dir / "eval.log"), "log_tail": "",
                **evaluate(snapshot)}

    engine = SimpleNamespace(
        wt=wt, protocol={"submit_on_dev_improvement": True, **protocol},
        records=SimpleNamespace(append=lambda name, row: rows.append(row)),
        workspace_dir=lambda task_id: wt / "workspace" / task_id,
        scratch_dir=lambda: wt / ".rsib" / "scratch",
        artifact_dir=lambda task_id, rnd: tmp_path / "artifacts" / f"r{rnd:03d}",
        working_snapshot=lambda task, dest, src: Path(shutil.copytree(src, dest)),
        run_split_eval=run_split_eval)
    task = SimpleNamespace(id="t", primary_metric="mse", direction="minimize", eval_gpus=1,
                           better=lambda a, b: a < b)
    budget = Budget({"wall_hours": 1, "gpu_hours": 10, "usd": 1.0, "tokens": 10 ** 6,
                     "val_max_calls": 1})
    services = Services(engine)
    token = services.grant("executor", task, 1, budget, PHASE_ENDPOINTS["executor"],
                           tmp_path / "transcripts" / "executor.jsonl", bar=dev_bar,
                           margin=dev_margin)
    return services, token, rows, budget


# ---- solver sources used by scripted executors ---------------------------------------------

POLY_SOLVER = '''
def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_predict(train_x, train_y, query_x, deg=DEG):
    feats = lambda x: [x ** k for k in range(deg + 1)]
    X = [feats(x) for x in train_x]
    A = [[sum(r[i] * r[j] for r in X) for j in range(deg + 1)] for i in range(deg + 1)]
    b = [sum(r[i] * y for r, y in zip(X, train_y)) for i in range(deg + 1)]
    w = _solve(A, b)
    return [sum(wi * f for wi, f in zip(w, feats(x))) for x in query_x]
'''

CONSTANT_SOLVER = '''
def fit_predict(train_x, train_y, query_x):
    return [0.0 for _ in query_x]
'''


def poly(deg: int) -> str:
    return POLY_SOLVER.replace("DEG", str(deg))


def call(_tool, **arguments):
    return {"name": _tool, "arguments": arguments}


def running(pid: int, wait_s: float = 2.0) -> bool:
    """Whether process `pid` still runs after up to `wait_s` seconds (a zombie has ended)."""
    deadline = time.monotonic() + wait_s
    while True:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True).stdout.strip()
        if not state or state.startswith("Z"):
            return False
        if time.monotonic() > deadline:
            return True
        time.sleep(0.05)


def linked_project(tmp_path):
    """A venv reached through a link to a whole project tree (as /shared/a/... -> /shared/b/...
    on a cluster), whose interpreter lives elsewhere; the tree also holds held-out data."""
    real = tmp_path / "fs12" / "projects"
    venv = real / "me" / "venv"
    (venv / "bin").mkdir(parents=True)
    base = tmp_path / "py" / "cpython-3.11.15"
    (base / "bin").mkdir(parents=True)
    (base / "bin" / "python3.11").write_text("interpreter\n")
    (venv / "bin" / "python").symlink_to(base / "bin" / "python3.11")
    (real / "me" / "val").mkdir()
    (real / "me" / "val" / "labels.json").write_text("labels")
    (tmp_path / "fsw").symlink_to(real)
    return tmp_path / "fsw" / "me" / "venv", venv, base
