"""Run a task's evaluator on one split of one workspace snapshot.

Evaluator contract (tests/test.sh, run with bash):
    RSIB_SPLIT       dev | val | test
    RSIB_DATA_DIR    data of that split (inputs + labels; only the evaluator sees labels)
    RSIB_TRAIN_DIR   dev split (training data the solution may use)
    RSIB_WORKSPACE   the solution to score (treat as read-only)
    RSIB_OUTPUT_DIR  write reward.json here: {"<primary metric>": float, ...} plus, optionally,
                     "metrics", "artifacts" and "info" (the Result contract, see contracts.py)
    RSIB_TASK_DIR    the task package (private copy)
    RSIB_PYTHON      python interpreter to use ("python" inside a container image)
    RSIB_SANDBOX     JSON list, possibly empty: a launcher prefix for the solution's process.
                     Run the solution as  prefix + ["--rw", io_dir, "--"] + command  so it sees
                     its snapshot, the run cache and io_dir only (never the labels).
    RSIB_GATEWAY_URL, RSIB_GATEWAY_TOKEN
                     only for tasks with [rsi.resources] model_access: the gateway and a token
                     that allows /v1/complete until the evaluation ends; hand them to the solution

Good evaluators run the solution in a child process that only sees the split's inputs, then
score its predictions in the parent (see tests/fixtures/tasks/toy-regression/tests/evaluate.py).

`run_eval` returns a Result ({"status", "score", "metrics", "artifacts", "info", "detail"}) plus
job fields (wall_s, gpu_hours, exit_code, job_id, backend, log_path, log_tail). Callers choose the
output directory: under the artifact root for dev, under the private root for val and test.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from ..compute import ComputeBackend, Job
from ..contracts import HIDDEN_SPLITS, parse_result
from ..envs import scrubbed_env
from ..tasks import Task
from ..util import finite_or_none, read_json, tail


def run_eval(task: Task, workspace: Path, split: str, out_dir: Path, backend: ComputeBackend,
             data_root: Path | None = None, timeout_s: float | None = None,
             container: str | None = None, sandbox=None,
             cancelled: Callable[[], bool] | None = None,
             gateway: tuple[str, str] | None = None,
             on_queue: Callable[[float, float | None], None] | None = None) -> dict:
    """`gateway`: the (url, token) of the evaluation's own grant, for tasks with model access.
    `on_queue`: told the job's queue wait while it lasts (Job.on_queue)."""
    url, token = gateway or (None, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_path = out_dir / "reward.json"
    if reward_path.exists():
        reward_path.unlink()
    env = scrubbed_env({
        "RSIB_SPLIT": split,
        "RSIB_DATA_DIR": task.split_dir(split, data_root),
        "RSIB_TRAIN_DIR": task.split_dir("dev", data_root),
        "RSIB_WORKSPACE": workspace,
        "RSIB_OUTPUT_DIR": out_dir,
        "RSIB_TASK_DIR": task.root,
        "RSIB_PYTHON": sys.executable,  # a backend that runs the job in an image sets "python"
        "RSIB_GATEWAY_URL": url, "RSIB_GATEWAY_TOKEN": token,  # None: not set
    })
    mounts = None
    if sandbox is not None and sandbox.enabled:
        from ..sandbox import python_roots
        launcher_python = "python3" if container else sys.executable
        env["RSIB_SANDBOX"] = sandbox.child_prefix_json(
            workspace, launcher_python, extra_ro=None if container else python_roots())
        if container:
            mounts = sandbox.container_mounts(
                ro=[task.root, task.split_dir(split, data_root), task.split_dir("dev", data_root),
                    workspace, sandbox.dir], rw=[out_dir])
    else:
        env["RSIB_SANDBOX"] = ""
    log_path = out_dir / "eval.log"
    job = Job(name=f"eval-{task.id}-{split}", command=f"bash {shlex.quote(str(task.evaluator))}",
              cwd=out_dir, env=env, gpus=task.eval_gpus,
              timeout_s=timeout_s or task.eval_timeout_s, log_path=log_path, container=container,
              mounts=mounts, cancelled=cancelled, on_queue=on_queue)
    res = backend.run(job)
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    result = {"status": "error", "score": None, "metrics": {}, "artifacts": [], "info": {},
              "detail": "", "wall_s": res.wall_s, "queued": res.queued, "queue_s": res.queue_s,
              "gpu_hours": res.gpu_hours, "exit_code": res.exit_code, "job_id": res.job_id,
              "backend": backend.name, "log_path": str(log_path), "log_tail": tail(log_text, 3000)}
    if res.status != "ok":
        result["detail"] = f"evaluator {res.status} (exit {res.exit_code}) {res.detail}".strip()
        return result
    try:
        raw = read_json(reward_path, default=None)
    except ValueError:  # truncated or not JSON: an error result keeps the job's cost
        result["detail"] = "reward.json is not valid JSON"
        return result
    if raw is None:
        result["detail"] = "evaluator wrote no reward.json"
        return result
    # the detail of a val or test result reaches the records and the improver's prompt
    fields, problems = parse_result(raw, out_dir, hidden=split in HIDDEN_SPLITS)
    result.update(fields)
    if problems and not fields["metrics"]:
        result["detail"] = "invalid reward.json: " + "; ".join(problems)[:500]
        return result
    score = finite_or_none(fields["metrics"].get(task.primary_metric))
    if score is None:
        result["detail"] = f"reward.json has no finite '{task.primary_metric}'"
        return result
    result.update(status="ok", score=score, detail="; ".join(problems)[:500])
    return result
