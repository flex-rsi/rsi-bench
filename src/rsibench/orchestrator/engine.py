"""The round lifecycle. This is the frozen anchor: the agent can rewrite the harness, never this.

Per task:
    round 0   score the starter workspace on val (baseline), tag <branch>/<task>/r0
    round k   [meta]      stream mode, first round of every task after the first (mode=full)
              executor    harness edits workspace/<task>/ using dev feedback and experiments;
                          with submit_on_dev_improvement, the first dev evaluation that beats
                          the accepted solution ends it and its evaluated state is the candidate
              enforce     out-of-scope edits are reverted and logged as violations
              val         the candidate is scored on the hidden validation split
              accept      the harness acceptance rule (I) decides; default rule in base mode
              commit      solution commit (+ revert when rejected), tag accepted rounds
              improver    harness edits harness/executor, harness/core (E) and, in mode=full,
                          harness/improver (I); not run in base mode
              [meta]      every `meta_every_rounds` rounds (mode=full)
              records     one line in rounds.jsonl, committed
    end       the task stops on budget, max_rounds, exhausted val calls or max_idle_rounds idle
              rounds in a row; tag <branch>/<task>/final
After all tasks the final commits are scored once on the test split (records/final.jsonl). Test
scores are never visible to the agent during the run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import tomllib
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from .. import gitops
from ..budget import Budget, Usage
from ..compute import make_backend
from ..contracts import HIDDEN_SPLITS, public_view, scalars
from ..envs import scrubbed_env
from ..models import load_model
from ..paths import (archive_root, artifact_root, data_root, private_root, repo_root, runs_root,
                     scratch_root)
from ..protocol import run_constant
from ..records import Records
from ..phase_limits import PhaseLimits
from ..sandbox import DEFAULT_MAX_PROCS, GitViewError, Sandbox, kill_leftovers, python_roots
from ..tasks import Task, load_task_dir
from ..util import kill_group, now_iso, read_json, sha256_tree, slugify, tail, write_json
from .evaluator import run_eval
from .gateway import (Gateway, Services, auto_submit_at, best_improvement, file_entries,
                      submit_margin)
from .policy import allowed_scopes, classify, default_accept

# The executor's scratch directory, relative to the worktree. `.rsib/` is git-ignored and scope
# enforcement skips it, so nothing in it is committed or counts as a change.
SCRATCH = ".rsib/scratch"
# What each harness phase's grant allows; the accept phase gets no token.
PHASE_ENDPOINTS = {"executor": ("complete", "experiment", "evaluate", "budget"),
                   "improver": ("complete", "budget"), "meta": ("complete", "budget")}


@dataclass
class TaskState:
    last_round: int = -1
    best_val: float | None = None
    baseline_val: float | None = None
    accepted_round: int | None = None
    val_calls: int = 0
    cumulative: Usage = field(default_factory=Usage)
    last_good_harness: str | None = None
    crash_streak: int = 0
    idle_streak: int = 0
    accepted_dev: float | None = None   # dev score of the accepted solution (submit rule bar)


@dataclass
class PhaseOutcome:
    status: str
    steps: int = 0
    summary: str = ""
    error: str = ""
    data: dict = field(default_factory=dict)
    submit: dict | None = None  # the executor's submission (the submit rule)
    warning: str = ""  # an orchestrator problem the phase ran despite (e.g. no Git view)

    def record(self) -> dict:
        out = {"status": self.status, "steps": int(self.steps), "summary": self.summary[:2000]}
        if self.error:
            out["error"] = self.error[:2000]
        if self.warning:
            out["warning"] = self.warning[:2000]
        return out


class RunEngine:
    def __init__(self, worktree: Path, backend: str = "local", site: dict | None = None,
                 push_remote: str | None = None, log=None):
        self.wt = Path(worktree).resolve()
        self.records = Records(self.wt)
        # The run's paths (scratch, Git views, artifacts, private store) and its sandbox come from
        # run.json, which a phase can edit: use the committed file, and the branch the worktree's
        # Git metadata names, which agents never reach.
        restored = self.records.restore_run(gitops.show(self.wt, "HEAD", "records/run.json"))
        self.run = self.records.run()
        self.branch = gitops.current_branch(self.wt)
        if self.run["branch"] != self.branch:
            raise RuntimeError(f"{self.wt} is on branch {self.branch}, but its committed "
                               f"records/run.json names {self.run['branch']}")
        self.slug = slugify(self.branch.replace("/", "--"))
        self.mode = self.run["mode"]
        self.protocol = self.run["protocol"]
        self.push_remote = push_remote
        self.site = site or {}
        self.log = log or _log_stderr
        if restored:
            self.log("records/run.json differed from its committed version; restored it")
        self.data_root = data_root()
        self.art_root = artifact_root() / self.slug
        self.private = private_root() / self.slug
        self.tasks = [self._load_private_task(t) for t in self.run["tasks"]]
        m = self.run["model"]
        self.model = load_model(m["spec"], m.get("effort"), m.get("params"))
        self.backend = make_backend(backend, site)
        # CPU-only experiments and evaluations: on SLURM when the site names a `cpu_partition`,
        # else on this machine
        self.cpu_jobs = "local" if backend == "local" else (
            backend if ((site or {}).get(backend) or {}).get("cpu_partition") else "local")
        self.cpu_backend = self.backend if self.cpu_jobs == backend else make_backend("local", site)
        self.phase_limits = PhaseLimits(site)
        self.containers = dict(((site or {}).get("slurm") or {}).get("containers") or {})
        self.services = Services(self)
        self.budgets: dict[str, Budget] = {}  # the live budget of each task being run
        self.gateway: Gateway | None = None
        self.gateway_site = dict((site or {}).get("gateway") or {})
        self._large_reported: set[str] = set()
        self.submit_rule = bool(self.protocol.get("submit_on_dev_improvement", False))
        self._close: dict | None = None
        self.sandbox = Sandbox(self.run.get("sandbox"), scratch_root() / self.slug,
                               [runs_root(), self.data_root, artifact_root(), private_root(),
                                scratch_root(), archive_root(), repo_root(), self.wt],
                               max_procs=(self.site.get("sandbox") or {}).get("max_procs",
                                                                              DEFAULT_MAX_PROCS))
        self.sandbox.prepare()

    # ---- paths -------------------------------------------------------------------------------
    def _load_private_task(self, entry: dict) -> Task:
        root = self.private / "tasks" / entry["id"]
        task = load_task_dir(root)
        if task.version_hash() != entry["version_hash"]:
            raise RuntimeError(f"private copy of task {entry['id']} does not match run.json; "
                               f"recreate the run or restore {root}")
        return task

    def workspace_dir(self, task_id: str) -> Path:
        return self.wt / "workspace" / task_id

    def scratch_dir(self) -> Path:
        return self.wt / SCRATCH

    def artifact_dir(self, task_id: str, rnd: int) -> Path:
        path = self.art_root / task_id / f"r{rnd:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def hidden_eval_dir(self, task_id: str, name: str) -> Path:
        """Output of a val or test evaluation: in the private store, never shown to the agent."""
        return self.private / "evals" / task_id / name

    def transcript_path(self, task_id: str, rnd: int, phase: str) -> Path:
        path = self.art_root / task_id / "transcripts" / f"r{rnd:03d}-{phase}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def budget_limits(self, task: Task) -> dict:
        return self.run["budgets"][task.id]

    # ---- evaluation --------------------------------------------------------------------------
    def container_for(self, task: Task, backend) -> str | None:
        """Image for a task's jobs on the SLURM backend (local jobs never use containers)."""
        if backend.name != "slurm" or not task.container:
            return None
        if task.container not in self.containers:
            raise RuntimeError(f"task {task.id} needs container '{task.container}'; map it in the "
                               f"site profile under [slurm.containers]")
        return self.containers[task.container]

    def run_split_eval(self, task: Task, workspace: Path, split: str, out_dir: Path, rnd: int,
                       budget: Budget | None = None, consumer: str | None = None,
                       cancelled=None) -> dict:
        """Score `workspace` on `split`. The evaluator of a task with model access holds a grant
        until the evaluation ends: consumer `eval-<split>` unless named, round `rnd`, charged to
        `budget` (None: metered only), capped at the task's `eval_model_calls_max`, transcript
        under `out_dir`. Such an evaluation waits while the model is known to refuse calls, and is
        run again (twice at most) if the model became unavailable while it ran: its calls may have
        timed out, so its result says nothing about the solution."""
        for attempt in range(3):
            outages = self.model.outages
            if task.model_access:
                self.wait_for_model(budget or self.live_budget(task.id), f"a {split} evaluation")
            result = self._split_eval_once(task, workspace, split, out_dir, rnd, budget, consumer,
                                           cancelled)
            if not task.model_access or self.model.outages == outages or attempt == 2 \
                    or (cancelled is not None and cancelled()):
                return result
            self.log(f"[{task.id}] r{rnd}: the model became unavailable during a {split} "
                     f"evaluation ({result['status']}, {result['score']}); evaluating again")
        return result

    def _split_eval_once(self, task: Task, workspace: Path, split: str, out_dir: Path, rnd: int,
                         budget: Budget | None, consumer: str | None, cancelled) -> dict:
        backend = self.backend if task.eval_gpus else self.cpu_backend
        container = self.container_for(task, backend)
        token = None
        if task.model_access:
            if self.gateway is None:
                raise RuntimeError(f"task {task.id} has model access: evaluate it in serving()")
            consumer = consumer or f"eval-{split}"
            token = self.services.grant(consumer, task, rnd, budget, ("complete",),
                                        out_dir / "transcripts" / f"{consumer}.jsonl",
                                        call_cap=task.eval_model_calls)
        # the queue wait is left out of the wall time of the task being run while it lasts
        waits = budget or self.live_budget(task.id)
        try:
            with waits.waiting() if waits is not None else nullcontext() as on_queue:
                return run_eval(task, workspace, split, out_dir, backend, data_root=self.data_root,
                                container=container, sandbox=self.sandbox, cancelled=cancelled,
                                gateway=(self.gateway.url, token) if token else None,
                                on_queue=on_queue)
        finally:
            self.services.revoke(token)

    def live_budget(self, task_id: str) -> Budget | None:
        """The budget of a task being run (None outside its rounds)."""
        return self.budgets.get(task_id)

    def wait_for_model(self, budget: Budget | None, what: str) -> None:
        """Hold off while the model is known to refuse calls (a provider's usage limit). The wait
        is left out of the wall-time budget."""
        if not self.model.blocked_until():
            return
        start = time.monotonic()
        self.log(f"the model is unavailable (provider limit) for about "
                 f"{self.model.blocked_until() - start:.0f}s; waiting before {what}")
        while until := self.model.blocked_until():
            time.sleep(min(60.0, max(1.0, until - time.monotonic())))
        if budget is not None:
            budget.add_wait(start, time.monotonic())

    @contextmanager
    def serving(self, tasks: list[Task] | None = None):
        """Keep the gateway serving inside the block; with `tasks`, only if one of them has model
        access. Re-entrant: a gateway that is already serving stays up."""
        needed = tasks is None or any(t.model_access for t in tasks)
        if self.gateway is not None or not needed:
            yield
            return
        self.gateway = Gateway(self.services, host=self.gateway_site.get("host", "127.0.0.1"),
                               advertise=self.gateway_site.get("advertise")).start()
        try:
            yield
        finally:
            self.gateway.stop()
            self.gateway = None

    def _record_eval(self, task: Task, rnd: int, phase: str, kind: str, result: dict,
                     out_dir: Path, started: str, exp_id: str | None = None) -> None:
        """Records keep the public view of a val/test evaluation; its files stay private."""
        hidden = kind.removeprefix("eval-") in HIDDEN_SPLITS
        public = task.public_metrics if hidden else None
        self.records.append("experiments", {
            "exp_id": exp_id or f"{task.id}-r{rnd:03d}-{kind}", "task_id": task.id, "round": rnd,
            "phase": phase, "kind": kind, "name": kind, "command": "", "gpus": task.eval_gpus,
            "backend": result["backend"], "job_id": result["job_id"], "started_at": started,
            "ended_at": now_iso(), "wall_s": round(result["wall_s"], 3),
            "queue_s": round(float(result.get("queue_s") or 0), 3),
            "gpu_hours": round(result["gpu_hours"], 6), "exit_code": result["exit_code"],
            "status": result["status"], "score": result["score"],
            "metrics": scalars(result["metrics"], public),
            "log_path": None if hidden else result["log_path"],
            "artifact_dir": None if hidden else str(out_dir),
        })

    def register_artifacts(self, task_id: str, rnd: int, root: Path, kind: str) -> None:
        for entry in file_entries(root):
            self.records.append("artifacts", {"task_id": task_id, "round": rnd, "kind": kind,
                                              **entry})

    # ---- harness phases ----------------------------------------------------------------------
    def harness_config(self, part: str) -> dict:
        path = self.wt / "harness" / part / "config.toml"
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def history(self, task: Task) -> list[dict]:
        rows = []
        for r in self.records.read("rounds"):
            if r["task_id"] != task.id:
                continue
            ex = (r.get("phases") or {}).get("executor") or {}
            rows.append({
                "round": r["round"], "decision": r["decision"], "val": r["val"].get("score"),
                "best_val_after": r.get("best_val_after"),
                "executor_status": ex.get("status"), "executor_summary": ex.get("summary", "")[:600],
                "harness_changes": {k: (v or {}).get("lines", 0)
                                    for k, v in (r.get("harness_changes") or {}).items()},
                "budget_used_frac": r.get("budget_used_frac"),
            })
        return rows

    def phase_context(self, phase: str, task: Task, rnd: int, budget: Budget, state: TaskState,
                      extra: dict | None = None) -> dict:
        ctx = {
            "phase": phase, "round": rnd, "mode": self.mode, "run_id": self.run["run_id"],
            "worktree": str(self.wt),
            "task": {
                "id": task.id, "domain": task.domain, "instruction": task.instruction,
                "metric": task.primary_metric, "direction": task.direction,
                "noise_sigma": task.noise_sigma,
                "anchors": {"baseline": task.baseline, "reference": task.reference},
                "workspace": f"workspace/{task.id}",
                "experiment_gpus_max": task.experiment_gpus_max,
                "model_access": task.model_access,
            },
            "paths": {
                "notes": f"records/notes/{task.id}.md",
                "artifacts": str(self.art_root / task.id),
                "lessons": "harness/improver/memory/lessons.md",
                "records": "records",
                "transcripts": str(self.transcript_path(task.id, rnd, phase).parent),
            },
            "state": {"best_val": state.best_val, "baseline_val": state.baseline_val,
                      "accepted_round": state.accepted_round, "val_calls_used": state.val_calls,
                      "val_calls_max": int(budget.limits["val_max_calls"]),
                      "accepted_dev": state.accepted_dev,
                      "auto_submit_at": auto_submit_at(task, state.accepted_dev,
                                                       submit_margin(task, self.protocol))
                      if self.submit_rule else None},
            "environment": {"sandboxed": self.sandbox.enabled,
                            "cache": str(self.sandbox.cache) if self.sandbox.enabled else None,
                            "git_history": self.sandbox.git_view,
                            "cpu_jobs": self.cpu_jobs, **self.phase_limits.describe()},
            "rules": {"submit_on_dev_improvement": self.submit_rule,
                      "max_model_calls_per_phase": int(self.protocol.get("max_model_calls_per_phase", 0) or 0),
                      "submit_grace_calls": int(self.protocol.get("submit_grace_calls", 3)),
                      "submit_min_gain": float(run_constant(self.protocol, "submit_min_gain")),
                      "unfinished_submit": str(run_constant(self.protocol, "unfinished_submit")),
                      "max_rounds": int(self.protocol["max_rounds"])},
            "history": self.history(task),
            "budget": {"remaining": budget.remaining(), "fractions": budget.fractions(),
                       "limits": budget.limits},
            "allow": {"edit_improver": "I" in allowed_scopes(phase, self.mode)
                      if phase != "accept" else False},
            "limits": {"max_harness_diff_lines": int(self.protocol["max_harness_diff_lines"])},
            "stream": {"index": [t.id for t in self.tasks].index(task.id),
                       "tasks": [t.id for t in self.tasks]},
        }
        if phase == "executor":
            ctx["task"]["dev_data"] = str(task.split_dir("dev", self.data_root))
            ctx["paths"]["scratch"] = SCRATCH
        ctx.update(extra or {})
        return ctx

    def reset_scratch(self) -> None:
        """Give the executor phase an empty scratch directory."""
        scratch = self.scratch_dir()
        if scratch.is_symlink() or scratch.is_file():
            scratch.unlink()
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)

    def run_phase(self, phase: str, task: Task, rnd: int, budget: Budget, state: TaskState,
                  extra: dict | None = None, timeout_s: float | None = None) -> PhaseOutcome:
        """Run a harness phase. A phase that calls the model starts once the model takes calls,
        and one that stopped with a model error because the model became unavailable (a usage
        limit) is run again when it is back, twice at most, unless it already submitted."""
        for attempt in range(3):
            outages = self.model.outages
            if phase in PHASE_ENDPOINTS:
                self.wait_for_model(budget, f"the {phase} phase")
            out = self._run_phase_once(phase, task, rnd, budget, state, extra, timeout_s)
            if not (phase in PHASE_ENDPOINTS and out.status == "model_error"
                    and self.model.outages > outages and out.submit is None and attempt < 2):
                return out
            self.log(f"[{task.id}] r{rnd}: the {phase} phase stopped because the model became "
                     f"unavailable; running it again once the model is back")
        return out

    def _run_phase_once(self, phase: str, task: Task, rnd: int, budget: Budget, state: TaskState,
                  extra: dict | None = None, timeout_s: float | None = None) -> PhaseOutcome:
        rsib_dir = self.wt / ".rsib"
        rsib_dir.mkdir(exist_ok=True)
        ctx_path, out_path = rsib_dir / f"ctx-{phase}.json", rsib_dir / f"out-{phase}.json"
        out_path.unlink(missing_ok=True)
        if phase == "executor":
            self.reset_scratch()
        context = self.phase_context(phase, task, rnd, budget, state, extra)
        write_json(ctx_path, context)
        env = scrubbed_env({
            "RSIB_PHASE": phase, "RSIB_ROUND": rnd, "RSIB_TASK": task.id,
            "RSIB_WORKTREE": self.wt,
            "RSIB_TRANSCRIPT_DIR": self.transcript_path(task.id, rnd, phase).parent,
        })
        if phase == "executor":
            env["RSIB_DEV_DATA"] = str(task.split_dir("dev", self.data_root))
        marker = os.urandom(8).hex()  # finds the phase's processes that left its group
        env["RSIB_SANDBOX_TOKEN"] = marker
        cmd = [sys.executable, "harness/run_phase.py", phase, "--context", str(ctx_path),
               "--output", str(out_path)]
        env.update(self.phase_limits.env())
        limit = float(self.protocol["phase_timeout_min"]) * 60
        remaining_wall = budget.remaining()["wall_s"]
        timeout = timeout_s or max(30.0, min(limit, remaining_wall + 60))
        log_path = self.artifact_dir(task.id, rnd) / f"{phase}.log"
        token = None
        if phase in PHASE_ENDPOINTS:
            bar = state.accepted_dev if (phase == "executor" and self.submit_rule) else None
            margin = submit_margin(task, self.protocol) if bar is not None else 0.0
            token = self.services.grant(phase, task, rnd, budget, PHASE_ENDPOINTS[phase],
                                        self.transcript_path(task.id, rnd, phase), bar=bar,
                                        margin=margin)
            env.update({"RSIB_GATEWAY_URL": self.gateway.url, "RSIB_GATEWAY_TOKEN": token})
        status, error, warning = "ok", "", ""
        resources = ExitStack()
        try:
            if self.sandbox.enabled:
                ro = python_roots() + [task.split_dir("dev", self.data_root)]
                mask, ro_files = [], []
                if self.sandbox.git_view:
                    # Git finds the view through <wt>/.git only, so git elsewhere (clone, -C)
                    # behaves as usual; GIT_* would redirect every git command of the phase.
                    env = {k: v for k, v in env.items() if not k.startswith("GIT_")}
                    try:
                        git_dir, gitfile = resources.enter_context(
                            self.sandbox.phase_git(self.wt, self.branch))
                        ro.append(git_dir)
                        ro_files.append((gitfile, self.wt / ".git"))
                    except (GitViewError, OSError) as exc:
                        warning = f"ran without the Git history view: {exc}"
                        self.log(f"[{task.id}] r{rnd}: the {phase} phase {warning}")
                        if not (self.wt / ".git").is_symlink():  # a mask would follow a link
                            mask.append(self.wt / ".git")
                        context["environment"]["git_history"] = False
                        write_json(ctx_path, context)
                env.update(self.sandbox.env())
                cmd = self.sandbox.prefix(ro, [self.wt, self.art_root, self.sandbox.dir],
                                          cwd=self.wt, mask=mask, ro_files=ro_files) + cmd
            cmd = self.phase_limits.prefix() + cmd
            with open(log_path, "w") as log:
                proc = subprocess.Popen(cmd, cwd=str(self.wt), env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                deadline, waited = time.monotonic() + timeout, budget.used().queue_s
                while True:  # time the model was unavailable or jobs were queued extends it
                    try:
                        code = proc.wait(timeout=5.0)
                        break
                    except subprocess.TimeoutExpired:
                        if self.model.blocked_until() or \
                                time.monotonic() < deadline + budget.used().queue_s - waited:
                            continue
                        code, status = None, "timeout"
                        break
                # what the phase left running would change the workspace after it is scored
                kill_group(proc.pid)
                proc.wait()
        finally:
            # a process that escaped the group (setsid) holds a dead token from here on
            grant = self.services.revoke(token)
            left = kill_leftovers(marker)
            if left:
                self.log(f"[{task.id}] r{rnd}: stopped {left} process(es) left behind by the "
                         f"{phase} phase")
            resources.close()
        out = read_json(out_path, default=None)
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        if status != "timeout":
            if isinstance(out, dict) and out.get("status"):
                status = str(out["status"])
            elif code != 0:
                status = "crash"
        if status == "crash":
            error = (out or {}).get("error") or tail(log_text, 1500)
        valid = {"ok", "finished", "stopped", "crash", "timeout", "budget", "refusal",
                 "model_error", "skipped"}
        if status not in valid:
            status = "ok"
        out = out if isinstance(out, dict) else {}
        return PhaseOutcome(status=status, steps=int(out.get("steps") or 0),
                            summary=str(out.get("summary") or ""), error=error, data=out,
                            submit=grant.submit if grant else None, warning=warning)

    # ---- scope enforcement -------------------------------------------------------------------
    def enforce(self, phase: str, task: Task, rnd: int) -> tuple[dict, int]:
        allowed = allowed_scopes(phase, self.mode)
        kept: dict[str, list[str]] = {"solution": [], "E": [], "I": [], "notes": []}
        bad: list[str] = []
        violations = 0
        tampered = self.records.tampered()
        if tampered:
            self.records.repair()
            self._violation(task, rnd, phase, "path", tampered,
                            "record files are written by the orchestrator only; restored")
            violations += 1
        owned = set(self.records.owned_paths())
        for ch in gitops.changes(self.wt):
            if ch.path.startswith(".rsib/") or ch.path in owned:
                continue
            scope = classify(ch.path, task.id)
            (kept[scope] if scope in allowed else bad).append(ch.path)
        if bad:
            gitops.restore(self.wt, bad)
            self._violation(task, rnd, phase, "path", bad,
                            f"{phase} may only change {sorted(allowed) or 'nothing'}; reverted")
            violations += 1
        max_bytes = float(self.protocol["max_file_mb"]) * (1 << 20)
        for scope, paths in kept.items():
            small = []
            for p in paths:
                full = self.wt / p
                if full.is_file() and full.stat().st_size > max_bytes:
                    if p not in self._large_reported:
                        self._large_reported.add(p)
                        self.records.append("artifacts", {
                            "task_id": task.id, "round": rnd, "kind": "large_file", "path": str(full),
                            "bytes": full.stat().st_size, "sha256": None,
                            "note": "left in the worktree, not committed"})
                        self._violation(task, rnd, phase, "large_file", [p],
                                        f"file exceeds {self.protocol['max_file_mb']} MB; not committed")
                        violations += 1
                else:
                    small.append(p)
            kept[scope] = small
        if phase in ("improver", "meta"):
            harness_paths = kept["E"] + kept["I"]
            lines = gitops.numstat(self.wt, harness_paths)
            if lines > int(self.protocol["max_harness_diff_lines"]):
                gitops.restore(self.wt, harness_paths)
                self._violation(task, rnd, phase, "diff_size", harness_paths,
                                f"{lines} changed lines > {self.protocol['max_harness_diff_lines']}; reverted")
                kept["E"], kept["I"] = [], []
                violations += 1
        return kept, violations

    def _pending_snapshot(self) -> dict[str, bytes | None]:
        """Uncommitted paths and their current bytes (None for deleted files)."""
        out = {}
        owned = set(self.records.owned_paths())
        for ch in gitops.changes(self.wt):
            if ch.path.startswith(".rsib/") or ch.path in owned:
                continue
            full = self.wt / ch.path
            out[ch.path] = full.read_bytes() if full.is_file() else None
        return out

    def _restore_unless_unchanged(self, saved: dict, task: Task, rnd: int, phase: str) -> int:
        """Undo anything a read-only phase changed, keeping the pending candidate intact."""
        bad = []
        owned = set(self.records.owned_paths())
        for ch in gitops.changes(self.wt):
            if ch.path.startswith(".rsib/") or ch.path in owned:
                continue
            full = self.wt / ch.path
            now = full.read_bytes() if full.is_file() else None
            if ch.path not in saved:
                gitops.restore(self.wt, [ch.path])
                bad.append(ch.path)
            elif now != saved[ch.path]:
                if saved[ch.path] is None:
                    full.unlink(missing_ok=True)
                else:
                    full.write_bytes(saved[ch.path])
                bad.append(ch.path)
        if bad:
            self._violation(task, rnd, phase, "path", bad, f"{phase} must not change files; reverted")
            return 1
        return 0

    def _violation(self, task: Task, rnd: int, phase: str, kind: str, paths: list[str],
                   detail: str) -> None:
        self.records.append("violations", {"task_id": task.id, "round": rnd, "phase": phase,
                                           "kind": kind, "paths": paths[:200], "detail": detail})

    def _trailers(self, task: Task, rnd: int, scope: str, **more) -> dict:
        t = {"Rsib-Run": self.branch, "Rsib-Task": task.id, "Rsib-Round": str(rnd),
             "Rsib-Scope": scope}
        t.update({f"Rsib-{k.replace('_', '-').title()}": str(v) for k, v in more.items()
                  if v is not None})
        return t

    def _commit_harness(self, task: Task, rnd: int, kept: dict, phase: str) -> dict:
        changes = {}
        for scope in ("E", "I"):
            lines = gitops.numstat(self.wt, kept[scope])
            sha = gitops.commit(self.wt, f"[{task.id} r{rnd}] {phase}: harness {scope} update",
                                self._trailers(task, rnd, scope, phase=phase), kept[scope])
            changes[scope] = {"commit": sha, "lines": lines if sha else 0}
        gitops.commit(self.wt, f"[{task.id} r{rnd}] {phase}: notes",
                      self._trailers(task, rnd, "notes", phase=phase), kept["notes"])
        return changes

    # ---- acceptance ----------------------------------------------------------------------------
    def decide(self, task: Task, rnd: int, score: float, budget: Budget, state: TaskState,
               ) -> tuple[bool, str, str, int]:
        sigma_k = float(self.protocol.get("default_accept_sigma_k", 2.0))
        min_rel = float(self.protocol.get("default_accept_min_rel", 0.0))
        if self.mode == "base":
            ok, why = default_accept(score, state.best_val, task.direction, task.noise_sigma,
                                     sigma_k, min_rel)
            return ok, why, "default", 0
        extra = {"accept_input": {
            "score": score, "best": state.best_val, "baseline": state.baseline_val,
            "direction": task.direction, "noise_sigma": task.noise_sigma,
            "history": self.history(task)}}
        saved = self._pending_snapshot()
        outcome = self.run_phase("accept", task, rnd, budget, state, extra, timeout_s=120)
        violations = self._restore_unless_unchanged(saved, task, rnd, "accept")
        decision = outcome.data.get("accept")
        if outcome.status in ("ok", "finished") and isinstance(decision, bool):
            return decision, str(outcome.data.get("reason", ""))[:500], "harness", violations
        ok, why = default_accept(score, state.best_val, task.direction, task.noise_sigma,
                                 sigma_k, min_rel)
        self._violation(task, rnd, "accept", "accept_fallback", [],
                        f"harness acceptance rule failed ({outcome.status}): "
                        f"{(outcome.error or '')[:300]}; used the default rule")
        return ok, why, "default", violations + 1

    # ---- rounds ------------------------------------------------------------------------------
    def baseline_round(self, task: Task, budget: Budget, state: TaskState) -> None:
        start, before = now_iso(), budget.used()
        out_dir = self.hidden_eval_dir(task.id, "r000-val")
        result = self.run_split_eval(task, self.workspace_dir(task.id), "val", out_dir, 0)
        if result.get("queued"):
            budget.add_wait(*result["queued"])
        self._record_eval(task, 0, "orchestrator", "eval-val", result, out_dir, start)
        state.baseline_val = state.best_val = result["score"]
        state.accepted_round = 0
        head = gitops.rev_parse(self.wt)
        state.last_good_harness = head
        candidate = {"changed": False}
        if self.submit_rule:
            state.accepted_dev = self.reference_dev(task, 0, budget, head)
            candidate["dev_score"] = state.accepted_dev
        self._append_round(task, 0, start, before, budget, phases={}, candidate=candidate,
                           val=result, decision="baseline",
                           reason="starter workspace" if result["status"] == "ok"
                           else f"baseline evaluation failed: {result.get('detail', '')}",
                           decided_by="orchestrator", best_before=None, accepted_commit=head,
                           harness_changes={}, violations=0, state=state)
        gitops.tag(self.wt, f"{self.branch}/{task.id}/r0", head)
        self.log(f"[{task.id}] baseline val = {result['score']} ({result['status']})")

    def one_round(self, task: Task, rnd: int, budget: Budget, state: TaskState,
                  first_of_later_task: bool, closing: dict | None = None) -> None:
        """One round. `closing` finishes an interrupted round instead of running its executor:
        {"executor": PhaseOutcome with its submission, if any, "before": Usage}."""
        start, before = now_iso(), budget.used()
        if closing:
            before = closing["before"]
        elif self.submit_rule and state.accepted_dev is None:
            state.accepted_dev = self.reference_dev(task, rnd, budget, gitops.rev_parse(self.wt))
        phases: dict = {}
        violations = 0
        harness_changes = {"E": {"commit": None, "lines": 0}, "I": {"commit": None, "lines": 0}}

        if first_of_later_task and self.mode == "full" and not closing:
            meta = self.run_phase("meta", task, rnd, budget, state, {"trigger": "task_boundary"})
            phases["meta"] = meta.record()
            kept, v = self.enforce("meta", task, rnd)
            violations += v
            harness_changes = _merge(harness_changes, self._commit_harness(task, rnd, kept, "meta"))

        harness_snapshot = gitops.rev_parse(self.wt)  # the harness this round actually runs
        if closing:
            ex = closing["executor"]
        else:
            ex = self.run_phase("executor", task, rnd, budget, state)
        submit = ex.submit
        unmeasured = False  # cut off with nothing a dev evaluation showed to be better
        if submit is None and self._submits_best_dev(ex, state):
            submit = self._best_dev_submission(task, rnd, state.accepted_dev)
            unmeasured = submit is None
        if submit and submit.get("unfinished"):
            self.restore_workspace(task, Path(submit["snapshot"]))
            ex.summary = (f"[stopped before finishing; its best dev evaluation is submitted: "
                          f"{submit['score']:.6g} > {submit['bar']:.6g}] " + ex.summary)
        elif submit:
            # Submit exactly the state whose dev evaluation improved on the accepted solution.
            self.restore_workspace(task, Path(submit["snapshot"]))
            if ex.status in ("budget", "model_error", "stopped"):
                ex.status = "finished"
            ex.summary = (f"[submitted after dev improvement: {submit['score']:.6g} > "
                          f"{submit['bar']:.6g}] " + ex.summary)
        phases["executor"] = ex.record()
        kept, v = self.enforce("executor", task, rnd)
        violations += v
        gitops.commit(self.wt, f"[{task.id} r{rnd}] executor: notes",
                      self._trailers(task, rnd, "notes", phase="executor"), kept["notes"])
        discarded = unmeasured and bool(kept["solution"])
        if discarded:
            gitops.restore(self.wt, kept["solution"])
            kept["solution"] = []

        best_before = state.best_val
        val = {"status": "skipped", "score": None, "metrics": {}}
        candidate = {"changed": bool(kept["solution"]), "commit": None,
                     "files_changed": len(kept["solution"])}
        if submit:
            candidate["dev_score"] = submit["score"]
        decision, reason, decided_by, accepted_commit = "no-change", "workspace unchanged", \
            "orchestrator", None
        if discarded:
            reason = (f"executor {ex.status} before finishing and no dev evaluation beat the "
                      "accepted solution: its unmeasured changes were discarded, no validation "
                      "call spent")
        if kept["solution"]:
            if state.val_calls >= int(budget.limits["val_max_calls"]):
                gitops.restore(self.wt, kept["solution"])
                val["status"] = "limit"
                decision, reason = "error", "validation call budget exhausted; candidate discarded"
                candidate["changed"] = False
            else:
                v_start = now_iso()
                out_dir = self.hidden_eval_dir(task.id, f"r{rnd:03d}-val")
                snapshot = self.working_snapshot(task, out_dir / "snapshot")
                if "dev_score" not in candidate:
                    candidate["dev_score"] = self.known_dev_score(task, rnd, sha256_tree(snapshot))
                val = self.run_split_eval(task, snapshot, "val", out_dir, rnd)
                self._record_eval(task, rnd, "orchestrator", "eval-val", val, out_dir, v_start)
                state.val_calls += 1
                budget.val_calls = state.val_calls
                budget.charge(gpu_hours=val["gpu_hours"])
                if val.get("queued"):
                    budget.add_wait(*val["queued"])
                if val["status"] != "ok":
                    accept, reason, decided_by = False, f"validation failed: {val.get('detail', '')}", \
                        "orchestrator"
                    decision = "reject"
                else:
                    accept, reason, decided_by, v = self.decide(task, rnd, val["score"], budget, state)
                    violations += v
                    decision = "accept" if accept else "reject"
                sha = gitops.commit(
                    self.wt, f"[{task.id} r{rnd}] candidate ({decision}, val={val['score']})",
                    self._trailers(task, rnd, "solution", decision=decision, val=val["score"],
                                   decided_by=decided_by),
                    kept["solution"], body=ex.summary[:2000])
                candidate["commit"] = sha
                if decision == "accept":
                    state.best_val = val["score"]
                    state.accepted_round = rnd
                    state.accepted_dev = candidate.get("dev_score")
                    accepted_commit = sha
                    gitops.tag(self.wt, f"{self.branch}/{task.id}/r{rnd}", sha)
                elif sha:
                    gitops.revert(self.wt, sha, self._trailers(task, rnd, "revert",
                                                               reverts=sha, decision="reject"))

        if self.mode != "base" and not budget.exhausted():
            imp_extra = {"last_round": {
                "executor": ex.record(), "val": {"status": val["status"], "score": val["score"]},
                "decision": decision, "decision_reason": reason,
                "transcript": str(self.transcript_path(task.id, rnd, "executor")),
                "experiments": [e for e in self.records.read("experiments")
                                if e["task_id"] == task.id and e["round"] == rnd],
            }}
            imp = self.run_phase("improver", task, rnd, budget, state, imp_extra)
            phases["improver"] = imp.record()
            kept, v = self.enforce("improver", task, rnd)
            violations += v
            harness_changes = _merge(harness_changes, self._commit_harness(task, rnd, kept, "improver"))
            every = int(self.harness_config("improver").get("meta_every_rounds", 0) or 0)
            if self.mode == "full" and every > 0 and rnd % every == 0 and not budget.exhausted():
                meta = self.run_phase("meta", task, rnd, budget, state, {"trigger": "periodic"})
                phases["meta"] = meta.record()
                kept, v = self.enforce("meta", task, rnd)
                violations += v
                harness_changes = _merge(harness_changes, self._commit_harness(task, rnd, kept, "meta"))

        violations += self._crash_guard(task, rnd, phases, state, harness_snapshot)
        self._append_round(task, rnd, start, before, budget, phases, candidate, val, decision,
                           reason, decided_by, best_before, accepted_commit, harness_changes,
                           violations, state)
        self.log(f"[{task.id}] r{rnd}: {decision} val={val['score']} best={state.best_val} "
                 f"E+{harness_changes['E']['lines']} I+{harness_changes['I']['lines']} "
                 f"budget={budget.fraction_used():.2f}")

    def _crash_guard(self, task: Task, rnd: int, phases: dict, state: TaskState,
                     snapshot: str) -> int:
        """Roll harness/ back after repeated crashes, to the last state that ran a clean round."""
        crashed = any((p or {}).get("status") == "crash" for p in phases.values())
        if not crashed:
            state.crash_streak = 0
            state.last_good_harness = snapshot
            return 0
        state.crash_streak += 1
        if state.crash_streak < int(self.protocol["consecutive_crash_rollback"]) \
                or not state.last_good_harness:
            return 0
        good = state.last_good_harness
        current = set(gitops.git(["ls-files", "--", "harness"], self.wt).split())
        keep = set(gitops.git(["ls-tree", "-r", "--name-only", good, "--", "harness"], self.wt).split())
        for path in current - keep:
            (self.wt / path).unlink(missing_ok=True)
        gitops.git(["checkout", good, "--", "harness"], self.wt)
        sha = gitops.commit(self.wt, f"[{task.id} r{rnd}] rollback harness to {good[:10]}",
                            self._trailers(task, rnd, "rollback", target=good), ["harness"])
        self._violation(task, rnd, "orchestrator", "crash_rollback", [],
                        f"{state.crash_streak} consecutive crashed rounds; harness restored to "
                        f"{good[:10]} (commit {sha})")
        state.crash_streak = 0
        return 1

    def _append_round(self, task, rnd, start, before: Usage, budget: Budget, phases, candidate,
                      val, decision, reason, decided_by, best_before, accepted_commit,
                      harness_changes, violations, state: TaskState) -> None:
        used = budget.used()
        cost = used.minus(before)
        state.cumulative = used
        state.last_round = rnd
        state.idle_streak = state.idle_streak + 1 if _idle(decision, harness_changes) else 0
        self.records.append("rounds", {
            "task_id": task.id, "round": rnd, "started_at": start, "ended_at": now_iso(),
            "phases": phases, "candidate": candidate,
            "val": public_view(val, task.public_metrics),
            "best_val_before": best_before, "best_val_after": state.best_val,
            "decision": decision, "decision_reason": reason[:500], "decided_by": decided_by,
            "accepted_commit": accepted_commit, "harness_changes": harness_changes,
            "violations": int(violations), "cost": cost.as_dict(), "cumulative": used.as_dict(),
            "budget_used_frac": round(budget.fraction_used(), 6),
        })
        for phase in ("meta", "executor", "improver"):
            tpath = self.transcript_path(task.id, rnd, phase)
            if tpath.exists() and tpath.stat().st_size:
                self.records.append("artifacts", {
                    "task_id": task.id, "round": rnd, "kind": "transcript", "path": str(tpath),
                    "bytes": tpath.stat().st_size, "sha256": None})
        gitops.commit(self.wt, f"[{task.id} r{rnd}] records", self._trailers(task, rnd, "records"),
                      ["records"])
        self._push()

    def _push(self) -> None:
        if not self.push_remote:
            return
        try:
            gitops.push(self.wt, self.push_remote, self.branch)
        except gitops.GitError as exc:
            self.log(f"warning: push to {self.push_remote} failed: {exc}")

    # ---- resume ------------------------------------------------------------------------------
    def task_state(self, task: Task) -> TaskState:
        state = TaskState()
        rows = [r for r in self.records.read("rounds") if r["task_id"] == task.id]
        for r in rows:
            state.last_round = r["round"]
            state.best_val = r.get("best_val_after")
            if r["decision"] == "baseline":
                state.baseline_val = r["val"].get("score")
                state.accepted_round = 0
            if r["decision"] in ("baseline", "accept"):
                state.accepted_dev = (r.get("candidate") or {}).get("dev_score")
            if r["decision"] == "accept":
                state.accepted_round = r["round"]
            if r["val"]["status"] in ("ok", "error") and r["decision"] != "baseline":
                state.val_calls += 1
            if r["decision"] != "baseline":
                state.idle_streak = state.idle_streak + 1 \
                    if _idle(r["decision"], r.get("harness_changes") or {}) else 0
            state.cumulative = Usage.from_dict(r.get("cumulative"))
        if rows:
            state.last_good_harness = gitops.rev_parse(self.wt)
            partial = self.partial_usage(task, state.last_round, rows[-1]["ended_at"])
            state.cumulative = Usage(*(a + b for a, b in zip(
                (state.cumulative.wall_s, state.cumulative.gpu_hours, state.cumulative.usd,
                 state.cumulative.input_tokens, state.cumulative.output_tokens),
                (partial.wall_s, partial.gpu_hours, partial.usd, partial.input_tokens,
                 partial.output_tokens))))
        return state

    def partial_usage(self, task: Task, last_round: int, last_ended: str) -> Usage:
        """Spend recorded after the last complete round (an interrupted round): charged model
        calls, GPU hours, and wall time up to the last recorded activity. It counts against the
        budget whether the interrupted round is discarded or closed."""
        use = Usage()
        stamps = []
        for u in self.records.read("usage"):
            if u["task_id"] == task.id and u["round"] > last_round and u.get("charged", True):
                use.usd += float(u.get("usd") or 0)
                use.input_tokens += int(u["input_tokens"]) + int(u.get("cache_read_tokens") or 0) \
                    + int(u.get("cache_write_tokens") or 0)
                use.output_tokens += int(u["output_tokens"])
                stamps.append(u["ts"])
        for e in self.records.read("experiments"):
            if e["task_id"] == task.id and e["round"] > last_round:
                use.gpu_hours += float(e.get("gpu_hours") or 0)
                stamps.append(e["ended_at"])
        if stamps:
            use.wall_s = max(0.0, _seconds(max(stamps)) - _seconds(last_ended))
        waited = sum(float(e.get("queue_s") or 0) for e in self.records.read("experiments")
                     if e["task_id"] == task.id and e["round"] > last_round)
        use.queue_s = min(waited, use.wall_s)
        use.wall_s -= use.queue_s
        return use

    def task_done(self, task: Task) -> bool:
        return gitops.ref_exists(self.wt, f"refs/tags/{self.branch}/{task.id}/final")

    # ---- main --------------------------------------------------------------------------------
    def _rewind_partial_round(self) -> None:
        """After an interruption, return to the last complete round.

        A round is complete once its records commit exists; pushes happen only after that commit,
        so anything newer is local, partial and safe to discard. Tags of discarded rounds go too.
        """
        commits = gitops.log_commits(self.wt, f"{self.run['base']['commit']}..HEAD")
        complete = {"init", "records", "final", "posthoc", "environment"}
        last_good = next((c["sha"] for c in reversed(commits)
                          if c["trailers"].get("Rsib-Scope") in complete), None)
        head = gitops.rev_parse(self.wt)
        pending = [c.path for c in gitops.changes(self.wt) if not c.path.startswith(".rsib/")]
        kept_records = {p: (self.wt / p).read_bytes() for p in self.records.owned_paths()
                        if p.endswith(".jsonl") and (self.wt / p).exists()}
        dropped = 0
        if last_good and last_good != head:
            dropped = len(commits) - 1 - [c["sha"] for c in commits].index(last_good)
            gitops.git(["reset", "-q", "--hard", last_good], self.wt)
        if pending or dropped:
            gitops.reset_hard(self.wt)
            for p, data in kept_records.items():  # append-only: the old content is a prefix
                if len(data) > (self.wt / p).stat().st_size:
                    (self.wt / p).write_bytes(data)
            recorded = {}
            for r in self.records.read("rounds"):
                recorded[r["task_id"]] = r["round"]
            for name in gitops.git(["tag", "--list", f"{self.branch}/*/r*"], self.wt).split():
                task_id, rtag = name[len(self.branch) + 1:].rsplit("/", 1)
                if rtag[1:].isdigit() and int(rtag[1:]) > recorded.get(task_id, -1):
                    gitops.git(["tag", "-d", name], self.wt)
            self.records.track_all()
            self._violation(self.tasks[0], 0, "orchestrator", "resume_discard", pending[:200],
                            f"interrupted round discarded: {dropped} partial commit(s), "
                            f"{len(pending)} uncommitted path(s)")
            gitops.commit(self.wt, "records: resume", {"Rsib-Run": self.branch,
                                                       "Rsib-Scope": "records"}, ["records"])

    def _model_down(self, task: Task) -> str | None:
        """The error of the last round of `task` when each of its phases ended with a model error."""
        rows = [r for r in self.records.read("rounds") if r["task_id"] == task.id]
        phases = [p or {} for p in (rows[-1].get("phases") or {}).values()] if rows else []
        if phases and all(p.get("status") == "model_error" for p in phases):
            return str(phases[0].get("error") or phases[0].get("summary") or "")[:300]
        return None

    def _pause_due(self) -> bool:
        """True when the hosting job is about to end (RSIB_STOP_AT, epoch seconds) and a new round
        would not fit in the remaining time (RSIB_STOP_MARGIN_S, default one hour)."""
        stop_at = os.environ.get("RSIB_STOP_AT")
        if not stop_at:
            return False
        margin = float(os.environ.get("RSIB_STOP_MARGIN_S", 3600))
        return time.time() + margin >= float(stop_at)

    def start(self, close_partial_round: bool = False) -> dict:
        if self.sandbox.git_view and (stale := self.sandbox.sweep_views()):
            self.log(f"removed {stale} Git view(s) left by an interrupted phase")
        if close_partial_round:
            self._prepare_close()
        else:
            self._rewind_partial_round()
        self.records.track_all()
        self.services._exp_counter = len(self.records.read("experiments"))
        try:
            with self.serving():
                for index, task in enumerate(self.tasks):
                    if self.task_done(task):
                        continue
                    self._run_task(task, index)
                self.finalize()
        except _Paused as exc:
            self.log(f"paused at a round boundary: {exc}; resume with `rsib run start`")
            self._push()
            return {"branch": self.branch, "paused": str(exc), "final": self.records.read("final")}
        self._push()
        return {"branch": self.branch, "final": self.records.read("final")}

    def _run_task(self, task: Task, index: int) -> None:
        state = self.task_state(task)
        usd_metering = bool(self.run["model"].get("usd_metering", True))
        budget = Budget(dict(self.budget_limits(task)), carried=state.cumulative,
                        usd_metering=usd_metering)
        self.budgets[task.id] = budget
        budget.val_calls = state.val_calls
        if state.last_round < 0:
            self.baseline_round(task, budget, state)
        rnd = state.last_round + 1
        if self._close and self._close["task"] == task.id and self._close["round"] == rnd:
            self.one_round(task, rnd, budget, state, first_of_later_task=False,
                           closing=self._closing_info(task, rnd, budget, state))
            self._close = None
            rnd += 1
        max_rounds = int(self.protocol["max_rounds"])
        while True:
            reason = budget.exhausted()
            if reason is None and rnd > max_rounds:
                reason = "max_rounds"
            if reason is None and state.val_calls >= int(budget.limits["val_max_calls"]):
                reason = "val_calls"
            idle_limit = int(self.protocol.get("max_idle_rounds", 0) or 0)
            if reason is None and idle_limit and state.idle_streak >= idle_limit:
                reason = "idle"
            if reason:
                break
            if self._pause_due():
                raise _Paused("the hosting job ends soon (RSIB_STOP_AT)")
            self.one_round(task, rnd, budget, state, first_of_later_task=(index > 0 and rnd == 1))
            rnd += 1
            error = self._model_down(task)
            if error:
                # Every phase of the round failed on the model (a provider message the adapter
                # does not know as a limit): stop instead of idling out and scoring the final
                # solution with a model that refuses every call.
                raise _Paused(f"every phase of round {rnd - 1} failed with a model error ({error})")
        head = gitops.rev_parse(self.wt)
        gitops.git(["tag", "-f", "-a", f"{self.branch}/{task.id}/final", "-m",
                    f"ended_reason={reason}\naccepted_round={state.accepted_round}\n"
                    f"best_val={state.best_val}\nbaseline_val={state.baseline_val}\n"
                    f"rounds={state.last_round}", head], self.wt, env=gitops.bot_env())
        self.log(f"[{task.id}] finished: {reason}, best val {state.best_val}")
        self._push()

    # ---- submit rule helpers -----------------------------------------------------------------
    def reference_dev(self, task: Task, rnd: int, budget: Budget, commit: str) -> float | None:
        """Dev score of the accepted solution at `commit` (the bar of the submit rule)."""
        out_dir = self.artifact_dir(task.id, rnd) / "eval-dev-reference"
        start = now_iso()
        workspace = self.snapshot_workspace(task, commit, out_dir / "snapshot")
        result = self.run_split_eval(task, workspace, "dev", out_dir, rnd, budget=budget)
        budget.charge(gpu_hours=result["gpu_hours"])
        if result.get("queued"):
            budget.add_wait(*result["queued"])
        self._record_eval(task, rnd, "orchestrator", "eval-dev", result, out_dir, start,
                          exp_id=_devref_id(task.id, rnd))
        self.log(f"[{task.id}] r{rnd}: dev score of the accepted solution = {result['score']}")
        return result["score"] if result["status"] == "ok" else None

    def _submits_best_dev(self, ex: PhaseOutcome, state: TaskState) -> bool:
        """Whether a round whose executor ended without finishing submits only its best dev
        evaluation (protocol `unfinished_submit` = best_dev; needs the submit rule's bar)."""
        return (ex.status not in ("finished", "ok") and self.submit_rule
                and state.accepted_dev is not None
                and run_constant(self.protocol, "unfinished_submit") == "best_dev")

    def _best_dev_submission(self, task: Task, rnd: int, bar: float) -> dict | None:
        """The best of the executor's dev evaluations in round `rnd` that beat `bar` (any margin;
        the first of equal scores), as a submission of its snapshot; None if none did."""
        rows = [e for call in self._dev_calls(task, rnd) for e in call]
        best = best_improvement(task, rows, bar)
        if best is None:
            return None
        e = rows[best]
        return {"snapshot": str(Path(e["artifact_dir"]) / "snapshot"), "score": e["score"],
                "bar": bar, "exp_id": e["exp_id"], "unfinished": True}

    def known_dev_score(self, task: Task, rnd: int, snapshot_sha: str) -> float | None:
        """Dev score of an identical snapshot evaluated in this round, if any."""
        for e in reversed(self.records.read("experiments")):
            if (e["task_id"] == task.id and e["round"] == rnd and e["kind"] == "eval-dev"
                    and e.get("snapshot_sha") == snapshot_sha and e["status"] == "ok"):
                return e["score"]
        return None

    def restore_workspace(self, task: Task, snapshot: Path) -> None:
        """Make workspace/<task> equal to `snapshot` for every committable file."""
        limit = float(self.protocol["max_file_mb"]) * (1 << 20)
        ws = self.workspace_dir(task.id)
        for path in list(ws.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.stat().st_size <= limit:
                if not (snapshot / path.relative_to(ws)).exists():
                    path.unlink()
        for src in snapshot.rglob("*"):
            if src.is_file():
                dst = ws / src.relative_to(snapshot)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def _prepare_close(self) -> None:
        """Validate that an interrupted round can be closed (no partial commits after the last
        complete round), and remember which round it is."""
        commits = gitops.log_commits(self.wt, f"{self.run['base']['commit']}..HEAD")
        tail = []
        for c in reversed(commits):
            if c["trailers"].get("Rsib-Scope") in ("init", "records"):
                break
            tail.append(c)
        if any(c["trailers"].get("Rsib-Scope") != "notes" for c in tail):
            raise RuntimeError("the interrupted round already committed a candidate or harness "
                               "edits; resume without --close-partial-round to discard it")
        for task in self.tasks:
            if self.task_done(task):
                continue
            rows = [r for r in self.records.read("rounds") if r["task_id"] == task.id]
            if not rows:
                return  # nothing to close: the task has not even finished its baseline
            self._close = {"task": task.id, "round": rows[-1]["round"] + 1,
                           "last_ended": rows[-1]["ended_at"],
                           "cumulative": Usage.from_dict(rows[-1]["cumulative"])}
            return

    def _closing_info(self, task: Task, rnd: int, budget: Budget, state: TaskState) -> dict:
        """Close an interrupted round: its executor counts as stopped; if one of its dev
        evaluation calls beat the accepted solution, the state the live submit rule picked from
        that call is what gets submitted."""
        bar = state.accepted_dev
        if self.submit_rule and bar is None:
            # exactly the reference row: a candidate's exp_id may end in "-devref" too
            ref = [e for e in self.records.read("experiments")
                   if e["exp_id"] == _devref_id(task.id, rnd) and e["phase"] == "orchestrator"
                   and e["status"] == "ok"]
            bar = ref[-1]["score"] if ref else self.reference_dev(task, rnd, budget, "HEAD")
            state.accepted_dev = bar
        submit = None
        if self.submit_rule and bar is not None:
            for rows in self._dev_calls(task, rnd):
                best = best_improvement(task, rows, bar, submit_margin(task, self.protocol))
                if best is not None:
                    e = rows[best]
                    submit = {"snapshot": str(Path(e["artifact_dir"]) / "snapshot"),
                              "score": e["score"], "bar": bar, "exp_id": e["exp_id"]}
                    break
        ex = PhaseOutcome(status="stopped", summary="executor stopped by the operator; the round "
                          "was closed with the work done so far", submit=submit)
        return {"executor": ex, "before": self._close["cumulative"]}

    def _dev_calls(self, task: Task, rnd: int) -> list[list[dict]]:
        """The executor's dev evaluations of a round that kept their snapshot, as one list of rows
        per call (the rows of a call share `<task>-r<NNN>-dev<NNN>`), in record order."""
        prefix = f"{task.id}-r{rnd:03d}-dev"
        calls: dict[str, list[dict]] = {}
        for e in self.records.read("experiments"):
            snap = Path(e.get("artifact_dir") or "/nonexistent") / "snapshot"
            if (e["task_id"] == task.id and e["round"] == rnd and e["kind"] == "eval-dev"
                    and e["phase"] == "executor" and snap.is_dir()):
                calls.setdefault(e["exp_id"].removeprefix(prefix).split("-", 1)[0], []).append(e)
        return list(calls.values())

    def snapshot_workspace(self, task: Task, commit: str, dest: Path) -> Path:
        """Extract workspace/<task> at `commit` to `dest`, a directory on the shared filesystem,
        so SLURM evaluators on other nodes can read it (a local /tmp would not be visible there)."""
        if dest.exists():
            shutil.rmtree(dest)
        gitops.archive_extract(self.wt, commit, f"workspace/{task.id}", dest)
        return dest / "workspace" / task.id

    def working_snapshot(self, task: Task, dest: Path, src: Path | None = None) -> Path:
        """Copy `src` (default workspace/<task>) for scoring on dev or val, without files over
        `max_file_mb`, __pycache__, *.pyc and .git. Other git-ignored files (build/, dist/, ...)
        are copied, although a commit, and so the test evaluation, leaves them out."""
        limit = float(self.protocol["max_file_mb"]) * (1 << 20)
        src = src or self.workspace_dir(task.id)
        if dest.exists():
            shutil.rmtree(dest)

        def ignore(directory, names):
            skip = []
            for name in names:
                path = Path(directory) / name
                if name in ("__pycache__", ".git") or name.endswith(".pyc"):
                    skip.append(name)
                elif path.is_file() and path.stat().st_size > limit:
                    skip.append(name)
            return skip

        shutil.copytree(src, dest, ignore=ignore, symlinks=False)
        return dest

    def finalize(self, retry_failed: bool = False) -> None:
        """Score each task's final commit on the test split, once, after the agent is done.

        With retry_failed, tasks whose latest test evaluation errored (for example an
        infrastructure failure) get one more line in final.jsonl; earlier lines are kept.
        """
        latest = {}
        for r in self.records.read("final"):
            latest[r["task_id"]] = r
        with self.serving(self.tasks):
            for task in self.tasks:
                if not self.task_done(task):
                    continue
                if task.id in latest and not (retry_failed
                                              and latest[task.id]["test"]["status"] != "ok"):
                    continue
                self._final_test(task)
        sha = gitops.commit(self.wt, "final: test scores",
                            {"Rsib-Run": self.branch, "Rsib-Scope": "final"}, ["records"])
        if sha is None and gitops.ref_exists(self.wt, f"refs/tags/{self.branch}/final"):
            self.end_of_run()  # completes a posthoc or archive step an earlier finalize missed
            return
        if all(self.task_done(t) for t in self.tasks):
            gitops.tag(self.wt, f"{self.branch}/final", sha or gitops.rev_parse(self.wt))
            self.end_of_run()

    def end_of_run(self) -> None:
        """After the final test: posthoc test scores of every accepted version, then the scratch
        inventory and archive (protocol `posthoc_at_end`, `archive_at_end`). Idempotent."""
        if self.protocol.get("posthoc_at_end"):
            from ..posthoc import score_missing
            with self.serving(self.tasks):
                added = score_missing(self, 1, self.log)
            if added:
                gitops.commit(self.wt, "posthoc: test scores of accepted versions",
                              {"Rsib-Run": self.branch, "Rsib-Scope": "posthoc"}, ["records"])
        if self.protocol.get("archive_at_end") and not self.records.read("environment"):
            self.archive_scratch()

    def archive_scratch(self) -> dict:
        """Inventory this run's scratch, then move it and the large artifacts to the archive."""
        from ..scratch import DEFAULT_RETENTION_DAYS, archive, inventory
        inv = inventory(self.sandbox.dir)
        days = float((self.site.get("sandbox") or {}).get("archive_days", DEFAULT_RETENTION_DAYS))
        rec = archive(self.sandbox.dir, self.art_root, self.run["run_id"], self.slug,
                      float(self.protocol.get("max_file_mb", 5)), days)
        row = {**rec, "inventory": inv}
        self.records.append("environment", row)
        gitops.commit(self.wt, "environment: scratch inventory and archive",
                      {"Rsib-Run": self.branch, "Rsib-Scope": "environment"}, ["records"])
        self.log(f"archived scratch ({inv['bytes'] / 1e9:.2f} GB, {len(inv['hf_models'])} HF "
                 f"models, {len(inv['python_packages'])} packages) and {rec['large_artifacts']} "
                 f"large artifacts as {rec['archive']}; deleted after {rec['delete_after'][:10]}")
        return row

    def _final_test(self, task: Task) -> None:
        """Score the commit of the task's final tag on test and append its final.jsonl line."""
        tag = f"{self.branch}/{task.id}/final"
        info = dict(line.split("=", 1) for line in gitops.git(
            ["tag", "-l", "--format=%(contents)", tag], self.wt).splitlines() if "=" in line)
        commit = gitops.rev_parse(self.wt, tag)
        rounds, final_round = int(info.get("rounds", 0) or 0), _int(info.get("accepted_round"))
        out_dir = self.hidden_eval_dir(task.id, "final-test")
        workspace = self.snapshot_workspace(task, commit, out_dir / "snapshot")
        start = now_iso()
        result = self.run_split_eval(task, workspace, "test", out_dir,
                                     rounds if final_round is None else final_round)
        self._record_eval(task, rounds, "orchestrator", "eval-test", result, out_dir, start)
        rows = [r for r in self.records.read("rounds") if r["task_id"] == task.id]
        totals = Usage.from_dict(rows[-1]["cumulative"]) if rows else Usage()
        best_val = _num(info.get("best_val"))
        self.records.append("final", {
            "task_id": task.id, "ended_reason": info.get("ended_reason", "unknown"),
            "rounds": rounds, "final_commit": commit, "final_round": final_round,
            "best_val": best_val, "baseline_val": _num(info.get("baseline_val")),
            "test": public_view(result, task.public_metrics),
            "normalized": {"val": task.normalize(best_val),
                           "test": task.normalize(result["score"])},
            "totals": totals.as_dict(),
        })
        self.log(f"[{task.id}] test = {result['score']} ({result['status']})")


def _devref_id(task_id: str, rnd: int) -> str:
    """exp_id of the orchestrator's dev evaluation of the accepted solution in round `rnd`."""
    return f"{task_id}-r{rnd:03d}-devref"


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Paused(Exception):
    """Raised between rounds when the run must stop without ending: the hosting job is about to
    end, or the model failed every phase of the last round."""


def _seconds(iso: str) -> float:
    import datetime as _dt
    return _dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.timezone.utc).timestamp()


def _idle(decision: str, harness_changes: dict) -> bool:
    """A round is idle when it produced no candidate and no harness change."""
    changed = any((harness_changes.get(s) or {}).get("commit") for s in ("E", "I"))
    return decision == "no-change" and not changed


def _merge(a: dict, b: dict) -> dict:
    out = {}
    for scope in ("E", "I"):
        x, y = a.get(scope, {}), b.get(scope, {})
        out[scope] = {"commit": y.get("commit") or x.get("commit"),
                      "lines": int(x.get("lines", 0)) + int(y.get("lines", 0))}
    return out


def _num(text):
    try:
        return None if text in (None, "None", "") else float(text)
    except ValueError:
        return None


def _int(text):
    try:
        return None if text in (None, "None", "") else int(text)
    except ValueError:
        return None
