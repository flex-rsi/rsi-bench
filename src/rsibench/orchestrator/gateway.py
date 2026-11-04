"""Local HTTP gateway between the editable harness and everything that costs budget.

The harness runs as a separate process and can rewrite its own code, so it is never trusted with
credentials or meters. It reaches the model, experiments and dev evaluation only through this
gateway, which enforces the budget and writes the records:

    POST /v1/complete     one model call              -> records/usage.jsonl + transcript
    POST /v1/experiment   one job on the backend       -> records/experiments.jsonl + artifacts
    POST /v1/evaluate     score the workspace, or several candidate directories at once, on dev
                                                       -> records/experiments.jsonl
    POST /v1/budget       remaining budget

Every process that may use the gateway (a harness phase, an experiment job, the evaluation of a
task with model access) holds a grant of its own, named by `Authorization: Bearer <token>`: it
fixes the endpoints, the budget charged, the model-call cap and the transcript of that process,
and a revoked or unknown token gets HTTP 401. Messages must follow schemas/messages.schema.json;
images arrive inline, the gateway never reads a file for the harness.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shlex
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..budget import Budget
from ..compute import Job
from ..contracts import parse_result, scalars
from ..envs import scrubbed_env
from ..models import ModelError, context
from ..models.pricing import cost_usd
from ..protocol import run_constant
from ..records import validate
from ..sandbox import python_roots
from ..tasks import Task
from ..util import append_jsonl, now_iso, read_json, sha256_file, sha256_tree, tail
from .evaluator import run_eval

CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")
EXPERIMENT_ENDPOINTS = ("complete", "budget")


class GatewayRefusal(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass(eq=False)
class Grant:
    """What one process may do through the gateway, and its own meters. `budget` is charged for
    its model calls (None: metered only). An executor's grant also holds the submit rule's state:
    the bar, the margin a dev score must beat it by, the submission and the grace calls after it."""

    id: str
    consumer: str
    task: Task
    round: int
    budget: Budget | None
    endpoints: frozenset[str]
    transcript: Path
    bar: float | None = None
    call_cap: int | None = None  # None: the protocol's max_model_calls_per_phase
    margin: float = 0.0  # dev gain over the bar that a submission needs (submit_margin)
    calls: int = 0
    sessions: dict = field(default_factory=dict)  # session label -> {"seen", "digest"}
    submit: dict | None = None
    grace: int = 0
    revoked: bool = False


def submit_margin(task, protocol: dict) -> float:
    """Dev gain over the accepted solution that the submit rule asks for: `submit_min_gain` of the
    task's anchor span |reference - baseline|, so one value fits every task; 0 (any gain) when
    the run sets none or the task has no anchors."""
    share = float(run_constant(protocol, "submit_min_gain") or 0.0)
    base, ref = getattr(task, "baseline", None), getattr(task, "reference", None)
    if share <= 0 or base is None or ref is None:
        return 0.0
    return share * abs(ref - base)


def auto_submit_at(task, bar: float | None, margin: float) -> float | None:
    """The dev score a submission has to beat: the bar moved by the margin in the task's
    direction (None while the bar is unknown)."""
    if bar is None:
        return None
    return bar + margin if task.direction == "maximize" else bar - margin


def best_improvement(task, rows: list[dict], bar: float | None, margin: float = 0.0) -> int | None:
    """Submit rule: the index of the best ok score in `rows` that beats `bar` by more than
    `margin` (the first of equal scores), or None."""
    line = auto_submit_at(task, bar, margin)
    if line is None:
        return None
    best = None
    for i, row in enumerate(rows):
        if (row["status"] == "ok" and task.better(row["score"], line)
                and (best is None or task.better(row["score"], rows[best]["score"]))):
            best = i
    return best


def _concurrently(fn, items: list) -> list:
    """[fn(item) for item in items], all at once in daemon threads, which, like the request
    threads, never hold up the orchestrator's exit. A worker's exception is raised here."""
    out, errors = [None] * len(items), []

    def work(i: int) -> None:
        try:
            out[i] = fn(items[i])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(items))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return out


def _unscored(exc: Exception) -> dict:
    """The run_eval fields of a candidate whose snapshot or evaluation could not be made."""
    return {"status": "error", "score": None, "metrics": {}, "artifacts": [], "info": {},
            "detail": f"not scored: {type(exc).__name__}: {exc}"[:1000], "wall_s": 0.0,
            "gpu_hours": 0.0, "exit_code": None, "job_id": None, "backend": "none",
            "log_path": None, "log_tail": ""}


def _submit_message(task, score: float, bar: float, cid: str | None) -> str:
    of = f" of candidate {cid}" if cid else ""
    what = (f"candidate {cid}, exactly as evaluated, replaces the workspace and is submitted"
            if cid else "exactly this evaluated state is submitted")
    return (f"dev {task.primary_metric} {score:.6g}{of} beats the accepted solution ({bar:.6g}). "
            f"This round ends now and {what} for validation. Update your notes and call finish; "
            f"further experiments and evaluations are refused.")


def message_problems(messages) -> list[str]:
    """The first schema errors of a neutral message list, else the first image whose data is not
    valid base64 of its declared format."""
    errors = validate("messages", messages, limit=3)
    if errors:
        return errors
    for i, msg in enumerate(messages):
        for part in context.images_in([msg]):
            try:
                data = base64.b64decode(part["data"], validate=True)
            except ValueError:  # binascii.Error, or non-ASCII text
                return [f"{i}: image data is not valid base64"]
            if context.sniff(data) != part["mime"]:
                return [f"{i}: image data is not a {part['mime']} image"]
    return []


def media_ref(part: dict, media: Path) -> dict:
    """Write an image part once under `media` (content-addressed) and return the reference
    that transcripts keep in its place."""
    name, data = context.image_file(part)
    target = media / name
    if not target.exists():
        media.mkdir(parents=True, exist_ok=True)
        tmp = media / f".{name}.{threading.get_ident()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return {"type": "image", "mime": part["mime"], "file": f"media/{name}", "bytes": len(data)}


class Services:
    """Request handlers. Each request acts for the grant its token names (`authorize`)."""

    def __init__(self, engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._exp_counter = 0
        self._grants: dict[str, Grant] = {}  # live grants by token
        self._issued = 0

    # ---- grants -----------------------------------------------------------------------------
    def grant(self, consumer: str, task: Task, rnd: int, budget: Budget | None, endpoints,
              transcript: Path, bar: float | None = None, call_cap: int | None = None,
              margin: float = 0.0) -> str:
        """Issue a grant for one process and return its token."""
        with self._lock:
            self._issued += 1
            token = secrets.token_hex(24)
            self._grants[token] = Grant(f"g{self._issued}", consumer, task, rnd, budget,
                                        frozenset(endpoints), Path(transcript), bar, call_cap,
                                        margin=margin)
        return token

    def revoke(self, token: str | None) -> Grant | None:
        """End a grant: from then on its token gets HTTP 401, and the jobs it requested that have
        not started never start. Returns the grant (None for an unknown token)."""
        with self._lock:
            grant = self._grants.pop(token, None)
            if grant is not None:
                grant.revoked = True
        return grant

    def authorize(self, token: str) -> Grant:
        """The live grant `token` names; any other token is refused with HTTP 401."""
        with self._lock:
            grant = self._grants.get(token)
        if grant is None:
            raise GatewayRefusal(401, "unauthorized", "unknown or revoked token")
        return grant

    def serve(self, grant: Grant, endpoint: str, payload: dict) -> dict:
        """One request of `grant`'s process; an endpoint outside the grant is HTTP 403."""
        if endpoint not in grant.endpoints:
            raise GatewayRefusal(403, "forbidden",
                                 f"the {grant.consumer} grant does not allow /v1/{endpoint}")
        return getattr(self, endpoint)(grant, payload)

    def _refuse_after_submit(self, grant: Grant) -> None:
        if grant.submit is not None:
            raise GatewayRefusal(409, "submit_required",
                                 "the dev score improved; this round is being submitted. Update your "
                                 "notes and call finish.")

    # ---- model ------------------------------------------------------------------------------
    def complete(self, grant: Grant, payload: dict) -> dict:
        eng, budget, task = self.engine, grant.budget, grant.task
        reason = budget.exhausted() if budget is not None else None
        if reason:
            raise GatewayRefusal(402, "budget_exhausted", reason)
        limit = grant.call_cap or int(eng.protocol.get("max_model_calls_per_phase", 0) or 0)
        with self._lock:
            grant.calls += 1
            calls = grant.calls
            if grant.submit is not None:
                grant.grace += 1
            grace = grant.grace
        if grace > int(eng.protocol.get("submit_grace_calls", 3)):
            raise GatewayRefusal(402, "submit_required",
                                 "the round was submitted after a dev improvement; the phase is over")
        if limit and calls > limit:
            source = "task's" if grant.call_cap else "protocol"
            raise GatewayRefusal(402, "phase_call_limit", f"the {grant.consumer} grant reached "
                                 f"the {source} limit of {limit} model calls")
        for key in ("system", "messages", "tools"):
            if key not in payload:
                raise GatewayRefusal(400, "bad_request", f"missing '{key}'")
        problems = message_problems(payload["messages"])
        if problems:
            raise GatewayRefusal(400, "bad_request", "invalid messages: " + "; ".join(problems))
        cap = int(eng.protocol["max_tokens_per_call"])
        max_tokens = max(256, min(int(payload.get("max_tokens") or cap), cap))
        meta = {"phase": grant.consumer, "task_id": task.id, "round": grant.round,
                "session": str(payload.get("session", "0")), "grant": grant.id}
        lowerings = eng.model.lowerings(payload["system"], payload["messages"], payload["tools"],
                                        meta)
        if lowerings and run_constant(eng.protocol, "exact_context"):
            raise GatewayRefusal(400, "inexact_context",
                                 f"the {eng.model.adapter} adapter would apply "
                                 f"{', '.join(lowerings)} to this context, and this run requires "
                                 f"exact_context")
        t0 = time.monotonic()
        try:
            resp = eng.model.complete(payload["system"], payload["messages"], payload["tools"],
                                      max_tokens, meta)
        except ModelError as exc:
            raise GatewayRefusal(502, "model_error", str(exc)) from exc
        latency = time.monotonic() - t0
        usd = resp.extra.get("cost_usd")  # adapters that report their own cost (claude-cli)
        if usd is None:
            usd = cost_usd(eng.model.adapter, eng.model.name, resp.input_tokens,
                           resp.output_tokens, resp.cache_read_tokens, resp.cache_write_tokens)
        row = {
            "ts": now_iso(), "task_id": task.id, "round": grant.round, "phase": grant.consumer,
            "model": resp.model, "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens, "cache_read_tokens": resp.cache_read_tokens,
            "cache_write_tokens": resp.cache_write_tokens,
            "usd": None if usd is None else round(usd, 6), "stop_reason": resp.stop_reason,
            "latency_s": round(latency, 3), "lowerings": lowerings,
        }
        for key in ("provider", "response_id"):
            if resp.extra.get(key) is not None:
                row[key] = resp.extra[key]
        if budget is not None:
            budget.charge(usd=usd or 0.0, input_tokens=resp.input_tokens + resp.cache_read_tokens
                          + resp.cache_write_tokens, output_tokens=resp.output_tokens)
        else:
            row["charged"] = False
        live = budget if budget is not None else getattr(eng, "live_budget", lambda _: None)(task.id)
        if resp.extra.get("outage") and live is not None:  # the model was unavailable for part of
            live.add_wait(*resp.extra["outage"])           # the call (also a metered evaluation's)
        eng.records.append("usage", row)
        with self._lock:  # calls of one grant may overlap; each line follows the one before it
            self._transcript(grant, payload, resp, lowerings)
        wire = resp.to_wire()
        wire["budget"] = budget.remaining() if budget is not None else None
        return wire

    def _transcript(self, grant: Grant, payload: dict, resp, lowerings: list[str]) -> None:
        """Append only what is new in this call, so transcripts grow linearly; a history whose
        earlier messages changed, or that no longer holds the previous reply in its place (e.g.
        a message put there instead), is appended whole. Images are kept as files under media/
        next to the transcript, never as base64 in it."""
        path, sessions = grant.transcript, grant.sessions
        session, messages = str(payload.get("session", "0")), payload["messages"]
        prev = sessions.get(session)
        entry = {"ts": now_iso(), "session": session}
        seen = 0
        if prev is None:
            entry["system"] = payload["system"]
            entry["tools"] = [t.get("name") for t in payload["tools"]]
        elif len(messages) >= prev["seen"] and messages[prev["seen"] - 1]["role"] == "assistant" \
                and context.digest(messages[:prev["seen"] - 1]) == prev["digest"]:
            seen = prev["seen"]  # what follows the previous reply is new
        else:
            entry["history_edited"] = True
        media = path.parent / "media"
        entry["new_messages"] = context.replace_images(messages[seen:],
                                                       lambda part: media_ref(part, media))
        entry["lowerings"] = lowerings
        entry["response"] = resp.to_wire()
        sessions[session] = {"seen": len(messages) + 1, "digest": context.digest(messages)}
        append_jsonl(path, entry)

    # ---- experiments ------------------------------------------------------------------------
    def experiment(self, grant: Grant, payload: dict) -> dict:
        self._refuse_after_submit(grant)
        eng, budget, task, rnd = self.engine, grant.budget, grant.task, grant.round
        command = str(payload.get("command") or "").strip()
        if not command:
            raise GatewayRefusal(400, "bad_request", "missing 'command'")
        gpus = int(payload.get("gpus") or 0)
        if gpus < 0 or gpus > task.experiment_gpus_max:
            raise GatewayRefusal(400, "bad_request",
                                 f"gpus must be between 0 and {task.experiment_gpus_max} for this task")
        reason = budget.exhausted()
        if reason:
            raise GatewayRefusal(402, "budget_exhausted", reason)
        remaining = budget.remaining()
        timeout = min(float(payload.get("timeout_min") or 60) * 60,
                      float(eng.protocol["experiment_timeout_min"]) * 60, remaining["wall_s"])
        if gpus:
            timeout = min(timeout, remaining["gpu_hours"] * 3600 / gpus)
        if timeout < 30:
            raise GatewayRefusal(402, "budget_exhausted", "not enough budget left for an experiment")
        with self._lock:
            self._exp_counter += 1
            exp_id = f"{task.id}-r{rnd:03d}-e{self._exp_counter:03d}"
        art_dir = eng.artifact_dir(task.id, rnd) / exp_id
        art_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = art_dir / "metrics.json"
        workspace = eng.workspace_dir(task.id)
        backend = eng.backend if gpus else eng.cpu_backend
        container = eng.container_for(task, backend)
        # the job's own grant: its model calls are charged to the task budget like the phase's
        token = self.grant("experiment", task, rnd, budget, EXPERIMENT_ENDPOINTS,
                           art_dir / "transcripts" / "experiment.jsonl")
        env = scrubbed_env({
            "RSIB_ARTIFACT_DIR": art_dir, "RSIB_METRICS_FILE": metrics_file,
            "RSIB_DEV_DATA": task.split_dir("dev", eng.data_root), "RSIB_WORKSPACE": workspace,
            "RSIB_PYTHON": sys.executable,  # a backend that runs the job in an image sets "python"
            "RSIB_GATEWAY_URL": eng.gateway.url, "RSIB_GATEWAY_TOKEN": token,
        })
        mounts, job_command = None, command  # the record keeps the agent's command
        sb = eng.sandbox
        if sb.enabled:
            env.update(sb.env())
            dev = task.split_dir("dev", eng.data_root)
            if container:
                mounts = sb.container_mounts(ro=[dev], rw=[workspace, eng.art_root, sb.dir])
            else:
                job_command = shlex.join(sb.prefix(python_roots() + [dev],
                                                   [workspace, eng.art_root, sb.dir],
                                                   cwd=workspace)
                                         + ["bash", "-c", command])
        job = Job(name=exp_id, command=job_command, cwd=workspace, env=env, gpus=gpus,
                  timeout_s=timeout, log_path=art_dir / "job.log", container=container,
                  mounts=mounts, cancelled=lambda: grant.revoked)
        started = now_iso()
        try:
            with budget.waiting() as on_queue:  # the queue wait extends the phase's deadline
                job.on_queue = on_queue
                res = backend.run(job)
        finally:
            self.revoke(token)
        budget.charge(gpu_hours=res.gpu_hours)
        if res.queued:
            budget.add_wait(*res.queued)
        raw = read_json(metrics_file, default={}) if metrics_file.exists() else {}
        fields, problems = parse_result(raw if isinstance(raw, dict) else {"value": raw}, art_dir,
                                        strict=False)
        log_text = job.log_path.read_text(errors="replace") if job.log_path.exists() else ""
        eng.records.append("experiments", {
            "exp_id": exp_id, "task_id": task.id, "round": rnd, "phase": grant.consumer,
            "kind": "experiment", "name": str(payload.get("name") or exp_id)[:200],
            "command": command[:4000], "gpus": gpus, "backend": backend.name,
            "job_id": res.job_id, "started_at": started, "ended_at": now_iso(),
            "wall_s": round(res.wall_s, 3), "gpu_hours": round(res.gpu_hours, 6),
            "queue_s": round(res.queue_s, 3),
            "exit_code": res.exit_code, "status": res.status, "score": None,
            "metrics": scalars(fields["metrics"]), "log_path": str(job.log_path),
            "artifact_dir": str(art_dir),
        })
        eng.register_artifacts(task.id, rnd, art_dir, kind="experiment")
        detail = "; ".join(x for x in [res.detail, *problems] if x)[:1000]
        return {"exp_id": exp_id, "status": res.status, "exit_code": res.exit_code,
                "wall_s": round(res.wall_s, 1), "gpu_hours": round(res.gpu_hours, 4),
                **fields, "log_tail": tail(log_text, 6000), "detail": detail,
                "artifact_dir": str(art_dir), "budget": budget.remaining()}

    # ---- dev evaluation ---------------------------------------------------------------------
    def evaluate(self, grant: Grant, payload: dict) -> dict:
        split = payload.get("split", "dev")
        if split != "dev":
            raise GatewayRefusal(403, "forbidden", "only the dev split can be evaluated on request")
        self._refuse_after_submit(grant)
        eng, budget, task = self.engine, grant.budget, grant.task
        batch = payload.get("candidates")
        sources = self._candidates(batch, task) if batch is not None \
            else [(None, eng.workspace_dir(task.id))]
        reason = budget.exhausted()
        if reason:
            raise GatewayRefusal(402, "budget_exhausted", reason)
        with self._lock:
            self._exp_counter += 1
            call_id = f"{task.id}-r{grant.round:03d}-dev{self._exp_counter:03d}"
        jobs = [(f"{call_id}-{cid}" if cid else call_id, src, cid) for cid, src in sources]
        if batch is None:
            done = [self._dev_eval(grant, *jobs[0])]
        else:
            done = _concurrently(lambda job: self._dev_eval(grant, *job), jobs)
        rows = [row for row, _ in done]
        bar = grant.bar
        rule = bool(eng.protocol.get("submit_on_dev_improvement"))
        best = best_improvement(task, rows, bar, grant.margin) if rule else None
        with self._lock:  # the rows of one call stay together, in request order
            for row in rows:
                eng.records.append("experiments", row)
            # the first improving call submits, as --close-partial-round assumes; a call that
            # outlived its phase (a revoked grant) submits nothing
            if best is not None and grant.submit is None and not grant.revoked:
                grant.submit = {"snapshot": str(Path(rows[best]["artifact_dir"]) / "snapshot"),
                                "score": rows[best]["score"], "bar": bar,
                                "exp_id": rows[best]["exp_id"]}
            else:
                best = None
        if batch is None:
            reply = {**done[0][1], "budget": budget.remaining()}
        else:
            reply = {"results": {cid: r for (cid, _), (_, r) in zip(sources, done)},
                     "budget": budget.remaining()}
        if best is not None:
            reply["submit_required"] = True
            reply["message"] = _submit_message(task, rows[best]["score"], bar, sources[best][0])
        elif rule and bar is not None and grant.submit is None:
            # how far the round is from submitting on its own
            reply["submit_rule"] = {"accepted_dev": bar,
                                    "auto_submit_at": auto_submit_at(task, bar, grant.margin)}
        return reply

    def _candidates(self, raw, task) -> list[tuple[str, Path]]:
        """The (id, directory) pairs of a batch evaluation. A directory must resolve inside the
        task's workspace or the scratch directory; a relative path is relative to the worktree."""
        eng = self.engine
        limit = int(run_constant(eng.protocol, "max_eval_candidates"))
        if not isinstance(raw, list) or not 1 <= len(raw) <= limit:
            raise GatewayRefusal(400, "bad_request",
                                 f"'candidates' must list 1 to {limit} {{id, path}} objects")
        roots = (eng.workspace_dir(task.id), eng.scratch_dir())
        out, seen = [], set()
        for entry in raw:
            if not (isinstance(entry, dict) and set(entry) == {"id", "path"}
                    and isinstance(entry["id"], str) and isinstance(entry["path"], str)):
                raise GatewayRefusal(400, "bad_request",
                                     'every candidate must be {"id": <string>, "path": <string>}')
            cid = entry["id"]
            if not CANDIDATE_ID.fullmatch(cid):
                raise GatewayRefusal(400, "bad_request",
                                     f"candidate id {cid!r} must match {CANDIDATE_ID.pattern}")
            if cid.lower() in seen:  # ids name output directories, which may ignore case
                raise GatewayRefusal(400, "bad_request",
                                     f"candidate id {cid!r} is used twice (ignoring case)")
            seen.add(cid.lower())
            path = Path(entry["path"])
            try:
                path = (path if path.is_absolute() else eng.wt / path).resolve()
                inside = path.is_dir() and any(path == r or r in path.parents for r in roots)
            except (OSError, RuntimeError, ValueError):  # RuntimeError: a symlink loop
                inside = False
            if not inside:
                raise GatewayRefusal(400, "bad_request",
                                     f"candidate {cid}: {entry['path'][:200]} is not a directory "
                                     f"inside workspace/{task.id} or the scratch directory")
            out.append((cid, path))
        return out

    def _dev_eval(self, grant: Grant, exp_id: str, source: Path,
                  cid: str | None) -> tuple[dict, dict]:
        """Score a snapshot of `source` (RunEngine.working_snapshot) on dev and charge its GPU
        hours; returns its experiments row and its reply. A candidate that cannot be scored, e.g. a directory that
        cannot be copied, gets an error result, so the other candidates of its call keep theirs."""
        eng, budget, task = self.engine, grant.budget, grant.task
        out_dir = eng.artifact_dir(task.id, grant.round) / exp_id
        started = now_iso()
        try:
            snapshot = eng.working_snapshot(task, out_dir / "snapshot", source)
            # once the requesting phase is over, nobody uses the result of a job not yet started
            result = eng.run_split_eval(task, snapshot, "dev", out_dir, grant.round, budget=budget,
                                        cancelled=lambda: grant.revoked)
        except Exception as exc:
            if cid is None:
                raise
            snapshot, result = None, _unscored(exc)
        budget.charge(gpu_hours=result["gpu_hours"])
        if result.get("queued"):
            budget.add_wait(*result["queued"])
        row = {
            "exp_id": exp_id, "task_id": task.id, "round": grant.round, "phase": grant.consumer,
            "kind": "eval-dev", "name": f"evaluate dev {cid}" if cid else "evaluate dev",
            "command": "", "gpus": task.eval_gpus, "backend": result["backend"],
            "job_id": result["job_id"], "started_at": started, "ended_at": now_iso(),
            "wall_s": round(result["wall_s"], 3), "gpu_hours": round(result["gpu_hours"], 6),
            "queue_s": round(float(result.get("queue_s") or 0), 3),
            "exit_code": result["exit_code"], "status": result["status"],
            "score": result["score"], "metrics": scalars(result["metrics"]),
            "log_path": result["log_path"], "artifact_dir": str(out_dir),
            "snapshot_sha": sha256_tree(snapshot) if snapshot else None,
        }
        reply = {"status": result["status"], "score": result["score"],
                 "metric": task.primary_metric, "direction": task.direction,
                 "metrics": result["metrics"], "artifacts": result["artifacts"],
                 "info": result["info"], "detail": result.get("detail", ""),
                 "log_tail": result["log_tail"], "artifact_dir": str(out_dir)}
        return row, reply

    def budget(self, grant: Grant, payload: dict) -> dict:
        b = grant.budget
        return {"remaining": b.remaining(), "fractions": b.fractions(),
                "used": b.used().as_dict(), "limits": b.limits}


class Gateway:
    ROUTES = {"/v1/complete": "complete", "/v1/experiment": "experiment",
              "/v1/evaluate": "evaluate", "/v1/budget": "budget"}

    def __init__(self, services: Services, host: str = "127.0.0.1", advertise: str | None = None):
        """Listen on `host` (127.0.0.1: this machine only). `url` names `advertise`, by default
        the bound address, or this machine's name when bound to every interface."""
        self.services = services
        self.advertise = advertise or (socket.gethostname() if host in ("", "0.0.0.0") else None)
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence the default stderr access log
                pass

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802 (http.server API)
                services, method = gateway.services, gateway.ROUTES.get(self.path)
                token = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
                try:
                    grant = services.authorize(token)
                    if method is None:
                        raise GatewayRefusal(404, "not_found", self.path)
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    return self._send(200, services.serve(grant, method, payload))
                except GatewayRefusal as exc:
                    return self._send(exc.status, {"error": exc.code, "message": exc.message})
                except json.JSONDecodeError as exc:
                    return self._send(400, {"error": "bad_request", "message": str(exc)})
                except Exception as exc:  # keep the gateway alive; report to the harness
                    return self._send(500, {"error": "internal", "message": repr(exc)})

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{self.advertise or host}:{port}"

    def start(self) -> "Gateway":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def file_entries(root: Path, max_hash_bytes: int = 512 << 20) -> list[dict]:
    out = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        out.append({"path": str(path), "bytes": size,
                    "sha256": sha256_file(path) if size <= max_hash_bytes else None})
    return out
