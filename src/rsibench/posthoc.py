"""Post-hoc test evaluation of every accepted version, after the run is over.

During a run the agent never sees a test score. Scoring each accepted snapshot on the test
split afterwards gives the true improvement curve, the val/test gap, and the false-acceptance
rate (accepted on val, worse on test). Results go to records/posthoc.jsonl.
"""

from __future__ import annotations

from pathlib import Path

from . import gitops
from .contracts import scalars
from .orchestrator.engine import RunEngine
from .records import latest_posthoc
from .util import now_iso


class RedoError(ValueError):
    """A redo request does not select accepted rounds."""


def _validate_redo(engine: RunEngine, redo: list[int] | None, redo_reason: str | None) -> None:
    if redo is None:
        if redo_reason is not None:
            raise RedoError("--redo-reason requires --redo")
        return
    if not redo:
        raise RedoError("--redo requires at least one accepted round")
    task_ids = {task.id for task in engine.tasks}
    accepted = {r["round"] for r in engine.records.read("rounds")
                if r["task_id"] in task_ids and r.get("accepted_commit")}
    invalid = set(redo) - accepted
    if invalid:
        raise RedoError("--redo rounds are not accepted: " + ", ".join(map(str, sorted(invalid))))


def posthoc(worktree: Path, every: int = 1, backend: str = "local", site: dict | None = None,
            log=print, *, redo: list[int] | None = None,
            redo_reason: str | None = None) -> list[dict]:
    engine = RunEngine(worktree, backend=backend, site=site, log=log)
    _validate_redo(engine, redo, redo_reason)
    with engine.serving(engine.tasks):  # for evaluators of tasks with model access
        added = score_missing(engine, every, log, redo=redo, redo_reason=redo_reason)
    if added:
        gitops.commit(engine.wt, "posthoc: test scores of accepted versions",
                      {"Rsib-Run": engine.branch, "Rsib-Scope": "posthoc"}, ["records"])
    return added


def score_missing(engine: RunEngine, every: int = 1, log=print, *,
                  redo: list[int] | None = None, redo_reason: str | None = None) -> list[dict]:
    """Score every accepted version not yet in posthoc.jsonl (every `every`-th, plus the last).
    With `redo`, score only the selected accepted rounds, appending superseding rows.
    The caller provides the serving context and commits the records."""
    _validate_redo(engine, redo, redo_reason)
    latest = latest_posthoc(engine.records.read("posthoc"))
    requested = set(redo) if redo is not None else None
    added = []
    for task in engine.tasks:
        rounds = [r for r in engine.records.read("rounds")
                  if r["task_id"] == task.id and r.get("accepted_commit")]
        picked = ([r for r in rounds if r["round"] in requested] if requested is not None else
                  [r for i, r in enumerate(rounds)
                   if i % max(1, every) == 0 or i == len(rounds) - 1])
        for r in picked:
            key = (task.id, r["round"])
            if requested is not None:
                row = _score(engine, task, r, log, redo=True, superseded=latest.get(key),
                             redo_reason=redo_reason)
            elif key not in latest:
                row = _score(engine, task, r, log)
            else:
                continue
            added.append(row)
            latest[key] = row
    return added


def _score(engine: RunEngine, task, r: dict, log, *, redo: bool = False,
           superseded: dict | None = None, redo_reason: str | None = None) -> dict:
    """Score the version round `r` accepted on test and append its posthoc.jsonl line."""
    commit = r["accepted_commit"]
    name = f"posthoc-r{r['round']:03d}"
    out_dir = engine.hidden_eval_dir(task.id, name)
    if redo:
        attempt = 1
        while True:
            out_dir = engine.hidden_eval_dir(task.id, f"{name}-redo{attempt}")
            try:
                out_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                attempt += 1
    workspace = engine.snapshot_workspace(task, commit, out_dir / "snapshot")
    result = engine.run_split_eval(task, workspace, "test", out_dir, r["round"], consumer="posthoc")
    row = {"task_id": task.id, "round": r["round"], "commit": commit,
           "label": "baseline" if r["decision"] == "baseline" else "accepted",
           "status": result["status"], "score": result["score"],
           "metrics": scalars(result["metrics"], task.public_metrics),
           "val_score": r["val"].get("score"),
           "evaluated_at": now_iso()}
    if redo:
        row.update(redo=True, supersedes=superseded.get("evaluated_at") if superseded else None)
        if redo_reason is not None:
            row["redo_reason"] = redo_reason
    engine.records.append("posthoc", row)
    log(f"[{task.id}] r{r['round']}: val={row['val_score']} test={row['score']}")
    return row
