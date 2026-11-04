"""Metrics: functions of one run's records that return a dict of values per task.

    from rsibench.metrics import metric

    @metric("best_round")
    def best_round(run, task):          # run: RunRecords, task: the task's entry in run.json
        accepted = [r["round"] for r in run.rows("rounds", task["id"]) if r["decision"] == "accept"]
        return {"best_round": accepted[-1] if accepted else 0}

`report.summarize` runs every registered metric and merges the dicts of each task; a key may come
from one metric only. The built-ins below make up the standard summary. Plugins register more
metrics when they are imported (`load_plugins`: module paths or .py files, from `--metrics a,b`
or $RSIB_METRICS).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

from . import localenv
from .records import latest_posthoc
from .util import read_json, read_jsonl

REGISTRY: dict[str, Callable[[RunRecords, dict], dict]] = {}
_LOADED: set[str] = set()


class MetricError(ValueError):
    pass


def metric(name: str):
    """Register `fn(run: RunRecords, task: dict) -> dict` as metric `name`."""
    def register(fn):
        if REGISTRY.get(name, fn) is not fn:
            raise MetricError(f"metric {name!r} is already registered by "
                              f"{getattr(REGISTRY[name], '__module__', '?')}")
        REGISTRY[name] = fn
        return fn
    return register


class RunRecords:
    """One run's records, read once: `run` is run.json, `files[name]` the rows of
    records/<name>.jsonl."""

    def __init__(self, records_dir: Path):
        records_dir = Path(records_dir)
        self.run = read_json(records_dir / "run.json")
        if self.run is None:
            raise FileNotFoundError(f"{records_dir / 'run.json'} not found; "
                                    "is this a records/ directory?")
        self.files = {p.stem: read_jsonl(p) for p in sorted(records_dir.glob("*.jsonl"))}

    def rows(self, name: str, task_id: str) -> list[dict]:
        """The rows of records/<name>.jsonl that belong to one task, in file order."""
        return [r for r in self.files.get(name, []) if r.get("task_id") == task_id]


# ---- plugins ---------------------------------------------------------------------------------

def plugin_names(flag: str | None = None) -> list[str]:
    """Plugin entries from a `--metrics` value, else from $RSIB_METRICS (comma separated)."""
    text = flag if flag is not None else (localenv.lookup("RSIB_METRICS") or "")
    return [name.strip() for name in text.split(",") if name.strip()]


def load_plugins(names: list[str]) -> None:
    """Import each entry, a module path (pkg.mod) or a .py file; importing registers its metrics."""
    for entry in names:
        is_file = entry.endswith(".py") or "/" in entry or os.sep in entry
        key = str(Path(entry).expanduser().resolve()) if is_file else entry
        if key in _LOADED:
            continue
        try:
            if is_file:
                _load_file(Path(key))
            else:
                importlib.import_module(entry)
        except Exception as exc:
            raise MetricError(f"cannot load metrics plugin {entry!r}: {exc}") from exc
        _LOADED.add(key)


def _load_file(path: Path) -> None:
    name = f"rsib_metrics_{len(_LOADED)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"{path} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look modules up there
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise


# ---- helpers for metric functions ----------------------------------------------------------

def normalize(score, anchors: dict | None):
    """(score - baseline) / (reference - baseline): 0 is the starter, 1 the reference."""
    anchors = anchors or {}
    b, r = anchors.get("baseline"), anchors.get("reference")
    if score is None or b is None or r is None or r == b:
        return None
    return (score - b) / (r - b)


def step_auc(points: list[tuple[float, float]]) -> float | None:
    """Area under a step curve on x in [0, 1]; before the first point the first y holds."""
    points = [(x, y) for x, y in points if y is not None]
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    area, prev_x, prev_y = 0.0, 0.0, points[0][1]
    for x, y in points:
        x = min(max(x, 0.0), 1.0)
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y
    return area + (1.0 - prev_x) * prev_y


def flat(values: dict, prefix: str = "") -> dict:
    """Nested dicts as one level, their keys joined with dots ({"cost": {"usd": 1}} -> cost.usd)."""
    out = {}
    for key, value in values.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "maximize" else a < b


def _diff(a, b):
    return a - b if a is not None and b is not None else None


def _final(run: RunRecords, task: dict) -> dict:
    """The task's latest final.jsonl line ({} before the test evaluation); a retry wins."""
    finals = run.rows("final", task["id"])
    return finals[-1] if finals else {}


def _baseline_val(rows: list[dict]):
    return next((r["val"]["score"] for r in rows if r["decision"] == "baseline"), None)


def _best_val(run: RunRecords, task: dict):
    best = _final(run, task).get("best_val")
    rows = run.rows("rounds", task["id"])
    return rows[-1]["best_val_after"] if best is None and rows else best


def _posthoc(run: RunRecords, task: dict) -> dict:
    return {rnd: row for (_, rnd), row in latest_posthoc(
        run.rows("posthoc", task["id"])).items()}


# ---- built-in metrics ------------------------------------------------------------------------

@metric("outcome")
def outcome(run: RunRecords, task: dict) -> dict:
    """Round decisions, baseline and best val, and the final test score."""
    rows = run.rows("rounds", task["id"])
    final = _final(run, task)
    decisions = [r["decision"] for r in rows if r["round"] > 0]
    return {"rounds": len(decisions), "accepted": decisions.count("accept"),
            "rejected": decisions.count("reject"), "no_change": decisions.count("no-change"),
            "errors": decisions.count("error"), "baseline_val": _baseline_val(rows),
            "best_val": _best_val(run, task), "test": (final.get("test") or {}).get("score"),
            "normalized": final.get("normalized"), "ended_reason": final.get("ended_reason")}


@metric("gains")
def gains(run: RunRecords, task: dict) -> dict:
    """Normalized val gain of the first attempt over the starter, and what later rounds added."""
    rows = run.rows("rounds", task["id"])
    after_first = next((r["best_val_after"] for r in rows if r["round"] == 1), None)
    base, first, best = (normalize(x, task.get("anchors"))
                         for x in (_baseline_val(rows), after_first, _best_val(run, task)))
    return {"first_round_gain": _diff(first, base), "later_gain": _diff(best, first)}


@metric("auc")
def auc(run: RunRecords, task: dict) -> dict:
    """Area under the normalized best-so-far val curve against the share of budget used, and the
    same for the test score of the current accepted version (needs `rsib posthoc`)."""
    rows, anchors, ph = run.rows("rounds", task["id"]), task.get("anchors"), _posthoc(run, task)
    val_curve = [(r["budget_used_frac"], normalize(r["best_val_after"], anchors)) for r in rows]
    test_curve, current = [], None
    for r in rows:
        if r.get("accepted_commit") and r["round"] in ph:
            current = ph[r["round"]]["score"]
        test_curve.append((r["budget_used_frac"], normalize(current, anchors)))
    return {"val_auc": step_auc(val_curve), "test_auc": step_auc(test_curve) if ph else None,
            "posthoc_points": len(ph)}


@metric("acceptance")
def acceptance(run: RunRecords, task: dict) -> dict:
    """Accepted on val, but worse on test than the previous accepted version (needs posthoc)."""
    ph, direction = _posthoc(run, task), task.get("direction", "maximize")
    accepted = [r for r in run.rows("rounds", task["id"]) if r.get("accepted_commit")]
    count = judged = 0
    for prev, cur in zip(accepted, accepted[1:]):
        a, b = ph.get(prev["round"]), ph.get(cur["round"])
        if a and b and a["score"] is not None and b["score"] is not None:
            judged += 1
            if b["score"] != a["score"] and not _better(b["score"], a["score"], direction):
                count += 1
    return {"false_acceptance": {"count": count, "judged": judged}}


@metric("harness")
def harness(run: RunRecords, task: dict) -> dict:
    """Harness commits and changed lines by scope (E = executor + core, I = improver)."""
    commits, lines = {"E": 0, "I": 0}, {"E": 0, "I": 0}
    for r in run.rows("rounds", task["id"]):
        for scope in ("E", "I"):
            info = (r.get("harness_changes") or {}).get(scope) or {}
            if info.get("commit"):
                commits[scope] += 1
                lines[scope] += int(info.get("lines", 0))
    return {"harness_commits": commits, "harness_lines": lines}


@metric("cost")
def cost(run: RunRecords, task: dict) -> dict:
    """Violations, the cumulative cost at the last recorded round, and the model calls of the
    val and final test evaluations (metered, never part of the budget)."""
    rows = run.rows("rounds", task["id"])
    calls = [u for u in run.rows("usage", task["id"])
             if u.get("phase") in ("eval-val", "eval-test")]
    return {"violations": len(run.rows("violations", task["id"])),
            "cost": rows[-1]["cumulative"] if rows else None,
            "eval_model": {
                "calls": len(calls),
                # input tokens as the budget counts them: cache reads and writes included
                "input_tokens": sum(int(u["input_tokens"]) + int(u.get("cache_read_tokens") or 0)
                                    + int(u.get("cache_write_tokens") or 0) for u in calls),
                "output_tokens": sum(int(u["output_tokens"]) for u in calls),
                "usd": round(sum(float(u.get("usd") or 0) for u in calls), 6)}}


@metric("progress")
def progress(run: RunRecords, task: dict) -> dict:
    """late_gain: normalized best val at the end minus at the last round that ended within the
    first half of the budget; near 0 when improvement saturated, None if the task ended there."""
    rows, anchors = run.rows("rounds", task["id"]), task.get("anchors")
    if not rows or rows[-1]["budget_used_frac"] <= 0.5:
        return {"late_gain": None}
    half = next((r["best_val_after"] for r in reversed(rows) if r["budget_used_frac"] <= 0.5),
                None)
    return {"late_gain": _diff(normalize(_best_val(run, task), anchors), normalize(half, anchors))}
