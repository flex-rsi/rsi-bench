"""Summaries computed from records/ only (works on a worktree or on any checkout of a branch).

Per task, a summary merges the dicts of every registered metric: the built-ins in metrics.py and
the metrics of loaded plugins (docs/configuration.md#metrics).
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import REGISTRY, MetricError, RunRecords, flat
from .records import Records

# task keys the main markdown table shows, and those that are not metrics; the rest go below it
TABLE_KEYS = ("task_id", "tier", "direction", "anchors", "rounds", "accepted", "rejected",
              "baseline_val", "best_val", "test", "normalized", "first_round_gain", "later_gain",
              "val_auc", "test_auc", "ended_reason", "harness_commits", "violations")


def protocol_id(run: dict) -> str:
    """Fingerprint of everything that must match for two runs to be comparable: the frozen protocol
    constants, the budgets and the task versions (model, mode and seed are reported separately)."""
    import hashlib
    key = {"protocol": run["protocol"], "budgets": run["budgets"],
           "tasks": [(t["id"], t["version_hash"]) for t in run["tasks"]]}
    sandbox = run.get("sandbox") or {}
    if sandbox.get("enabled"):  # absent for unsandboxed runs, so their ids stay as they were
        key["sandbox"] = {"read_only": sorted(sandbox.get("read_only", []))}
        if sandbox.get("git_view"):  # phases can read the run's history; absent before, so
            key["sandbox"]["git_view"] = True  # earlier runs keep their ids
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def summarize(records_dir: Path, metrics: list[str] | None = None) -> dict:
    """Per task, the merged dicts of every registered metric (or of the `metrics` named)."""
    names = list(REGISTRY) if metrics is None else list(metrics)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise MetricError(f"unknown metric(s) {', '.join(unknown)}; "
                          f"registered: {', '.join(REGISTRY)}")
    records = RunRecords(records_dir)
    run = records.run
    out = {"run_id": run["run_id"], "mode": run["mode"], "model": run["model"]["name"],
           "effort": run["model"].get("effort"), "seed": run["seed"],
           "base": run["base"]["ref"], "improver_source": run.get("improver_source"),
           "executor_source": run.get("executor_source"),
           "protocol_id": protocol_id(run), "metrics": names,
           "sandboxed": bool((run.get("sandbox") or {}).get("enabled")),
           "tasks": [_task_summary(records, task, names) for task in run["tasks"]]}
    out["model_calls"] = len(records.files.get("usage", []))
    return out


def _task_summary(records: RunRecords, task: dict, names: list[str]) -> dict:
    entry = {"task_id": task["id"], "tier": task["tier"],
             "direction": task.get("direction", "maximize"), "anchors": task.get("anchors")}
    source = dict.fromkeys(entry, "the summary itself")
    for name in names:
        values = REGISTRY[name](records, task)
        if not isinstance(values, dict):
            raise MetricError(f"metric {name!r} returned {type(values).__name__}, not a dict")
        for key, value in values.items():
            if key in source:
                raise MetricError(f"key {key!r} comes from both {source[key]} and "
                                  f"metric {name!r}")
            source[key] = f"metric {name!r}"
            entry[key] = value
    return entry


def to_markdown(summary: dict, rounds: list[dict] | None = None) -> str:
    lines = [f"# {summary['run_id']}", "",
             f"mode `{summary['mode']}` · model `{summary['model']}` (effort {summary['effort']}) · "
             f"seed {summary['seed']} · base `{summary['base']}` · model calls {summary['model_calls']}",
             ""]
    lines.append("| task | rounds | acc/rej | baseline val | best val | test | norm test | "
                 "1st-round gain | later gain | val AUC | test AUC | ended | E/I commits | violations |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for t in summary["tasks"]:
        norm = (t["normalized"] or {}).get("test")
        lines.append(
            f"| {t['task_id']} | {t['rounds']} | {t['accepted']}/{t['rejected']} | "
            f"{_fmt(t['baseline_val'])} | {_fmt(t['best_val'])} | {_fmt(t['test'])} | {_fmt(norm)} | "
            f"{_fmt(t['first_round_gain'])} | {_fmt(t['later_gain'])} | "
            f"{_fmt(t['val_auc'])} | {_fmt(t['test_auc'])} | {t['ended_reason']} | "
            f"{t['harness_commits']['E']}/{t['harness_commits']['I']} | {t['violations']} |")
    lines += ["", "Gains are in normalized val units: `1st-round gain` is what the first attempt "
              "achieved over the starter; `later gain` is what the remaining rounds (iteration and "
              "harness edits) added on top."]
    lines += _other_metrics(summary["tasks"])
    if rounds:
        lines += ["", "| task | round | decision | val | best | E lines | I lines | budget used |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        for r in rounds:
            hc = r.get("harness_changes") or {}
            lines.append(
                f"| {r['task_id']} | {r['round']} | {r['decision']} | {_fmt(r['val'].get('score'))} | "
                f"{_fmt(r.get('best_val_after'))} | {(hc.get('E') or {}).get('lines', 0)} | "
                f"{(hc.get('I') or {}).get('lines', 0)} | {r['budget_used_frac']:.2f} |")
    return "\n".join(lines) + "\n"


def _other_metrics(tasks: list[dict]) -> list[str]:
    """The task keys the main table leaves out (nested keys joined with dots), one row each."""
    values = [flat({k: v for k, v in t.items() if k not in TABLE_KEYS}) for t in tasks]
    keys = list(dict.fromkeys(k for v in values for k in v))
    if not keys:
        return []
    lines = ["", "Other metrics (see docs/configuration.md#metrics).", "",
             "| metric | " + " | ".join(t["task_id"] for t in tasks) + " |",
             "|---|" + "---:|" * len(tasks)]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(_fmt(v.get(key)) for v in values) + " |")
    return lines


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def report(worktree: Path, fmt: str = "md") -> str:
    records_dir = Path(worktree) / "records"
    summary = summarize(records_dir)
    if fmt == "json":
        return json.dumps(summary, indent=2)
    return to_markdown(summary, Records(Path(worktree)).read("rounds"))
