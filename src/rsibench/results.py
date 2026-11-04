"""Append-only results index on its own branch (default `results`).

Run branches are never merged anywhere. To get one table across runs, CI (or a person) appends a
summary row for a run branch to `runs.jsonl` on the results branch. The commit is built with git
plumbing, so no worktree or checkout is needed and concurrent CI jobs only race on the ref update.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from . import gitops
from .metrics import MetricError, flat
from .records import SchemaError, validate
from .report import summarize
from .util import now_iso


def summary_row(repo: Path, ref: str, accept_unsandboxed: str | None = None) -> dict:
    """The run's summary as one result row. `accept_unsandboxed` (a reason) counts a run made
    without the sandbox as a standard one: the row says `sandboxed: true` and keeps the reason in
    `sandbox_waiver`; the run's own records are not changed."""
    with tempfile.TemporaryDirectory(prefix="rsib-results-") as tmp:
        gitops.archive_extract(repo, ref, "records", Path(tmp))
        summary = summarize(Path(tmp) / "records")
    for task in summary["tasks"]:
        taken = _clashes(task)
        if taken:
            raise MetricError(f"{ref}, task {task['task_id']}: metric key(s) {', '.join(taken)} "
                              "clash with the results table's own fields or standard columns; "
                              "rename them")
    if accept_unsandboxed and not summary.get("sandboxed"):
        summary["sandboxed"] = True
        summary["sandbox_waiver"] = {"reason": accept_unsandboxed, "at": now_iso()}
    summary["schema_version"] = "1.0"
    summary["ref"] = ref
    summary["commit"] = gitops.rev_parse(repo, ref)
    summary["indexed_at"] = now_iso()
    errors = validate("result_row", summary)
    if errors:
        raise SchemaError("result row: " + "; ".join(errors))
    return summary


def append(repo: Path, ref: str, results_branch: str = "results",
           accept_unsandboxed: str | None = None) -> dict:
    row = summary_row(repo, ref, accept_unsandboxed)
    full = f"refs/heads/{results_branch}"
    parent = gitops.rev_parse(repo, full) if gitops.ref_exists(repo, full) else None
    existing = gitops.show(repo, full, "runs.jsonl") if parent else ""
    lines = [json.loads(x) for x in (existing or "").splitlines() if x.strip()]
    # One row per (run, commit, sandbox status): re-indexing the same commit is a no-op, while a
    # waiver for an indexed unsandboxed run adds a newer row that tables use instead.
    if any(r.get("run_id") == row["run_id"] and r.get("commit") == row["commit"]
           and bool(r.get("sandboxed")) == bool(row.get("sandboxed")) for r in lines):
        return row
    text = (existing or "") + json.dumps(row) + "\n"
    blob = gitops.git(["hash-object", "-w", "--stdin"], repo, input_text=text).strip()
    entries = []
    if parent:
        for line in gitops.git(["ls-tree", full], repo).splitlines():
            if line.split("\t", 1)[1] != "runs.jsonl":
                entries.append(line)
    entries.append(f"100644 blob {blob}\truns.jsonl")
    readme = ("# Results index\n\nOne JSON line per indexed run branch commit; written by "
              "`rsib results append`. Do not edit by hand.\n")
    if not any(e.endswith("\tREADME.md") for e in entries):
        rblob = gitops.git(["hash-object", "-w", "--stdin"], repo, input_text=readme).strip()
        entries.append(f"100644 blob {rblob}\tREADME.md")
    tree = gitops.git(["mktree"], repo, input_text="\n".join(entries) + "\n").strip()
    args = ["commit-tree", tree, "-m", f"index {row['run_id']} @ {row['commit'][:10]}"]
    if parent:
        args[2:2] = ["-p", parent]
    commit = gitops.git(args, repo, env=gitops.bot_env()).strip()
    old = parent or "0" * 40
    gitops.git(["update-ref", full, commit, old], repo)
    return row


# ---- reading and tabulating the index ----------------------------------------------------------

METRICS = ("normalized_test", "test", "best_val", "first_round_gain", "later_gain", "val_auc",
           "test_auc", "rounds", "accepted", "usd", "hours")
GROUP = ("task", "model", "effort", "condition", "base", "protocol")
# the table's own fields, which are never columns, and the standard columns flatten() derives;
# a metric key with one of these names would be shadowed by them
FIELDS = GROUP + ("run_id", "seed", "ended", "runs", "seeds")
ALIASES = ("normalized_test", "usd", "hours")
# markdown label and number format of the standard columns; any other column shows its name
FORMATS = {"normalized_test": ("norm test", ".3f"), "test": ("test", ".4f"),
           "best_val": ("best val", ".4f"), "first_round_gain": ("1st-round gain", ".3f"),
           "later_gain": ("later gain", ".3f"), "val_auc": ("val AUC", ".3f"),
           "test_auc": ("test AUC", ".3f"), "rounds": ("rounds", ".1f"),
           "accepted": ("accepted", ".1f"), "usd": ("USD", ".2f"), "hours": ("hours", ".2f")}


def load_index(repo: Path, ref: str = "results") -> list[dict]:
    """Rows of runs.jsonl on the results branch (local `results`, else `origin/results`)."""
    for candidate in (ref, f"origin/{ref}"):
        text = gitops.show(repo, candidate, "runs.jsonl")
        if text is not None:
            return valid_rows([json.loads(x) for x in text.splitlines() if x.strip()])
    raise FileNotFoundError(f"no runs.jsonl on {ref} or origin/{ref}; run `rsib results append` first")


def rows_from_branches(repo: Path, pattern: str = "run/*") -> list[dict]:
    """Summaries computed directly from run branches (local and remote-tracking), no index needed."""
    import fnmatch

    refs = []
    for ref in gitops.git(["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"],
                          repo).split():
        if ref.startswith("refs/heads/"):
            short = ref[len("refs/heads/"):]
        else:
            short = ref[len("refs/remotes/"):].split("/", 1)[-1]
        if fnmatch.fnmatch(short, pattern):  # '*' also matches '/' here (multi-level names)
            refs.append(ref)
    rows, seen = [], set()
    for ref in refs:
        commit = gitops.rev_parse(repo, ref)
        if commit in seen or gitops.show(repo, ref, "records/run.json") is None:
            continue
        seen.add(commit)
        rows.append(summary_row(repo, ref))
    return rows


def valid_rows(rows: list[dict]) -> list[dict]:
    """Rows that match result_row.schema.json; others are reported and skipped."""
    good = []
    for row in rows:
        errors = validate("result_row", row)
        if errors:
            print(f"skipping invalid result row {row.get('run_id')}: {errors[0]}", file=sys.stderr)
        else:
            good.append(row)
    return good


def latest_per_run(rows: list[dict]) -> list[dict]:
    """Keep the most informative row of each run: finished beats unfinished, then most recent."""
    best: dict[str, dict] = {}
    for row in rows:
        key = row["run_id"]
        done = all(t.get("test") is not None for t in row.get("tasks", []))
        rank = (done, row.get("indexed_at", ""))
        if key not in best or rank >= best[key][0]:
            best[key] = (rank, row)
    return [r for _, r in best.values()]


def condition(row: dict) -> str:
    src, e_src = row.get("improver_source"), row.get("executor_source")
    cond = f"probe:{src['ref']}" if src else row["mode"]
    return f"{cond}+E:{e_src['ref']}" if e_src else cond


def flatten(rows: list[dict]) -> list[dict]:
    """One record per (run, task): the grouping fields and every numeric value of the task's
    summary, nested keys joined with dots (cost.usd), plus the standard columns normalized_test
    (= normalized.test), usd (= cost.usd) and hours (= cost.wall_s / 3600)."""
    out = []
    for row in rows:
        for t in row.get("tasks", []):
            cost = t.get("cost") or {}
            rec = {k: v for k, v in flat(t).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
            rec.update({  # the table's own fields win over metric keys of the same name
                "normalized_test": (t.get("normalized") or {}).get("test"), "usd": cost.get("usd"),
                "hours": (cost.get("wall_s") or 0) / 3600 if cost else None,
                "run_id": row["run_id"], "task": t["task_id"], "model": row["model"],
                "effort": row.get("effort"), "condition": condition(row), "base": row.get("base"),
                "protocol": row.get("protocol_id"), "seed": row.get("seed"),
                "ended": t.get("ended_reason"),
            })
            out.append(rec)
    return out


def _clashes(task: dict) -> list[str]:
    """Keys of a task summary named like a field or a standard column of the table."""
    return [k for k in task if k in FIELDS or k in ALIASES]


def _stats(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "sem": None}
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "sem": std / n ** 0.5 if n > 1 else None}


def build_table(rows: list[dict], filters: dict | None = None,
                columns: list[str] | None = None, include_unsandboxed: bool = False) -> dict:
    """Mean, std and n per group of the `columns` (default METRICS); the gains always compare
    normalized_test."""
    columns = list(columns or METRICS)
    fields = [c for c in columns if c in FIELDS]
    if fields:
        raise MetricError(f"table fields are not columns: {', '.join(fields)}; a column is a "
                          "numeric key of the task summaries")
    rows = latest_per_run(rows)
    excluded = 0
    if not include_unsandboxed:  # rows from before the field existed were not sandboxed
        excluded = len({r["run_id"] for r in rows if r.get("sandboxed") is not True})
        rows = [r for r in rows if r.get("sandboxed") is True]
    records = flatten(rows)
    for key, value in (filters or {}).items():
        if value is not None:
            records = [r for r in records if str(r.get(key)) == str(value)]
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in GROUP), []).append(r)
    table, norm = [], {}
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        recs = groups[key]
        entry = dict(zip(GROUP, key))
        entry["runs"] = len(recs)
        entry["seeds"] = sorted({r["seed"] for r in recs})
        for m in columns:
            entry[m] = _stats([r.get(m) for r in recs])
        norm[key] = _stats([r["normalized_test"] for r in recs])
        table.append(entry)

    gains = []
    for (task, model, effort, cond, base, proto), cur in norm.items():
        if cond == "fixed-i":
            ref_cond, kind = "base", "fixed-i - base"
        elif cond.startswith("probe:"):
            ref_cond, kind = "fixed-i", f"{cond} - fixed-i"
        elif cond == "full":
            ref_cond, kind = "base", "full - base"
        else:
            continue
        ref = norm.get((task, model, effort, ref_cond, base, proto))
        if ref and ref["mean"] is not None and cur["mean"] is not None:
            gains.append({"task": task, "model": model, "effort": effort, "base": base, "protocol": proto,
                          "comparison": kind, "gain": cur["mean"] - ref["mean"],
                          "n": (cur["n"], ref["n"])})
    return {"columns": columns, "groups": table, "gains": gains,
            "runs": len({r["run_id"] for r in records}), "excluded_unsandboxed": excluded}


def _f(stat: dict | None, spec: str = ".3f") -> str:
    if not stat or stat.get("mean") is None:
        return "-"
    text = format(stat["mean"], spec)
    if stat.get("sem") is not None:
        text += " ± " + format(stat["sem"], spec)
    return text


def table_markdown(t: dict) -> str:
    columns = t.get("columns", METRICS)
    formats = [FORMATS.get(c, (c, ".4g")) for c in columns]
    lines = [f"{t['runs']} run(s). Values are mean ± standard error over runs (seeds); "
             "`norm test` is (test - baseline) / (reference - baseline). `1st-round gain` is the "
             "normalized val gain of the first attempt, `later gain` what the later rounds added."
             + (f" {t['excluded_unsandboxed']} unsandboxed run(s) left out (--include-unsandboxed)."
                if t.get("excluded_unsandboxed") else ""), "",
             "| task | model | effort | condition | base | protocol | runs | "
             + " | ".join(label for label, _ in formats) + " |",
             "|---|---|---|---|---|---|---:|" + "---:|" * len(columns)]
    for e in t["groups"]:
        cells = " | ".join(_f(e[c], spec) for c, (_, spec) in zip(columns, formats))
        lines.append(
            f"| {e['task']} | {e['model']} | {e['effort'] or '-'} | {e['condition']} | {e['base']} | "
            f"{e['protocol']} | {e['runs']} | {cells} |")
    if t["gains"]:
        lines += ["", "| task | model | effort | base | comparison | gain (norm test) | runs |",
                  "|---|---|---|---|---|---:|---|"]
        for g in t["gains"]:
            lines.append(f"| {g['task']} | {g['model']} | {g['effort'] or '-'} | {g['base']} | "
                         f"{g['comparison']} | {g['gain']:+.3f} | {g['n'][0]} vs {g['n'][1]} |")
    return "\n".join(lines) + "\n"


def table_csv(t: dict) -> str:
    import csv
    import io
    columns = t.get("columns", METRICS)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["task", "model", "effort", "condition", "base", "protocol", "runs"] +
               [f"{m}_{s}" for m in columns for s in ("mean", "std", "n")])
    for e in t["groups"]:
        w.writerow([e["task"], e["model"], e["effort"], e["condition"], e["base"], e["protocol"], e["runs"]] +
                   [e[m][s] for m in columns for s in ("mean", "std", "n")])
    return buf.getvalue()
