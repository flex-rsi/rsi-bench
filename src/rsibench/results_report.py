"""One markdown report of every run, built from the run branches.

    rsib results report [--branches PATTERN ...] [--include-unsandboxed] [--output RESULTS.md]
                        [--commit-to results] [--push origin]

For each run branch (local or remote-tracking, newest commit per run) it reads `records/` and
renders:

- an overview with one row per run across all tasks;
- per task, a score table (the primary test metric, the task's secondary test metrics, best val,
  AUCs), a process table (rounds, gains, false acceptances, harness commits, violations, cost,
  end reason) and the post-hoc test curve of every run's accepted versions;
- the cross-run table of `rsib results table` (mean ± s.e. per condition, gains between conditions);
- the sandbox waivers recorded in the results index (`rsib results append --accept-unsandboxed`).

Unsandboxed runs are left out unless `include_unsandboxed`, as in `rsib results table`.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from . import gitops
from .records import latest_posthoc
from .results import build_table, load_index, summary_row, table_markdown
from .util import now_iso

DEFAULT_PATTERNS = ("run/*", "*/run/*")


def _jsonl(text: str | None) -> list[dict]:
    return [json.loads(x) for x in (text or "").splitlines() if x.strip()]


def _run_refs(repo: Path, patterns) -> list[str]:
    refs = []
    for ref in gitops.git(["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"],
                          repo).split():
        short = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else \
            ref[len("refs/remotes/"):].split("/", 1)[-1]
        if any(fnmatch.fnmatch(short, p) for p in patterns):
            refs.append(ref)
    return refs


def _waivers(repo: Path, index_ref: str) -> dict[str, dict]:
    try:
        rows = load_index(repo, index_ref)
    except FileNotFoundError:
        return {}
    out = {}
    for row in rows:  # later rows win
        if row.get("sandbox_waiver"):
            out[row["run_id"]] = row["sandbox_waiver"]
    return out


def collect(repo: Path, patterns=DEFAULT_PATTERNS, index_ref: str = "results") -> list[dict]:
    """One entry per run: its summary row plus the records the report needs."""
    newest: dict[str, tuple[int, str]] = {}
    for ref in _run_refs(repo, patterns):
        text = gitops.show(repo, ref, "records/run.json")
        if text is None:
            continue
        run_id = json.loads(text)["run_id"]
        when = int(gitops.git(["log", "-1", "--format=%ct", ref], repo).strip() or 0)
        if run_id not in newest or when > newest[run_id][0]:
            newest[run_id] = (when, ref)
    waivers = _waivers(repo, index_ref)
    entries = []
    for run_id, (_, ref) in sorted(newest.items()):
        row = summary_row(repo, ref)
        if not row.get("sandboxed") and run_id in waivers:
            row["sandboxed"], row["sandbox_waiver"] = True, waivers[run_id]
        entries.append({
            "ref": ref, "row": row,
            "run": json.loads(gitops.show(repo, ref, "records/run.json")),
            "final": _jsonl(gitops.show(repo, ref, "records/final.jsonl")),
            "posthoc": _jsonl(gitops.show(repo, ref, "records/posthoc.jsonl")),
        })
    return entries


# ---- rendering ---------------------------------------------------------------------------------

def _f(x, spec: str = ".4f") -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, (int, float)):
        return format(x, spec) if isinstance(x, float) else str(x)
    return str(x)


def _sandbox(row: dict) -> str:
    if row.get("sandbox_waiver"):
        return "waiver"
    return "yes" if row.get("sandboxed") else "no"


def _short(ref: str) -> str:
    for prefix in ("refs/heads/", "refs/remotes/"):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            return ref.split("/", 1)[1] if prefix == "refs/remotes/" else ref
    return ref


def _task_rows(entries: list[dict]) -> dict[str, list[tuple[dict, dict, dict]]]:
    """task id -> [(entry, task summary, final line)]"""
    out: dict[str, list] = {}
    for e in entries:
        finals = {f["task_id"]: f for f in e["final"]}
        for t in e["row"]["tasks"]:
            out.setdefault(t["task_id"], []).append((e, t, finals.get(t["task_id"], {})))
    return out


def render(entries: list[dict], include_unsandboxed: bool = False,
           generated_at: str | None = None) -> str:
    shown = [e for e in entries if include_unsandboxed or e["row"].get("sandboxed") is True]
    left_out = len(entries) - len(shown)
    lines = ["# Results", "",
             f"Generated {generated_at or now_iso()} by `rsib results report` from "
             f"{len(entries)} run branch(es)"
             + (f"; {left_out} unsandboxed run(s) left out (`--include-unsandboxed`)."
                if left_out else ".")
             + " Do not edit by hand.", ""]
    if not shown:
        return "\n".join(lines + ["No runs to report.", ""])

    by_task = _task_rows(shown)
    norm = lambda t: (t.get("normalized") or {}).get("test")  # noqa: E731
    lines += ["## Overview", "",
              "| task | model | effort | mode | sandbox | norm test | test | best val | "
              "rounds (accepted) | USD | hours | ended | run |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for task_id in sorted(by_task):
        for e, t, _ in sorted(by_task[task_id], key=lambda x: -(norm(x[1]) or -1e9)):
            r, cost = e["row"], t.get("cost") or {}
            lines.append(
                f"| {task_id} | {r['model']} | {r.get('effort') or '-'} | {r['mode']} | "
                f"{_sandbox(r)} | {_f(norm(t), '.3f')} | {_f(t.get('test'))} | "
                f"{_f(t.get('best_val'))} | {t.get('rounds')} ({t.get('accepted')}) | "
                f"{_f(cost.get('usd'), '.2f')} | {_f((cost.get('wall_s') or 0) / 3600, '.2f')} | "
                f"{t.get('ended_reason') or '-'} | `{_short(e['ref'])}` |")
    lines.append("")

    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda x: -(norm(x[1]) or -1e9))
        spec = next((tt for tt in rows[0][0]["run"]["tasks"] if tt["id"] == task_id), {})
        metric, anchors = spec.get("metric", "score"), spec.get("anchors") or {}
        secondary = sorted({k for _, _, f in rows
                            for k, v in ((f.get("test") or {}).get("metrics") or {}).items()
                            if isinstance(v, float) and k != metric})
        lines += [f"## {task_id}", "",
                  f"Primary metric `{metric}` ({spec.get('direction', 'maximize')}); starter "
                  f"{_f(anchors.get('baseline'))}, reference {_f(anchors.get('reference'))} "
                  "(the normalization anchors, on val).", "",
                  "### Scores (test unless noted)", "",
                  "| # | model | effort | mode | sandbox | norm test | " + metric + " | "
                  + "".join(f"{k} | " for k in secondary) + "best val | val AUC | test AUC |",
                  "|---:|---|---|---|---|---:|---:|" + "---:|" * len(secondary) + "---:|---:|---:|"]
        for i, (e, t, f) in enumerate(rows, 1):
            r, m = e["row"], ((f.get("test") or {}).get("metrics") or {})
            lines.append(
                f"| {i} | {r['model']} | {r.get('effort') or '-'} | {r['mode']} | {_sandbox(r)} | "
                f"{_f(norm(t), '.3f')} | {_f(t.get('test'))} | "
                + "".join(f"{_f(m.get(k))} | " for k in secondary)
                + f"{_f(t.get('best_val'))} | {_f(t.get('val_auc'), '.3f')} | "
                  f"{_f(t.get('test_auc'), '.3f')} |")
        lines += ["", "### Process and cost", "",
                  "| model | mode | rounds | acc / rej / no-change | 1st-round gain | later gain | "
                  "false acc. | E / I commits | violations | USD | wall h | GPU h | tokens (M) | "
                  "ended | base | protocol |",
                  "|---|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|---|"]
        for e, t, _ in rows:
            r, cost = e["row"], t.get("cost") or {}
            fa, hc = t.get("false_acceptance") or {}, t.get("harness_commits") or {}
            tokens = ((cost.get("input_tokens") or 0) + (cost.get("output_tokens") or 0)) / 1e6
            lines.append(
                f"| {r['model']} | {r['mode']} | {t.get('rounds')} | {t.get('accepted')} / "
                f"{t.get('rejected')} / {t.get('no_change', 0)} | "
                f"{_f(t.get('first_round_gain'), '.3f')} | {_f(t.get('later_gain'), '.3f')} | "
                f"{fa.get('count', '-')} / {fa.get('judged', '-')} | {hc.get('E', 0)} / {hc.get('I', 0)} | "
                f"{t.get('violations', 0)} | {_f(cost.get('usd'), '.2f')} | "
                f"{_f((cost.get('wall_s') or 0) / 3600, '.2f')} | {_f(cost.get('gpu_hours'), '.2f')} | "
                f"{_f(tokens, '.1f')} | {t.get('ended_reason') or '-'} | {r.get('base')} | "
                f"{r.get('protocol_id')} |")
        lines += ["", "### Test score of each accepted version (post hoc; val in parentheses)", ""]
        for e, _, _ in rows:
            points = [p for (tid, _), p in latest_posthoc(e["posthoc"]).items()
                      if tid == task_id]
            curve = " → ".join(f"r{p['round']} {_f(p.get('score'))} ({_f(p.get('val_score'))})"
                               for p in sorted(points, key=lambda p: p["round"]))
            lines.append(f"- {e['row']['model']} ({e['row']['mode']}, `{_short(e['ref'])}`): "
                         f"{curve or 'not scored yet (`rsib posthoc`)'}")
        lines.append("")

    table = build_table([e["row"] for e in shown], include_unsandboxed=True)
    lines += ["## Across runs (mean ± s.e. per condition)", "", table_markdown(table)]
    waived = [e for e in shown if e["row"].get("sandbox_waiver")]
    if waived:
        lines += ["## Sandbox waivers", ""]
        for e in waived:
            w = e["row"]["sandbox_waiver"]
            lines.append(f"- `{e['row']['run_id']}` ({w.get('at', '')[:10]}): {w.get('reason')}")
        lines.append("")
    lines += ["## Columns", "",
              "- **norm test**: (test − starter) / (reference − starter) on the task's primary metric.",
              "- **1st-round gain** / **later gain**: normalized val gain of the first round over the "
              "starter / of the later rounds on top of it.",
              "- **val AUC** / **test AUC**: area under the normalized best-so-far curve against the "
              "share of the budget used (test AUC from the post-hoc scores).",
              "- **false acc.**: accepted versions whose test score fell / pairs judged.",
              "- **sandbox**: yes, no, or waiver (made without the sandbox, accepted as standard).", ""]
    return "\n".join(lines)


def commit_to_branch(repo: Path, branch: str, path: str, text: str, message: str) -> str | None:
    """Write one file on `branch` with git plumbing (no checkout); None when unchanged."""
    full = f"refs/heads/{branch}"
    parent = gitops.rev_parse(repo, full) if gitops.ref_exists(repo, full) else None
    if parent and gitops.show(repo, full, path) == text:
        return None
    blob = gitops.git(["hash-object", "-w", "--stdin"], repo, input_text=text).strip()
    entries = []
    if parent:
        entries = [line for line in gitops.git(["ls-tree", full], repo).splitlines()
                   if line.split("\t", 1)[1] != path]
    entries.append(f"100644 blob {blob}\t{path}")
    tree = gitops.git(["mktree"], repo, input_text="\n".join(entries) + "\n").strip()
    args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
    commit = gitops.git(args, repo, env=gitops.bot_env()).strip()
    gitops.git(["update-ref", full, commit, parent or "0" * 40], repo)
    return commit
