"""`rsib` command line.

    rsib task list | check <id> | prepare <id> | calibrate <id>
    rsib run new --task <id> [--task <id> ...] --model <model> [--auth cli|api] [--mode full]
    rsib run start <worktree> [--backend local|slurm] [--push origin]
    rsib run status <worktree>
    rsib run finalize <worktree> [--backend slurm]   (retry failed test evaluations)
    rsib posthoc <worktree> [--every K] [--redo N[,N...] [--redo-reason TEXT]]
    rsib report <worktree> [--format md|json] [--metrics PLUGIN,...]
    rsib validate --ref <branch> [--repo PATH]
    rsib results append --ref <branch> [--results-branch results] [--metrics PLUGIN,...]
    rsib results table [--branches 'run/*'] [--column NAME ...] [--format md|csv|json]
    rsib push <worktree> [--remote origin]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__, gitops
from .compute import LocalBackend, load_site
from .paths import data_root, repo_root, runs_root, site_config_path
from .protocol import parse_budget_overrides


def _site(args) -> dict:
    path = Path(args.site).expanduser() if getattr(args, "site", None) else site_config_path()
    return load_site(path)


def _worktree(arg: str) -> Path:
    path = Path(arg)
    if not (path / "records" / "run.json").exists():
        candidate = runs_root() / arg
        if (candidate / "records" / "run.json").exists():
            return candidate
        raise SystemExit(f"{arg}: not a run worktree (records/run.json missing)")
    return path.resolve()


# ---- task commands ---------------------------------------------------------------------------

def cmd_task(args) -> int:
    from .tasks import list_tasks, load_task
    if args.task_cmd == "list":
        for t in list_tasks():
            print(f"{t.id:32s} tier={t.tier:3s} domain={t.domain:16s} metric={t.primary_metric} "
                  f"({t.direction})")
        return 0
    task = load_task(args.task_id)
    if args.task_cmd == "check":
        print(f"{task.id}: ok (tier {task.tier}, metric {task.primary_metric} {task.direction}, "
              f"anchors {task.baseline} -> {task.reference}, sigma {task.noise_sigma})")
        for split in ("dev", "val", "test"):
            d = task.split_dir(split)
            print(f"  {split:4s} {d} {'present' if d.exists() else 'MISSING (run rsib task prepare)'}")
        return 0
    if args.task_cmd == "prepare":
        script = task.entry("prepare", "prepare.py")
        if script is None or not script.exists():
            print(f"{task.id}: no prepare script; place data under {data_root() / task.id} yourself")
            return 1
        out = data_root() / task.id
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--out", str(out)]
        print("+", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(task.root))
    if args.task_cmd == "calibrate":
        from .compute import make_backend
        from .orchestrator.evaluator import run_eval
        site = _site(args)
        backend = make_backend(args.backend, site) if task.eval_gpus else LocalBackend({})
        container = None
        if backend.name == "slurm" and task.container:
            container = site.get("slurm", {}).get("containers", {}).get(task.container)
            if not container:
                print(f"map container '{task.container}' under [slurm.containers] in the site file")
                return 1
        workspace = Path(args.workspace).resolve() if args.workspace else task.workspace_src
        scores = []
        scratch = runs_root() / "_calibrate" / task.id  # on the shared filesystem (SLURM jobs)
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rsib-calib-", dir=scratch) as tmp:
            for i in range(args.repeats):
                res = run_eval(task, workspace, args.split, Path(tmp) / f"r{i}", backend,
                               container=container)
                if res["status"] != "ok":
                    print(res.get("detail", ""), res["log_tail"][-1500:])
                print(f"  repeat {i}: {res['status']} {task.primary_metric}={res['score']}")
                if res["score"] is not None:
                    scores.append(res["score"])
        if not scores:
            print("no successful evaluation")
            return 1
        sigma = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        print(f"\nsuggested task.toml values for {task.id}:\n[rsi.anchors]\n"
              f"baseline = {statistics.mean(scores):.6g}\n[rsi.noise]\nsigma = {sigma:.6g}\n"
              f"repeats = {len(scores)}\n(reference = score of a strong known solution; set it by hand)")
        return 0
    return 2


# ---- run commands ----------------------------------------------------------------------------

def _protocol_overrides(args) -> dict:
    out = {}
    if args.max_rounds is not None:
        out["max_rounds"] = args.max_rounds
    if args.max_calls_per_phase is not None:
        out["max_model_calls_per_phase"] = args.max_calls_per_phase
    if args.max_idle_rounds is not None:
        out["max_idle_rounds"] = args.max_idle_rounds
    if args.exact_context:
        out["exact_context"] = True
    if args.submit_min_gain is not None:
        out["submit_min_gain"] = args.submit_min_gain
    return out


def _purge_expired() -> None:
    """Delete archives of finished runs whose retention period is over."""
    from .scratch import purge
    for info in purge():
        print(f"deleted expired archive {info['archive']} ({info['run_id']})", file=sys.stderr)


def cmd_run(args) -> int:
    if args.run_cmd == "new":
        from .runs import new_run
        from .sandbox import probe, run_config
        params = json.loads(args.model_params) if args.model_params else {}
        sandbox = run_config(_site(args), args.sandbox)
        if sandbox["enabled"]:
            err = probe()
            if err:
                print(f"error: the filesystem sandbox does not work here: {err}\n"
                      "Runs are sandboxed by default; pass --no-sandbox to create an unsandboxed "
                      "run (it is left out of results tables by default).", file=sys.stderr)
                return 2
        _purge_expired()
        branch, wt = new_run(
            args.task, args.model, mode=args.mode, seed=args.seed, effort=args.effort,
            model_params=params, auth=args.auth, base_ref=args.base, improver_from=args.improver_from,
            executor_from=args.executor_from,
            budget_overrides=parse_budget_overrides(args.budget), tier=args.tier,
            protocol_overrides=_protocol_overrides(args), sandbox=sandbox,
            namespace=args.namespace,
            notes=args.notes or "", log=lambda msg: print(msg, file=sys.stderr))
        print(json.dumps({"branch": branch, "worktree": str(wt)}))
        return 0
    if args.run_cmd == "start":
        from .orchestrator.engine import RunEngine
        _purge_expired()
        engine = RunEngine(_worktree(args.worktree), backend=args.backend, site=_site(args),
                           push_remote=args.push)
        result = engine.start(close_partial_round=args.close_partial_round)
        if result.get("paused"):
            print(f"paused between rounds: {result['paused']}; run `rsib run start` again to continue")
            return 0
        for f in result["final"]:
            print(f"{f['task_id']}: best val {f['best_val']} test {f['test']['score']} "
                  f"(normalized {f['normalized']['test']}) ended: {f['ended_reason']}")
        return 0
    if args.run_cmd == "finalize":
        from .orchestrator.engine import RunEngine
        engine = RunEngine(_worktree(args.worktree), backend=args.backend, site=_site(args))
        engine.records.track_all()
        engine.finalize(retry_failed=True)
        for f in engine.records.read("final"):
            print(f"{f['task_id']}: test {f['test']['status']} {f['test']['score']}")
        return 0
    if args.run_cmd == "archive":
        from . import gitops
        from .orchestrator.engine import RunEngine
        engine = RunEngine(_worktree(args.worktree), site=_site(args))
        done = gitops.ref_exists(engine.wt, f"refs/tags/{engine.branch}/final")
        if not done and not args.force:
            print("error: the run has no final tag yet; archive a finished run, or pass --force",
                  file=sys.stderr)
            return 2
        if engine.records.read("environment") and not args.force:
            print("already archived (records/environment.jsonl); --force archives again",
                  file=sys.stderr)
            return 0
        engine.records.track_all()
        row = engine.archive_scratch()
        print(json.dumps({k: row[k] for k in ("archive", "archived_at", "delete_after")}))
        return 0
    if args.run_cmd == "status":
        from .records import Records
        rec = Records(_worktree(args.worktree))
        run = rec.run()
        rounds = rec.read("rounds")
        print(f"{run['branch']}  mode={run['mode']} model={run['model']['name']}")
        for t in run["tasks"]:
            rows = [r for r in rounds if r["task_id"] == t["id"]]
            last = rows[-1] if rows else None
            print(f"  {t['id']}: rounds={len(rows) - 1 if rows else 0} "
                  f"best_val={last['best_val_after'] if last else None} "
                  f"budget_used={last['budget_used_frac'] if last else 0:.2f}")
        for f in rec.read("final"):
            print(f"  final {f['task_id']}: test={f['test']['score']} ({f['ended_reason']})")
        return 0
    return 2


def cmd_posthoc(args) -> int:
    from .posthoc import RedoError, posthoc
    try:
        posthoc(_worktree(args.worktree), every=args.every, backend=args.backend, site=_site(args),
                redo=args.redo, redo_reason=args.redo_reason)
    except RedoError as exc:
        raise SystemExit(f"rsib: {exc}") from None
    return 0


def _redo_rounds(text: str) -> list[int]:
    try:
        rounds = [int(part) for part in text.split(",")]
        if any(r < 0 for r in rounds):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected nonnegative rounds N[,N...]") from None
    return list(dict.fromkeys(rounds))


def _load_metrics(args) -> None:
    from .metrics import load_plugins, plugin_names
    load_plugins(plugin_names(args.metrics))


def cmd_report(args) -> int:
    from .report import report
    _load_metrics(args)
    text = report(_worktree(args.worktree), fmt=args.format)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_ref
    repo = Path(args.repo).resolve() if args.repo else repo_root()
    errors = validate_ref(repo, args.ref)
    if errors:
        print(f"{args.ref}: {len(errors)} problem(s)")
        for e in errors:
            print("  -", e)
        return 1
    print(f"{args.ref}: ok")
    return 0


def cmd_results(args) -> int:
    from .results import (append, build_table, load_index, rows_from_branches, table_csv,
                          table_markdown)
    repo = Path(args.repo).resolve() if args.repo else repo_root()
    if args.results_cmd == "report":
        from .results_report import DEFAULT_PATTERNS, collect, commit_to_branch, render
        text = render(collect(repo, args.branches or DEFAULT_PATTERNS, args.results_branch),
                      include_unsandboxed=args.include_unsandboxed)
        if args.output:
            Path(args.output).write_text(text)
        if args.commit_to:
            sha = commit_to_branch(repo, args.commit_to, "RESULTS.md", text,
                                   "RESULTS.md: regenerated by rsib results report")
            print(f"RESULTS.md {'committed as ' + sha[:10] if sha else 'unchanged'} on "
                  f"{args.commit_to}", file=sys.stderr)
            if sha and args.push:
                gitops.git(["push", "-q", args.push, f"refs/heads/{args.commit_to}"], repo)
        if not args.output and not args.commit_to:
            print(text)
        return 0
    if args.results_cmd == "table":
        if args.branches:  # index rows keep the metrics computed when they were appended
            _load_metrics(args)
        rows = rows_from_branches(repo, args.branches) if args.branches else \
            load_index(repo, args.results_branch)
        table = build_table(rows, {"task": args.task, "model": args.model, "base": args.base},
                            columns=args.column, include_unsandboxed=args.include_unsandboxed)
        text = {"md": table_markdown, "csv": table_csv,
                "json": lambda t: json.dumps(t, indent=2)}[args.format](table)
        if args.output:
            Path(args.output).write_text(text)
        else:
            print(text)
        return 0
    _load_metrics(args)
    row = append(repo, args.ref, args.results_branch, args.accept_unsandboxed)
    print(json.dumps({"run_id": row["run_id"], "commit": row["commit"]}))
    if args.push:
        gitops.git(["push", "-q", args.push, f"refs/heads/{args.results_branch}"], repo)
    return 0


def cmd_push(args) -> int:
    wt = _worktree(args.worktree)
    from .records import Records
    branch = Records(wt).run()["branch"]
    gitops.push(wt, args.remote, branch)
    print(f"pushed {branch} (+ tags) to {args.remote}")
    return 0


METRICS_HELP = ("metric plugins to load, comma separated: module paths or .py files "
                "(default $RSIB_METRICS)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsib", description="agent-rsi-benchmarking command line")
    p.add_argument("--version", action="version", version=f"rsib {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("task", help="inspect and prepare task packages")
    tsub = t.add_subparsers(dest="task_cmd", required=True)
    tsub.add_parser("list")
    for name in ("check", "prepare"):
        x = tsub.add_parser(name)
        x.add_argument("task_id")
    c = tsub.add_parser("calibrate", help="score the starter workspace N times (anchors, noise)")
    c.add_argument("task_id")
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--split", default="val")
    c.add_argument("--workspace", default=None, help="score this solution instead of the starter")
    c.add_argument("--backend", default="local", choices=["local", "slurm"],
                   help="backend for GPU evaluators (CPU evaluators always run locally)")
    c.add_argument("--site", default=None)
    t.set_defaults(func=cmd_task)

    r = sub.add_parser("run", help="create, start and inspect runs")
    rsub = r.add_subparsers(dest="run_cmd", required=True)
    n = rsub.add_parser("new", help="create a run branch + worktree")
    n.add_argument("--task", action="append", required=True,
                   help="task id; repeat for a stream (tasks run in this order)")
    n.add_argument("--model", required=True,
                   help="model name (claude-opus-5-5, gpt-5.6-sol, gemini-..., doubao-..., "
                        "<name>@<url> for an OpenAI-compatible server) or a full "
                        "<adapter>:<model> spec (claude-cli, codex-cli, anthropic, openai, mock)")
    n.add_argument("--auth", default=None, choices=["cli", "api"],
                   help="cli: a local CLI login (default; claude-* via Claude Code, gpt-* via "
                        "Codex); api: an API key from the environment or the .env file")
    n.add_argument("--effort", default=None, help="reasoning effort passed to the model")
    n.add_argument("--model-params", default=None, help="JSON dict of adapter parameters")
    n.add_argument("--mode", default="full", choices=["base", "fixed-i", "full"])
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--base", default=None,
                   help="release tag to branch from (default: newest harness-v* tag in the "
                        "checked-out branch)")
    n.add_argument("--improver-from", default=None, help="ref whose harness/improver to import (probe runs)")
    n.add_argument("--executor-from", default=None,
                   help="ref whose E (harness/ except harness/improver) to import, e.g. a run's "
                        "final E in mode base to measure what the harness edits gained")
    n.add_argument("--tier", default=None, help="override every task's budget tier")
    n.add_argument("--budget", action="append", help="override a budget limit, e.g. usd=20")
    n.add_argument("--max-rounds", type=int, default=None, help="override protocol max_rounds")
    n.add_argument("--max-idle-rounds", type=int, default=None,
                   help="override protocol max_idle_rounds (0: idle rounds never end the run)")
    n.add_argument("--max-calls-per-phase", type=int, default=None,
                   help="override protocol max_model_calls_per_phase")
    n.add_argument("--submit-min-gain", type=float, default=None,
                   help="override protocol submit_min_gain: the dev gain, as a share of the task's "
                        "anchor span, that ends a round and submits its state")
    n.add_argument("--exact-context", action="store_true",
                   help="set protocol exact_context: refuse model calls whose context the "
                        "adapter cannot deliver as given")
    n.add_argument("--namespace", default=None,
                   help="prefix of the run branch, e.g. robo -> robo/run/... "
                        "(default $RSIB_RUN_NAMESPACE, else none: run/...)")
    n.add_argument("--notes", default=None)
    n.add_argument("--sandbox", dest="sandbox", action="store_true", default=None,
                   help="restrict what agent code can read (default: site [sandbox] enabled)")
    n.add_argument("--no-sandbox", dest="sandbox", action="store_false")
    n.add_argument("--site", default=None, help="site profile (default $RSIB_SITE)")
    s = rsub.add_parser("start", help="run (or resume) a run to completion")
    s.add_argument("worktree")
    s.add_argument("--backend", default="local", choices=["local", "slurm"])
    s.add_argument("--site", default=None, help="site TOML (default $RSIB_SITE or ~/.config/rsib/site.toml)")
    s.add_argument("--push", default=None, help="git remote to push the branch to after every round")
    s.add_argument("--close-partial-round", action="store_true",
                   help="finish an interrupted round with the work done so far instead of discarding it")
    fz = rsub.add_parser("finalize", help="(re)score final commits on test where that failed")
    fz.add_argument("worktree")
    fz.add_argument("--backend", default="local", choices=["local", "slurm"])
    fz.add_argument("--site", default=None)
    st = rsub.add_parser("status")
    st.add_argument("worktree")
    ar = rsub.add_parser("archive", help="inventory a finished run's scratch and move it, with the "
                                         "large artifacts, to the archive (normally automatic)")
    ar.add_argument("worktree")
    ar.add_argument("--site", default=None)
    ar.add_argument("--force", action="store_true", help="archive even if the run is not finished")
    r.set_defaults(func=cmd_run)

    av = sub.add_parser("archive", help="list or delete archived scratch of finished runs")
    asub = av.add_subparsers(dest="archive_cmd", required=True)
    asub.add_parser("list")
    pg = asub.add_parser("purge", help="delete archives past their retention (or all matching)")
    pg.add_argument("--run", default=None, help="only this run id or archive name")
    pg.add_argument("--all", action="store_true", help="delete matching archives now, expired or not")
    av.set_defaults(func=cmd_archive)

    ph = sub.add_parser("posthoc", help="score accepted versions on the test split")
    ph.add_argument("worktree")
    ph.add_argument("--every", type=int, default=1)
    ph.add_argument("--redo", type=_redo_rounds, default=None, metavar="N[,N...]",
                    help="re-score only these accepted rounds, appending superseding rows")
    ph.add_argument("--redo-reason", default=None, metavar="TEXT",
                    help="reason stored on redo rows (requires --redo)")
    ph.add_argument("--backend", default="local", choices=["local", "slurm"])
    ph.add_argument("--site", default=None)
    ph.set_defaults(func=cmd_posthoc)

    rp = sub.add_parser("report", help="summarize a run from its records")
    rp.add_argument("worktree")
    rp.add_argument("--format", default="md", choices=["md", "json"])
    rp.add_argument("--output", default=None)
    rp.add_argument("--metrics", default=None, metavar="PLUGINS", help=METRICS_HELP)
    rp.set_defaults(func=cmd_report)

    v = sub.add_parser("validate", help="integrity checks for a run branch")
    v.add_argument("--ref", required=True)
    v.add_argument("--repo", default=None)
    v.set_defaults(func=cmd_validate)

    res = sub.add_parser("results", help="append a run summary to the results branch")
    rsub2 = res.add_subparsers(dest="results_cmd", required=True)
    a = rsub2.add_parser("append")
    a.add_argument("--ref", required=True)
    a.add_argument("--repo", default=None)
    a.add_argument("--results-branch", default="results")
    a.add_argument("--push", default=None, help="remote to push the results branch to")
    a.add_argument("--accept-unsandboxed", default=None, metavar="REASON",
                   help="count a run made without the sandbox as a standard run; the reason is "
                        "kept in the row (sandbox_waiver)")
    a.add_argument("--metrics", default=None, metavar="PLUGINS", help=METRICS_HELP)
    rp2 = rsub2.add_parser("report", help="markdown report of every run, read from the run branches")
    rp2.add_argument("--repo", default=None)
    rp2.add_argument("--branches", action="append", default=None, metavar="PATTERN",
                     help="run branch pattern, repeatable (default: run/* and */run/*)")
    rp2.add_argument("--results-branch", default="results",
                     help="index whose sandbox waivers apply (default: results)")
    rp2.add_argument("--include-unsandboxed", action="store_true")
    rp2.add_argument("--output", default=None, help="write the report to this file")
    rp2.add_argument("--commit-to", default=None, metavar="BRANCH",
                     help="commit the report as RESULTS.md on this branch (e.g. results)")
    rp2.add_argument("--push", default=None, help="remote to push that branch to")
    tb = rsub2.add_parser("table", help="group runs by task/model/condition: mean ± s.e. and gains")
    tb.add_argument("--repo", default=None)
    tb.add_argument("--results-branch", default="results", help="index to read (default: results)")
    tb.add_argument("--branches", default=None,
                    help="read run branches matching this pattern (e.g. 'run/*') instead of the index")
    tb.add_argument("--task", default=None)
    tb.add_argument("--model", default=None)
    tb.add_argument("--base", default=None)
    tb.add_argument("--column", action="append", default=None, metavar="NAME",
                    help="column to show, repeatable: a key of the task summaries, nested keys "
                         "joined with dots (cost.usd); default: the standard columns")
    tb.add_argument("--metrics", default=None, metavar="PLUGINS",
                    help=METRICS_HELP + "; used with --branches")
    tb.add_argument("--include-unsandboxed", action="store_true",
                    help="also count runs made without the filesystem sandbox")
    tb.add_argument("--format", default="md", choices=["md", "csv", "json"])
    tb.add_argument("--output", default=None)
    res.set_defaults(func=cmd_results)

    pu = sub.add_parser("push", help="push a run branch and its tags")
    pu.add_argument("worktree")
    pu.add_argument("--remote", default="origin")
    pu.set_defaults(func=cmd_push)
    return p


def cmd_archive(args) -> int:
    from .scratch import list_archives, purge
    if args.archive_cmd == "list":
        for info in list_archives():
            print(f"{info['archive']}  run={info['run_id']}  archived={info['archived_at'][:19]}  "
                  f"delete_after={info['delete_after'][:19]}")
        return 0
    gone = purge(run=args.run, force=args.all)
    for info in gone:
        print(f"deleted {info['archive']} ({info['run_id']})")
    if not gone:
        print("nothing to delete")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .metrics import MetricError
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except MetricError as exc:
        raise SystemExit(f"rsib: {exc}") from None


if __name__ == "__main__":
    sys.exit(main())
