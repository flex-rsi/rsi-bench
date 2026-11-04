"""Metrics: the built-in summary of a known run, the progress metric, plugins (files, modules,
--metrics, RSIB_METRICS), key collisions, and configurable results-table columns."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rsibench import metrics
from rsibench.cli import main
from rsibench.metrics import MetricError, load_plugins
from rsibench.records import latest_posthoc, validate
from rsibench.report import summarize
from rsibench.results import METRICS, build_table, flatten, load_index, table_csv, table_markdown
from rsibench.results_report import collect, render

BUILTINS = ["outcome", "gains", "auc", "acceptance", "harness", "cost", "progress"]
BUILTIN_KEYS = {
    "task_id", "tier", "direction", "anchors", "rounds", "accepted", "rejected", "no_change",
    "errors", "baseline_val", "best_val", "test", "normalized", "ended_reason", "first_round_gain",
    "later_gain", "val_auc", "test_auc", "posthoc_points", "false_acceptance", "harness_commits",
    "harness_lines", "violations", "cost", "eval_model", "late_gain"}

PLUGIN = '''
from rsibench.metrics import metric


@metric("best_round")
def best_round(run, task):
    accepted = [r["round"] for r in run.rows("rounds", task["id"]) if r["decision"] == "accept"]
    return {"best_round": accepted[-1] if accepted else 0,
            "seen": {"rounds": len(run.files["rounds"])}}
'''


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Plugins a test loads stay out of the other tests; the user's RSIB_METRICS is ignored."""
    monkeypatch.delenv("RSIB_METRICS", raising=False)
    saved, loaded = dict(metrics.REGISTRY), set(metrics._LOADED)
    yield
    metrics.REGISTRY.clear()
    metrics.REGISTRY.update(saved)
    metrics._LOADED.clear()
    metrics._LOADED.update(loaded)
    for name in [m for m in sys.modules if m.startswith(("rsib_metrics_", "rsib_test_plugins"))]:
        del sys.modules[name]


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _round(n, decision, val, best, frac, commit=None, e=0, i=0):
    changes = {s: {"commit": f"{s}{n}" if lines else None, "lines": lines}
               for s, lines in (("E", e), ("I", i))}
    return {"task_id": "t", "round": n, "decision": decision, "best_val_after": best,
            "val": {"status": "skipped" if val is None else "ok", "score": val, "metrics": {}},
            "accepted_commit": commit, "harness_changes": changes, "budget_used_frac": frac,
            "cumulative": {"wall_s": 7200 * frac, "gpu_hours": 0.0, "usd": 2.0 * frac,
                           "input_tokens": 1000 * n, "output_tokens": 100 * n}}


def known_run(root: Path, fracs=(0.0, 0.3, 0.5, 0.8, 1.0), anchors=True) -> Path:
    """records/ of a finished run (minimize, anchors 1 -> 0): val 1.0, accept 0.5, reject 0.7,
    accept 0.4, no change; test 0.45; post-hoc test 1.0, 0.6, 0.65 (one false acceptance)."""
    rec = Path(root) / "records"
    rec.mkdir(parents=True)
    run = {"run_id": "run/t/mock-scripted/full/s1", "mode": "full",
           "model": {"name": "mock-scripted", "effort": "high"}, "seed": 1,
           "base": {"ref": "harness-v0.0.1", "commit": "0" * 40}, "improver_source": None,
           "protocol": {"max_rounds": 4}, "budgets": {"t": {"usd": 2.0}},
           "tasks": [{"id": "t", "tier": "XS", "version_hash": "h", "direction": "minimize",
                      "anchors": {"baseline": 1.0, "reference": 0.0} if anchors else None}]}
    (rec / "run.json").write_text(json.dumps(run))
    f0, f1, f2, f3, f4 = fracs
    _jsonl(rec / "rounds.jsonl", [
        _round(0, "baseline", 1.0, 1.0, f0, "c0"), _round(1, "accept", 0.5, 0.5, f1, "c1", e=10),
        _round(2, "reject", 0.7, 0.5, f2, i=4), _round(3, "accept", 0.4, 0.4, f3, "c3"),
        _round(4, "no-change", None, 0.4, f4)])
    _jsonl(rec / "final.jsonl", [{"task_id": "t", "ended_reason": "budget_usd", "best_val": 0.4,
                                  "test": {"status": "ok", "score": 0.45, "metrics": {}},
                                  "normalized": {"val": 0.6, "test": 0.55}}])
    _jsonl(rec / "posthoc.jsonl", [{"task_id": "t", "round": r, "score": s}
                                   for r, s in ((0, 1.0), (1, 0.6), (3, 0.65))])
    _jsonl(rec / "violations.jsonl", [{"task_id": "t"}])
    _jsonl(rec / "usage.jsonl", [{"task_id": "t"}] * 3)
    return rec


def plugin_file(tmp_path: Path, source: str = PLUGIN, name: str = "lab_metrics.py") -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def run_repo(tmp_path: Path) -> Path:
    """A git repository whose branch run/t holds the known run's records."""
    repo = tmp_path / "repo"
    repo.mkdir()
    known_run(repo)
    for args in (["init", "-q", "-b", "run/t"], ["add", "-A"], ["commit", "-qm", "records"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                       check=True, capture_output=True)
    return repo


def test_builtin_metrics_give_the_standard_summary(tmp_path):
    summary = summarize(known_run(tmp_path))
    assert summary["metrics"] == BUILTINS
    assert summary["protocol_id"] == "d7d7a6a825fd" and summary["model_calls"] == 3
    task, = summary["tasks"]
    assert set(task) == BUILTIN_KEYS
    assert {k: task[k] for k in ("rounds", "accepted", "rejected", "no_change", "errors",
                                 "baseline_val", "best_val", "test", "ended_reason",
                                 "posthoc_points", "violations")} == {
        "rounds": 4, "accepted": 2, "rejected": 1, "no_change": 1, "errors": 0,
        "baseline_val": 1.0, "best_val": 0.4, "test": 0.45, "ended_reason": "budget_usd",
        "posthoc_points": 3, "violations": 1}
    assert task["first_round_gain"] == pytest.approx(0.5)
    assert task["later_gain"] == pytest.approx(0.1)
    assert task["val_auc"] == pytest.approx(0.37) and task["test_auc"] == pytest.approx(0.27)
    assert task["false_acceptance"] == {"count": 1, "judged": 2}
    assert task["harness_commits"] == {"E": 1, "I": 1}
    assert task["harness_lines"] == {"E": 10, "I": 4}
    assert task["normalized"] == {"val": 0.6, "test": 0.55} and task["cost"]["usd"] == 2.0
    assert task["eval_model"] == {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    row = {**summary, "schema_version": "1.0", "ref": "run/t", "commit": "a" * 40,
           "indexed_at": "2026-10-02T00:00:00Z"}
    assert validate("result_row", row) == []


def test_latest_posthoc_uses_append_order_and_keeps_tasks_separate():
    original = {"task_id": "t", "round": 1, "score": 0.6,
                "evaluated_at": "2026-10-11T10:00:00Z"}
    other = {"task_id": "other", "round": 1, "score": 0.9}
    retry = {"task_id": "t", "round": 1, "score": None, "status": "error",
             "evaluated_at": "2026-10-11T09:00:00Z"}
    rows = [original, other, retry]
    assert latest_posthoc(rows) == {("t", 1): retry, ("other", 1): other}
    assert rows == [original, other, retry]  # the audit history remains intact


def test_posthoc_redo_updates_summary_metrics_without_counting_duplicates(tmp_path):
    rec = known_run(tmp_path)
    with (rec / "posthoc.jsonl").open("a") as f:
        f.write(json.dumps({"task_id": "t", "round": 3, "score": 0.2, "redo": True}) + "\n")
        f.write(json.dumps({"task_id": "other", "round": 3, "score": 0.99}) + "\n")
    task, = summarize(rec)["tasks"]
    assert task["posthoc_points"] == 3
    assert task["test_auc"] == pytest.approx(0.36)
    assert task["false_acceptance"] == {"count": 0, "judged": 2}


def test_results_report_uses_latest_posthoc_in_the_summary_and_curve(tmp_path):
    repo = run_repo(tmp_path)
    with (repo / "records" / "posthoc.jsonl").open("a") as f:
        f.write(json.dumps({"task_id": "t", "round": 3, "score": 0.2, "redo": True}) + "\n")
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qam", "redo records"], cwd=repo, check=True, capture_output=True)
    entries = collect(repo)
    task, = entries[0]["row"]["tasks"]
    assert task["posthoc_points"] == 3 and task["test_auc"] == pytest.approx(0.36)
    text = render(entries, include_unsandboxed=True)
    assert "r0 1.0000 (-) → r1 0.6000 (-) → r3 0.2000 (-)" in text
    assert "r3 0.6500" not in text and text.count("r3 ") == 1


def test_eval_model_adds_up_the_model_calls_of_val_and_test_evaluations(tmp_path):
    rec = known_run(tmp_path)
    _jsonl(rec / "usage.jsonl", [
        {"task_id": "t", "round": 1, "phase": phase, "input_tokens": 10, "output_tokens": 2,
         "cache_read_tokens": 4, "cache_write_tokens": 1, "usd": 0.25}
        for phase in ("executor", "experiment", "eval-dev", "eval-val", "eval-test", "posthoc")])
    task = summarize(rec)["tasks"][0]
    # not posthoc calls: those depend on whether and how often `rsib posthoc` ran
    assert task["eval_model"] == {"calls": 2, "input_tokens": 30, "output_tokens": 4, "usd": 0.5}
    assert task["cost"]["usd"] == 2.0  # the budget's cost never holds them


@pytest.mark.parametrize("fracs, late", [
    ((0.0, 0.3, 0.5, 0.8, 1.0), 0.1),   # best 0.5 at half the budget, 0.4 at the end
    ((0.0, 0.1, 0.2, 0.5, 1.0), 0.0),   # no gain after half the budget: saturated
    ((0.0, 0.6, 0.7, 0.8, 1.0), 0.6),   # only the baseline ended within the first half
    ((0.0, 0.1, 0.2, 0.3, 0.4), None),  # the second half never came
])
def test_progress_is_the_gain_after_half_of_the_budget(tmp_path, fracs, late):
    got = summarize(known_run(tmp_path, fracs))["tasks"][0]["late_gain"]
    assert got is None if late is None else got == pytest.approx(late)


def test_progress_without_anchors_is_not_computable(tmp_path):
    task = summarize(known_run(tmp_path, anchors=False))["tasks"][0]
    assert task["late_gain"] is None and task["first_round_gain"] is None


def test_summarize_runs_the_metrics_named(tmp_path):
    rec = known_run(tmp_path)
    summary = summarize(rec, metrics=["progress", "auc"])
    assert summary["metrics"] == ["progress", "auc"]
    assert set(summary["tasks"][0]) == {"task_id", "tier", "direction", "anchors", "late_gain",
                                        "val_auc", "test_auc", "posthoc_points"}
    with pytest.raises(MetricError, match="unknown metric.*nope"):
        summarize(rec, metrics=["nope"])


@pytest.mark.parametrize("how", ["flag", "environment", "dotenv"])
def test_a_plugin_file_adds_keys_to_the_report(tmp_path, monkeypatch, how):
    root, plugin = tmp_path / "run", plugin_file(tmp_path)
    known_run(root)
    args = ["report", str(root), "--format", "json", "--output", str(tmp_path / "s.json")]
    if how == "flag":
        args += ["--metrics", f"{plugin}, "]
    elif how == "environment":
        monkeypatch.setenv("RSIB_METRICS", str(plugin))
    else:
        (tmp_path / "local.env").write_text(f"RSIB_METRICS={plugin}\n")
        monkeypatch.setenv("RSIB_ENV_FILE", str(tmp_path / "local.env"))
    assert main(args) == 0
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["metrics"] == BUILTINS + ["best_round"]
    task = summary["tasks"][0]
    assert set(task) == BUILTIN_KEYS | {"best_round", "seen"}
    assert task["best_round"] == 3 and task["seen"] == {"rounds": 5}
    assert main(["report", str(root), "--output", str(tmp_path / "s.md")]) == 0
    text = (tmp_path / "s.md").read_text()
    assert "| best_round | 3 |" in text and "| seen.rounds | 5 |" in text
    assert "| late_gain | 0.1 |" in text


def test_plugins_load_from_module_paths_and_the_flag_overrides_the_environment(
        tmp_path, monkeypatch):
    package = tmp_path / "rsib_test_plugins"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "lab.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("RSIB_METRICS", "rsib_test_plugins.missing")
    root = tmp_path / "run"
    known_run(root)
    with pytest.raises(SystemExit, match="cannot load metrics plugin 'rsib_test_plugins.missing'"):
        main(["report", str(root)])
    out = tmp_path / "s.json"
    assert main(["report", str(root), "--format", "json", "--output", str(out),
                 "--metrics", "rsib_test_plugins.lab"]) == 0
    assert json.loads(out.read_text())["tasks"][0]["best_round"] == 3
    assert main(["report", str(root), "--metrics", ""]) == 0  # an empty flag loads nothing


def test_plugins_that_cannot_load_are_errors_naming_them(tmp_path):
    broken = plugin_file(tmp_path, "def broken(:\n", "broken.py")
    taken = plugin_file(tmp_path, PLUGIN.replace('"best_round")', '"outcome")'), "taken.py")
    for entry, reason in ((str(tmp_path / "missing.py"), "No such file"),
                          (str(tmp_path), "is not a Python file"),
                          (str(broken), "invalid syntax"),
                          ("rsib_test_plugins_nowhere", "No module named"),
                          (str(taken), "metric 'outcome' is already registered")):
        with pytest.raises(MetricError, match="cannot load metrics plugin") as err:
            load_plugins([entry])
        assert entry in str(err.value) and reason in str(err.value)
    assert list(metrics.REGISTRY) == BUILTINS


def test_two_metrics_returning_one_key_is_an_error_naming_both(tmp_path):
    rec = known_run(tmp_path / "run")
    load_plugins([str(plugin_file(tmp_path)), str(plugin_file(
        tmp_path, PLUGIN.replace('"best_round")', '"twin")'), "twin.py"))])
    with pytest.raises(MetricError, match="'best_round' comes from both metric 'best_round' and "
                                          "metric 'twin'"):
        summarize(rec)
    load_plugins([str(plugin_file(
        tmp_path, 'from rsibench.metrics import metric\nmetric("mine")(lambda run, task: '
                  '{"rounds": 0})\n', "mine.py"))])
    with pytest.raises(MetricError, match="'rounds' comes from both metric 'outcome' and "
                                          "metric 'mine'"):
        summarize(rec, metrics=BUILTINS + ["mine"])
    with pytest.raises(SystemExit, match="rsib: key 'best_round' comes from both"):
        main(["report", str(tmp_path / "run")])


def test_result_rows_carry_the_plugin_keys_they_were_appended_with(tmp_path, capsys):
    repo = run_repo(tmp_path)
    plugin = plugin_file(tmp_path)
    assert main(["results", "append", "--ref", "run/t", "--repo", str(repo),
                 "--metrics", str(plugin)]) == 0
    row, = load_index(repo)  # validated on read
    assert row["metrics"][-1] == "best_round" and row["tasks"][0]["best_round"] == 3
    metrics.REGISTRY.pop("best_round")  # a later process without the plugin
    metrics._LOADED.clear()

    def table(*args) -> dict:
        out = tmp_path / "table.csv"
        assert main(["results", "table", "--repo", str(repo), "--column", "best_round",
                     "--column", "late_gain", "--format", "csv", "--output", str(out),
                     "--include-unsandboxed", *args]) == 0
        return next(csv.DictReader(out.open()))

    assert table()["best_round_mean"] == "3.0"  # index rows keep their metrics
    assert table("--branches", "run/*")["best_round_mean"] == ""
    assert table("--branches", "run/*", "--metrics", str(plugin))["best_round_mean"] == "3.0"
    assert float(table("--branches", "run/*")["late_gain_mean"]) == pytest.approx(0.1)


def _row(run: str, seed: int, usd: float, late: float, best_round: int) -> dict:
    return {"run_id": run, "sandboxed": True, "mode": "base", "model": "m", "effort": None, "seed": seed,
            "base": "harness-v0.3.0", "protocol_id": "a" * 12, "improver_source": None,
            "indexed_at": "2026-10-02T00:00:00Z",
            "tasks": [{"task_id": "t", "test": 0.5, "normalized": {"val": 0.4, "test": 0.25 * seed},
                       "late_gain": late, "false_acceptance": {"count": seed, "judged": 2},
                       "cost": {"usd": usd, "wall_s": 1800 * seed}, "ended_reason": "budget_usd",
                       "best_round": best_round, "solved": True}]}


def test_table_columns_are_summary_keys_and_keep_the_standard_names():
    rows = [_row("a", 1, 1.0, 0.1, 3), _row("b", 2, 3.0, 0.3, 5)]
    rec = flatten(rows)[0]
    assert rec["normalized.test"] == rec["normalized_test"] == 0.25
    assert rec["cost.usd"] == rec["usd"] == 1.0 and rec["hours"] == 0.5
    assert rec["false_acceptance.count"] == 1 and rec["best_round"] == 3
    assert "solved" not in rec and "ended_reason" not in rec and rec["ended"] == "budget_usd"
    columns = ["normalized_test", "cost.usd", "late_gain", "false_acceptance.count", "best_round",
               "hours"]
    t = build_table(rows, columns=columns)
    group, = t["groups"]
    assert t["columns"] == columns and "usd" not in group
    assert group["best_round"] == {"n": 2, "mean": 4.0, "std": pytest.approx(2 ** 0.5),
                                   "sem": pytest.approx(1.0)}
    md = table_markdown(t)
    assert "| runs | norm test | cost.usd | late_gain | false_acceptance.count | best_round | " \
           "hours |" in md
    assert "| 2 | 0.375 ± 0.125 | 2 ± 1 | 0.2 ± 0.1 | 1.5 ± 0.5 | 4 ± 1 | 0.75 ± 0.25 |" in md
    header = table_csv(t).splitlines()[0].split(",")
    assert header[7:] == [f"{c}_{s}" for c in columns for s in ("mean", "std", "n")]
    standard = build_table(rows)
    assert standard["columns"] == list(METRICS)
    assert "| norm test | test | best val | 1st-round gain |" in table_markdown(standard)


def test_the_table_fields_are_not_columns(tmp_path):
    repo = run_repo(tmp_path)
    assert main(["results", "append", "--ref", "run/t", "--repo", str(repo)]) == 0
    for name in ("model", "runs"):
        with pytest.raises(SystemExit, match=f"rsib: table fields are not columns: {name};"):
            main(["results", "table", "--repo", str(repo), "--column", "test", "--column", name])
    with pytest.raises(MetricError, match="not columns: task, seeds, ended, run_id;"):
        build_table([_row("a", 1, 1.0, 0.1, 3)],
                    columns=["task", "late_gain", "seeds", "ended", "run_id", "hours"])


def test_metric_keys_cannot_take_the_names_of_table_fields_or_standard_columns(tmp_path):
    repo = run_repo(tmp_path)
    clash = plugin_file(tmp_path, 'from rsibench.metrics import metric\nmetric("clash")(lambda '
                                  'run, task: {"hours": 9.0, "model": 7, "runs": {"n": 1}})\n',
                        "clash.py")
    refused = "task t: metric key.s. hours, model, runs clash with the results table's own fields"
    with pytest.raises(SystemExit, match=f"rsib: run/t, {refused}"):
        main(["results", "append", "--ref", "run/t", "--repo", str(repo), "--metrics", str(clash)])
    with pytest.raises(FileNotFoundError):
        load_index(repo)  # nothing was indexed
    with pytest.raises(SystemExit, match=refused):
        main(["results", "table", "--repo", str(repo), "--branches", "run/*",
              "--metrics", str(clash)])

    rows = [_row("a", 1, 1.0, 0.1, 3)]  # written another way: the table keeps its own meaning
    rows[0]["tasks"][0].update(hours=99.0, model=7)
    rec, = flatten(rows)
    assert rec["hours"] == 0.5 and rec["model"] == "m"
    assert build_table(rows, columns=["hours"])["groups"][0]["hours"]["mean"] == 0.5
