"""Posthoc selection and append-only redo contracts, with a stub evaluator."""

from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rsibench.posthoc as ph
from rsibench.cli import build_parser, main
from rsibench.records import Records


def accepted(rnd, task_id="t", accepted_commit=None):
    return {"task_id": task_id, "round": rnd,
            "accepted_commit": accepted_commit or f"commit-{task_id}-{rnd}",
            "decision": "baseline" if rnd == 0 else "accept", "val": {"score": 0.5}}


def scored(rnd, task_id="t", **extra):
    return {"task_id": task_id, "round": rnd, "commit": f"commit-{task_id}-{rnd}",
            "status": "ok", "score": 9.0, "evaluated_at": f"old-{task_id}-{rnd}", **extra}


def stub_engine(tmp_path, rounds, rows=(), task_ids=("t",)):
    records = Records(tmp_path / "run")
    records.dir.mkdir(parents=True)
    records.path("rounds").write_text("".join(json.dumps(r) + "\n" for r in rounds))
    for row in rows:
        records.append("posthoc", row)

    def snapshot(task, commit, dest):
        dest.mkdir(parents=True)
        (dest / "commit").write_text(commit)
        return dest

    def evaluate(task, workspace, split, out_dir, rnd, consumer):
        assert split == "test" and consumer == "posthoc"
        (out_dir / "evidence").write_text(f"stub evaluation {task.id} {rnd}")
        return {"status": "ok", "score": 1.0, "metrics": {"public": 1, "private": [2]}}

    return SimpleNamespace(
        wt=tmp_path / "run", branch="run/stub", records=records,
        tasks=[SimpleNamespace(id=t, public_metrics=["public"]) for t in task_ids],
        hidden_eval_dir=lambda task_id, name: tmp_path / "private" / task_id / name,
        snapshot_workspace=Mock(side_effect=snapshot),
        run_split_eval=Mock(side_effect=evaluate), serving=Mock(return_value=nullcontext()),
    )


def test_redo_appends_supersedes_and_preserves_evidence(tmp_path, monkeypatch):
    old = scored(10)
    engine = stub_engine(tmp_path, [accepted(0), accepted(10)], [old])
    old_dir = engine.hidden_eval_dir("t", "posthoc-r010")
    old_dir.mkdir(parents=True)
    (old_dir / "evidence").write_text("original evidence")
    original = engine.records.path("posthoc").read_bytes()
    timestamps = iter(["redo-first", "redo-second"])
    monkeypatch.setattr(ph, "now_iso", lambda: next(timestamps))

    first = ph.score_missing(engine, every=99, redo=[10], redo_reason="provider outage")
    second = ph.score_missing(engine, redo=[10])

    assert engine.records.path("posthoc").read_bytes().startswith(original)
    assert engine.records.read("posthoc") == [old, *first, *second]
    assert first[0]["redo"] is True and first[0]["supersedes"] == old["evaluated_at"]
    assert first[0]["redo_reason"] == "provider outage"
    assert first[0]["commit"] == old["commit"]
    assert first[0]["metrics"] == {"public": 1}
    assert second[0]["supersedes"] == first[0]["evaluated_at"]
    assert "redo_reason" not in second[0]
    assert (old_dir / "evidence").read_text() == "original evidence"
    assert (old_dir.with_name("posthoc-r010-redo1") / "evidence").exists()
    assert (old_dir.with_name("posthoc-r010-redo2") / "evidence").exists()
    assert [call.args[3].name for call in engine.run_split_eval.call_args_list] == [
        "posthoc-r010-redo1", "posthoc-r010-redo2"]


def test_redo_supersedes_latest_appended_row_per_task(tmp_path):
    rows = [scored(1, evaluated_at="2099-old"),
            scored(1, "other", evaluated_at="other-last"),
            scored(1, evaluated_at="2000-latest", redo=True, supersedes="2099-old")]
    engine = stub_engine(tmp_path, [accepted(1), accepted(1, "other")], rows,
                         task_ids=("t", "other"))
    added = ph.score_missing(engine, redo=[1])
    assert [(r["task_id"], r["supersedes"]) for r in added] == [
        ("t", "2000-latest"), ("other", "other-last")]


def test_redo_round_without_existing_row(tmp_path):
    engine = stub_engine(tmp_path, [accepted(0), accepted(1)])
    added = ph.score_missing(engine, redo=[1, 1])
    assert len(added) == 1 and added[0]["round"] == 1
    assert added[0]["redo"] is True and added[0]["supersedes"] is None
    assert "redo_reason" not in added[0]


def test_redo_failed_attempt_gets_new_directory(tmp_path):
    engine = stub_engine(tmp_path, [accepted(1)], [scored(1)])
    engine.run_split_eval.side_effect = RuntimeError("stub failure")
    with pytest.raises(RuntimeError, match="stub failure"):
        ph.score_missing(engine, redo=[1])
    assert len(engine.records.read("posthoc")) == 1
    failed = engine.hidden_eval_dir("t", "posthoc-r001-redo1")
    assert (failed / "snapshot" / "commit").exists()
    engine.run_split_eval.side_effect = None
    engine.run_split_eval.return_value = {"status": "ok", "score": 1, "metrics": {}}
    ph.score_missing(engine, redo=[1])
    assert engine.run_split_eval.call_args.args[3].name == "posthoc-r001-redo2"
    assert (failed / "snapshot" / "commit").exists()


@pytest.mark.parametrize("redo", [[2], [1, 2], [99], []])
def test_nonaccepted_redo_errors_before_scoring(tmp_path, redo):
    rejected = {**accepted(2), "accepted_commit": None, "decision": "reject"}
    engine = stub_engine(tmp_path, [accepted(1), rejected], [scored(1)])
    original = engine.records.path("posthoc").read_bytes()
    with pytest.raises(ph.RedoError, match="--redo"):
        ph.score_missing(engine, redo=redo)
    engine.snapshot_workspace.assert_not_called()
    engine.run_split_eval.assert_not_called()
    assert engine.records.path("posthoc").read_bytes() == original
    assert not (tmp_path / "private").exists()


def test_invalid_redo_does_not_start_serving(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path, [accepted(1)])
    monkeypatch.setattr(ph, "RunEngine", Mock(return_value=engine))
    with pytest.raises(ph.RedoError, match="not accepted: 2"):
        ph.posthoc(engine.wt, redo=[2])
    engine.serving.assert_not_called()


def test_no_redo_retains_missing_selection_and_row_shape(tmp_path):
    engine = stub_engine(tmp_path, [accepted(i) for i in range(5)],
                         [scored(0, status="error", score=None)])
    added = ph.score_missing(engine, every=2)
    assert [r["round"] for r in added] == [2, 4]
    assert all(not {"redo", "supersedes", "redo_reason"}.intersection(r) for r in added)
    assert [c.args[3].name for c in engine.run_split_eval.call_args_list] == [
        "posthoc-r002", "posthoc-r004"]
    assert ph.score_missing(engine, every=2) == []


def test_posthoc_commits_appended_records(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path, [accepted(1)], [scored(1)])
    monkeypatch.setattr(ph, "RunEngine", Mock(return_value=engine))
    commit = Mock()
    monkeypatch.setattr(ph.gitops, "commit", commit)
    assert len(ph.posthoc(engine.wt, redo=[1])) == 1
    engine.serving.assert_called_once_with(engine.tasks)
    assert commit.call_args.args[3] == ["records"]


def test_redo_reason_requires_redo(tmp_path):
    engine = stub_engine(tmp_path, [accepted(1)])
    with pytest.raises(ph.RedoError, match="--redo-reason requires --redo"):
        ph.score_missing(engine, redo_reason="outage")
    engine.run_split_eval.assert_not_called()


@pytest.mark.parametrize("value", ["", ",1", "1,", "-1", "wrong"])
def test_cli_rejects_malformed_redo_rounds(value):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["posthoc", "run", "--redo", value])
    assert exc.value.code == 2


def test_cli_passes_redo_options(tmp_path, monkeypatch):
    import rsibench.cli as cli
    run = tmp_path / "run"
    monkeypatch.setattr(cli, "_worktree", lambda _: run)
    monkeypatch.setattr(cli, "_site", lambda _: {})
    posthoc = Mock(return_value=[])
    monkeypatch.setattr(ph, "posthoc", posthoc)
    assert main(["posthoc", str(run), "--redo", "1,6,10,1",
                 "--redo-reason", "invalid model calls"]) == 0
    assert posthoc.call_args.kwargs["redo"] == [1, 6, 10]
    assert posthoc.call_args.kwargs["redo_reason"] == "invalid model calls"


def test_cli_nonaccepted_round_is_a_clear_error(tmp_path, monkeypatch):
    import rsibench.cli as cli
    engine = stub_engine(tmp_path, [accepted(1)])
    monkeypatch.setattr(cli, "_worktree", lambda _: engine.wt)
    monkeypatch.setattr(cli, "_site", lambda _: {})
    monkeypatch.setattr(ph, "RunEngine", Mock(return_value=engine))
    with pytest.raises(SystemExit, match="rsib: --redo rounds are not accepted: 2"):
        main(["posthoc", str(engine.wt), "--redo", "2"])
    engine.serving.assert_not_called()
