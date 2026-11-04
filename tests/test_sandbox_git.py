"""Run history is readable without exposing the shared repository's other objects or metadata."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import call, write_script

from rsibench import sandbox as sandbox_module
from rsibench.sandbox import GitViewError, Sandbox, _write_ref, probe, python_roots, run_config

namespaces = pytest.mark.skipif(probe() is not None,
                                reason=f"unprivileged user namespaces unavailable: {probe()}")


def git(cwd, *args, check=True, **kwargs):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                           *args], cwd=cwd, check=check, capture_output=True, text=True, **kwargs)


@pytest.fixture
def history(tmp_path):
    repo, worktree = tmp_path / "repo", tmp_path / "own-run"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    source = "starter\n" + "shared-looking content\n" * 1000
    (repo / "policy.py").write_text(source)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "release")
    base = git(repo, "rev-parse", "HEAD").stdout.strip()
    branch = "run/task/model/full/s1"
    git(repo, "worktree", "add", "-qb", branch, str(worktree), base)
    git(worktree, "tag", f"{branch}/task/r0")
    (worktree / "policy.py").write_text(source + "accepted policy\n")
    git(worktree, "commit", "-qam", "accepted candidate")
    accepted = git(worktree, "rev-parse", "HEAD").stdout.strip()
    git(worktree, "tag", "-a", f"{branch}/task/r1", "-m", "own accepted version")
    rejected_source = source + "rejected candidate, still recoverable\n"
    (worktree / "policy.py").write_text(rejected_source)
    git(worktree, "commit", "-qam", "rejected candidate")
    rejected = git(worktree, "rev-parse", "HEAD").stdout.strip()
    git(worktree, "revert", "--no-edit", rejected)

    # Other models' solutions share a pack with our history, and a similar blob may be stored as
    # a delta against ours (or ours against it). A local clone/copy of .git would disclose all of
    # these even without their refs.
    git(repo, "checkout", "-qb", "run/foreign/model/base/s2", base)
    (repo / "policy.py").write_text(source + "FOREIGN MODEL PRIVATE SOLUTION\n")
    git(repo, "commit", "-qam", "foreign run secret")
    foreign_commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    foreign_blob = git(repo, "rev-parse", "HEAD:policy.py").stdout.strip()
    git(repo, "tag", "-a", "run/foreign/model/base/s2/task/r1", "-m", "foreign record secret")
    # A tag with our prefix is insufficient authorization if it points off our ancestry.
    git(repo, "tag", f"{branch}/task/r99")
    git(repo, "gc", "--prune=now")
    unreachable = git(repo, "hash-object", "-w", "--stdin", input="unreachable private result\n")
    git(repo, "remote", "add", "origin", "https://private.example/other-run.git")
    git(repo, "config", "core.hooksPath", "/hidden/private-hooks")
    (repo / ".git" / "hooks" / "private-hook").write_text("private hook contents\n")
    git(repo, "replace", accepted, foreign_commit)
    scratch = tmp_path / "scratch"
    sb = Sandbox({"enabled": True, "git_view": True}, scratch, [repo, worktree])
    sb.prepare()
    return SimpleNamespace(repo=repo, worktree=worktree, branch=branch, sb=sb,
                           accepted=accepted, rejected=rejected, rejected_source=rejected_source,
                           foreign=[foreign_commit, foreign_blob, unreachable.stdout.strip()])


def views_left(sb):
    return list(sb.views.iterdir()) if sb.views.exists() else []


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def test_phase_git_contains_only_own_reachable_objects_and_recovers_rejected_candidate(history):
    h = history
    before = hashes(h.repo / ".git")
    with h.sb.phase_git(h.worktree, h.branch) as (view, gitfile):
        def own(*args, **kwargs):
            return git(h.worktree, "--git-dir", str(view), *args, **kwargs)

        assert gitfile.read_text() == f"gitdir: {view}\n"
        assert own("config", "core.bare").stdout.strip() == "false"
        assert own("show", f"{h.rejected}:policy.py").stdout == h.rejected_source
        assert own("rev-parse", f"{h.branch}/task/r1^{{commit}}").stdout.strip() == h.accepted
        refs = own("for-each-ref", "--format=%(refname)").stdout.splitlines()
        assert refs == [f"refs/heads/{h.branch}", f"refs/tags/{h.branch}/task/r0",
                        f"refs/tags/{h.branch}/task/r1"]
        expected = {line.split()[0] for line in git(
            h.worktree, "--no-replace-objects", "rev-list", "--objects", "HEAD",
            f"{h.branch}/task/r0", f"{h.branch}/task/r1").stdout.splitlines()}
        actual = set(own("cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
                     .stdout.splitlines())
        assert actual == expected
        for secret in h.foreign:
            assert own("cat-file", "-e", secret, check=False).returncode != 0
        assert own("remote").stdout == ""
        assert not (view / "hooks").exists()
        assert not (view / "logs").exists()
        assert not (view / "objects" / "info" / "alternates").exists()
        assert "private" not in (view / "config").read_text()
        assert own("fsck", "--full").returncode == 0
    assert not view.exists() and not gitfile.exists()
    assert hashes(h.repo / ".git") == before


def test_phase_git_takes_the_branch_from_git_and_refuses_a_tampered_one(history, tmp_path):
    h = history
    escaped = tmp_path / "escaped"
    for recorded in (f"../../../../../../..{escaped}", h.branch + "-other"):
        with pytest.raises(GitViewError, match="the run records"):
            with h.sb.phase_git(h.worktree, recorded):
                pass
    assert not escaped.exists()
    assert views_left(h.sb) == []
    # refs are written only below the view, whatever their name claims
    view = tmp_path / "view"
    view.mkdir()
    for name in ("refs/heads/../../../escaped", "HEAD", "objects/x"):
        with pytest.raises(GitViewError, match="outside the view"):
            _write_ref(view, name, "0" * 40)
    assert not escaped.exists() and list(view.iterdir()) == []


def test_phase_git_ignores_git_environment_templates_and_poisoned_old_path(history, monkeypatch):
    h = history
    poisoned = h.sb.dir / "git"
    poisoned.symlink_to(h.repo / ".git", target_is_directory=True)
    template = h.sb.dir / "template"
    (template / "hooks").mkdir(parents=True)
    (template / "hooks" / "secret-hook").write_text("must not be copied\n")
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(template))
    monkeypatch.setenv("GIT_DIR", str(h.repo / ".git"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(h.repo / ".git" / "objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/secret-hooks")
    monkeypatch.setenv("RSIB_PASS_ENV", "GIT_DIR,GIT_TEMPLATE_DIR,GIT_ALTERNATE_OBJECT_DIRECTORIES,"
                       "GIT_CONFIG_COUNT,GIT_CONFIG_KEY_0,GIT_CONFIG_VALUE_0")
    before = hashes(h.repo / ".git")
    with pytest.raises(RuntimeError, match="phase failed"):
        with h.sb.phase_git(h.worktree, h.branch) as (view, _):
            assert view.parent != poisoned
            assert not (view / "hooks").exists()
            assert not (view / "objects" / "info" / "alternates").exists()
            assert "secret" not in (view / "config").read_text()
            raise RuntimeError("phase failed")
    assert not view.exists()
    assert poisoned.is_symlink()
    assert hashes(h.repo / ".git") == before


def test_phase_git_is_outside_experiment_and_container_mounts(history):
    h = history
    dev, artifacts = h.sb.dir.parent / "dev", h.sb.dir.parent / "artifacts"
    dev.mkdir()
    artifacts.mkdir()
    with h.sb.phase_git(h.worktree, h.branch) as (view, gitfile):
        assert view.is_relative_to(h.sb.views) and gitfile.is_relative_to(h.sb.views)
        assert h.sb.views == h.sb.dir.parent / "_phase-git" / h.sb.dir.name
        assert not view.is_relative_to(h.sb.dir)
        prefix = h.sb.prefix(python_roots() + [dev], [h.worktree, artifacts, h.sb.dir])
        mounted = [Path(prefix[i + 1]).resolve() for i, option in enumerate(prefix[:-1])
                   if option in ("--ro", "--rw")]
        assert not any(view.is_relative_to(root) for root in mounted)
        mounts = h.sb.container_mounts(ro=[dev], rw=[h.worktree, artifacts, h.sb.dir])
        assert not any(view.is_relative_to(Path(mount.split(":", 1)[0])) for mount in mounts)


def test_git_view_is_recorded_for_new_runs_and_part_of_the_protocol_id():
    from rsibench.report import protocol_id

    assert run_config({})["git_view"] is True
    assert run_config({"sandbox": {"git_view": False}})["git_view"] is False
    assert run_config({}, enabled=False)["git_view"] is False
    assert Sandbox({"enabled": True}, Path("/x/run"), []).git_view is False  # recorded before
    assert Sandbox({"enabled": False, "git_view": True}, Path("/x/run"), []).git_view is False
    run = {"protocol": {"max_rounds": 3}, "budgets": {"t": {"usd": 1}},
           "tasks": [{"id": "t", "version_hash": "abc"}],
           "sandbox": {"enabled": True, "read_only": ["/models"]}}
    key = {"protocol": run["protocol"], "budgets": run["budgets"], "tasks": [["t", "abc"]],
           "sandbox": {"read_only": ["/models"]}}
    earlier = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    assert protocol_id(run) == earlier  # runs without the setting keep their id
    assert protocol_id({**run, "sandbox": {**run["sandbox"], "git_view": False}}) == earlier
    assert protocol_id({**run, "sandbox": {**run["sandbox"], "git_view": True}}) != earlier


# ---- the engine's phase launch ------------------------------------------------------------------

def _engine(sandbox, config):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.runs import new_run

    _, worktree = new_run(["toy-regression"], "mock:finish", mode="base", seed=42,
                          sandbox=config, repo=sandbox.repo, log=lambda *a: None)
    logs = []
    return RunEngine(worktree, log=logs.append), Path(worktree), logs


def new_run_for(sandbox, seed):
    from rsibench.runs import new_run

    branch, worktree = new_run(["toy-regression"], "mock:finish", mode="base", seed=seed,
                               sandbox={"enabled": True, "git_view": True}, repo=sandbox.repo,
                               log=lambda *a: None)
    return branch, Path(worktree)


def _launch(engine, monkeypatch, start, leftovers=None):
    from rsibench.budget import Budget
    from rsibench.orchestrator import engine as engine_module
    from rsibench.orchestrator.engine import TaskState

    # Replace this module's subprocess binding only: the real Git exporter still runs Git.
    monkeypatch.setattr(engine_module, "subprocess", SimpleNamespace(
        Popen=start, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired))
    if leftovers:
        monkeypatch.setattr(engine_module, "kill_leftovers", leftovers)
    task = engine.tasks[0]
    return engine._run_phase_once("accept", task, 1, Budget(engine.budget_limits(task)),
                                  TaskState())


def _options(command, name):
    return [command[i + 1:i + 1 + (2 if name == "--ro-file" else 1)]
            for i, option in enumerate(command) if option == name]


def test_phase_launch_mounts_a_gitfile_exports_no_git_variables_and_cleans_up(sandbox, monkeypatch):
    engine, worktree, _ = _engine(sandbox, {"enabled": True, "git_view": True})
    captured, events = {}, []

    def start(cmd, **kwargs):
        captured.update(cmd=cmd, env=kwargs["env"])
        (gitfile, target), = _options(cmd, "--ro-file")
        directory = Path(Path(gitfile).read_text().removeprefix("gitdir: ").strip())
        captured.update(gitfile=Path(gitfile), target=target, directory=directory)
        env = kwargs["env"]  # the test's own environment carries the poisoned GIT_* values
        head = git(worktree, "--git-dir", str(directory), "rev-parse", "HEAD", env=env).stdout
        assert head == git(worktree, "rev-parse", "HEAD", env=env).stdout
        events.append("start")
        raise OSError("process could not start")

    def leftovers(marker):
        # The view remains available until all of the phase's processes have been stopped.
        assert captured["directory"].exists()
        events.append("leftovers")
        return 0

    monkeypatch.setenv("RSIB_PASS_ENV", "GIT_COMMON_DIR,GIT_OBJECT_DIRECTORY,GIT_DIR,"
                       "GIT_ALTERNATE_OBJECT_DIRECTORIES,GIT_CONFIG_COUNT,GIT_CONFIG_KEY_0,"
                       "GIT_CONFIG_VALUE_0")
    for name in ("GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_DIR"):
        monkeypatch.setenv(name, "/other-run/private.git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/other-run")
    with pytest.raises(OSError, match="process could not start"):
        _launch(engine, monkeypatch, start, leftovers)
    assert events == ["start", "leftovers"]
    assert not captured["directory"].exists() and not captured["gitfile"].exists()
    assert captured["target"] == str(worktree / ".git")
    assert not [name for name in captured["env"] if name.startswith("GIT_")]
    command = captured["cmd"]
    assert ["--ro", str(captured["directory"])] in [command[i:i + 2] for i in range(len(command))]
    assert [str(worktree / ".git")] not in _options(command, "--mask")
    context = json.loads((worktree / ".rsib" / "ctx-accept.json").read_text())
    assert context["environment"]["git_history"] is True


@pytest.mark.parametrize("failure", ["pack-objects", "tampered-branch"])
def test_phase_runs_without_the_view_when_it_cannot_be_built(sandbox, monkeypatch, failure):
    engine, worktree, logs = _engine(sandbox, {"enabled": True, "git_view": True})
    if failure == "pack-objects":
        def run(args, **kwargs):
            if "pack-objects" in args:
                raise subprocess.CalledProcessError(128, args, stderr=b"fatal: simulated failure\n")
            return subprocess.run(args, **kwargs)

        monkeypatch.setattr(sandbox_module, "subprocess", SimpleNamespace(
            run=run, PIPE=subprocess.PIPE, CalledProcessError=subprocess.CalledProcessError))
        reason = "git pack-objects failed (exit 128): fatal: simulated failure"
    else:
        engine.branch = "../../../../escaped"  # e.g. records/run.json edited by an agent
        reason = "the run records '../../../../escaped'"
    captured = {}

    def start(cmd, **kwargs):
        captured.update(cmd=cmd, env=kwargs["env"])
        return subprocess.Popen(["true"], start_new_session=True)

    monkeypatch.setenv("RSIB_PASS_ENV", "GIT_DIR")
    monkeypatch.setenv("GIT_DIR", "/other-run/private.git")
    outcome = _launch(engine, monkeypatch, start)
    assert outcome.status == "ok"
    assert reason in outcome.warning and reason in outcome.record()["warning"]
    assert any(reason in line and "without the Git history view" in line for line in logs)
    command = captured["cmd"]
    assert [str(worktree / ".git")] in _options(command, "--mask")
    assert not _options(command, "--ro-file")
    assert not [name for name in captured["env"] if name.startswith("GIT_")]
    context = json.loads((worktree / ".rsib" / "ctx-accept.json").read_text())
    assert context["environment"]["git_history"] is False
    assert views_left(engine.sandbox) == []


def test_runs_recorded_without_git_view_launch_phases_as_before(sandbox, monkeypatch):
    engine, worktree, _ = _engine(sandbox, {"enabled": True})
    captured = {}

    def no_view(*args, **kwargs):
        raise AssertionError("a run without git_view must not build a view")

    def start(cmd, **kwargs):
        captured.update(cmd=cmd, env=kwargs["env"])
        return subprocess.Popen(["true"], start_new_session=True)

    monkeypatch.setattr(engine.sandbox, "phase_git", no_view)
    monkeypatch.setenv("RSIB_PASS_ENV", "GIT_AUTHOR_NAME")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "kept as before")
    outcome = _launch(engine, monkeypatch, start)
    assert outcome.status == "ok" and not outcome.warning and "warning" not in outcome.record()
    assert not _options(captured["cmd"], "--ro-file")
    assert [str(worktree / ".git")] not in _options(captured["cmd"], "--mask")
    assert captured["env"]["GIT_AUTHOR_NAME"] == "kept as before"
    context = json.loads((worktree / ".rsib" / "ctx-accept.json").read_text())
    assert context["environment"]["git_history"] is False


def test_start_sweeps_only_this_runs_stale_views(sandbox, monkeypatch):
    engine, _, logs = _engine(sandbox, {"enabled": True, "git_view": True})
    stale = engine.sandbox.views / "tmp-killed-phase"
    (stale / "git" / "objects").mkdir(parents=True)
    other = engine.sandbox.views.parent / "another-run" / "tmp-live-phase"
    other.mkdir(parents=True)

    def stop():
        raise RuntimeError("stop after the sweep")

    monkeypatch.setattr(engine, "_rewind_partial_round", stop)
    with pytest.raises(RuntimeError, match="stop after the sweep"):
        engine.start()
    assert not stale.exists() and other.is_dir()
    assert "removed 1 Git view(s) left by an interrupted phase" in logs


# ---- inside the namespace -----------------------------------------------------------------------

EXPERIMENT_CHECK = r"""
import json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
result = {"visible": directory.exists()}
try:
    (directory / "index").write_text("experiment tampered phase history")
    result["write"] = "allowed"
except OSError:
    result["write"] = "refused"
print(json.dumps(result))
"""


@namespaces
def test_experiment_namespace_cannot_reach_phase_git_even_with_its_path(history):
    h = history
    dev, artifacts = h.sb.dir.parent / "dev", h.sb.dir.parent / "artifacts"
    dev.mkdir()
    artifacts.mkdir()
    with h.sb.phase_git(h.worktree, h.branch) as (view, _):
        before = hashes(view)
        prefix = h.sb.prefix(python_roots() + [dev], [h.worktree, artifacts, h.sb.dir],
                             cwd=h.worktree)
        proc = subprocess.run(prefix + [sys.executable, "-c", EXPERIMENT_CHECK, str(view)],
                              cwd=h.worktree, env={**os.environ, **h.sb.env()},
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == {"visible": False, "write": "refused"}
        assert hashes(view) == before


NAMESPACE_CHECK = r"""
import json, os, pathlib, subprocess, sys
worktree, source, rejected, secret, scratch = sys.argv[1:]
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)
other, clone = os.path.join(scratch, "other"), os.path.join(scratch, "clone")
git("init", "-q", other)
git("-C", other, "-c", "user.name=a", "-c", "user.email=a@b", "commit", "-q", "--allow-empty",
    "-m", "other repository")
gitfile = (pathlib.Path(worktree) / ".git").read_text()
result = {"rejected": git("show", rejected + ":policy.py").stdout,
          "log": git("log", "--oneline").stdout,
          "refs": git("for-each-ref", "--format=%(refname)").stdout,
          "status": git("status", "--short").stdout,
          "foreign": git("cat-file", "-e", secret).returncode,
          "source_visible": pathlib.Path(source).exists(),
          "gitfile": gitfile,
          "write_ref": git("update-ref", "refs/heads/forbidden", "HEAD").returncode,
          "other_log": git("-C", other, "log", "--format=%s").stdout,
          "clone": git("clone", "-q", other, clone).returncode,
          "clone_log": git("-C", clone, "log", "--format=%s").stdout}
try:
    (pathlib.Path(gitfile.removeprefix("gitdir: ").strip()) / "index").write_text("tampered")
    result["write_index"] = "allowed"
except OSError:
    result["write_index"] = "refused"
try:
    (pathlib.Path(worktree) / ".git").write_text("gitdir: /elsewhere\n")
    result["write_gitfile"] = "allowed"
except OSError:
    result["write_gitfile"] = "refused"
print(json.dumps(result))
"""


@namespaces
def test_sandbox_git_is_read_only_and_leaves_other_repositories_alone(history):
    h = history
    before, gitfile_before = hashes(h.repo / ".git"), (h.worktree / ".git").read_text()
    with h.sb.phase_git(h.worktree, h.branch) as (view, gitfile):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        prefix = h.sb.prefix(python_roots() + [view], [h.worktree, h.sb.dir], cwd=h.worktree,
                             ro_files=[(gitfile, h.worktree / ".git")])
        proc = subprocess.run(prefix + [sys.executable, "-c", NAMESPACE_CHECK, str(h.worktree),
                              str(h.repo / ".git"), h.rejected, h.foreign[0], str(h.sb.dir)],
                              cwd=h.worktree, env={**env, **h.sb.env()}, capture_output=True,
                              text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)
        assert result["gitfile"] == f"gitdir: {view}\n"
        assert result["rejected"] == h.rejected_source
        assert "rejected candidate" in result["log"]
        assert "foreign" not in result["log"] + result["refs"]
        assert result["status"] == ""
        assert result["foreign"] != 0
        assert not result["source_visible"]
        assert result["write_ref"] != 0 and result["write_index"] == "refused"
        assert result["write_gitfile"] == "refused"
        assert result["other_log"] == result["clone_log"] == "other repository\n"
        assert result["clone"] == 0
    assert hashes(h.repo / ".git") == before
    assert (h.worktree / ".git").read_text() == gitfile_before


@pytest.mark.parametrize("openat2", [True, False])
def test_open_no_symlinks_pins_regular_files_and_refuses_any_symlink(tmp_path, monkeypatch,
                                                                     openat2):
    from rsibench import sandbox_exec

    monkeypatch.setattr(sandbox_exec, "_use_openat2", openat2)
    (tmp_path / "real").mkdir()
    target = tmp_path / "real" / "file"
    target.write_text("x")
    (tmp_path / "linked-dir").symlink_to(tmp_path / "real")
    (tmp_path / "real" / "linked-file").symlink_to(target)
    fd = sandbox_exec.open_no_symlinks(str(target))
    try:
        assert os.path.samestat(os.fstat(fd), os.stat(target))
    finally:
        os.close(fd)
    refused = {str(tmp_path / "linked-dir" / "file"): "symlink",
               str(tmp_path / "real" / "linked-file"): "symlink",
               str(tmp_path / "real"): "not a regular file",
               str(tmp_path / "real" / ".." / "real" / "file"): "normalized",
               "relative/file": "normalized",
               str(tmp_path / "missing"): "No such file"}
    for path, reason in refused.items():
        with pytest.raises(sandbox_exec.SandboxError, match=reason):
            sandbox_exec.open_no_symlinks(path)


@namespaces
def test_launcher_ro_file_pins_both_files_inside_writable_trees(tmp_path):
    hidden, tree, readonly = tmp_path / "hidden", tmp_path / "tree", tmp_path / "readonly"
    for d in (hidden, tree / "real", readonly):
        d.mkdir(parents=True)
    source, target = hidden / "gitfile", tree / "real" / "target"
    source.write_text("gitdir: /view\n")
    target.write_text("original\n")
    (readonly / "file").write_text("read-only tree\n")
    (tree / "link").symlink_to(tree / "real")
    (tree / "real" / "file-link").symlink_to(target)
    (tmp_path / "source-dir-link").symlink_to(hidden)
    base = [sys.executable, str(sandbox_module.LAUNCHER), "--hide", "/home", "--hide",
            str(hidden), "--rw", str(tree), "--ro", str(readonly)]
    check = ("cat {t}; ls -A {h} | wc -l; (echo x > {t}) 2> /dev/null || echo WRITE_REFUSED"
             .format(t=target, h=hidden))
    ok = subprocess.run(base + ["--ro-file", str(source), str(target), "--", "sh", "-c", check],
                        capture_output=True, text=True, timeout=60)
    # the source, under a covered root, is mounted from its descriptor
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.split("\n") == ["gitdir: /view", "0", "WRITE_REFUSED", ""]
    refused = {(source, Path("/etc/hostname")): "outside the read-write trees",
               (source, readonly / "file"): "outside the read-write trees",
               (source, tree / "link" / "target"): "symlink",
               (source, tree / "real" / "file-link"): "symlink",
               (source, tree / "real" / "missing"): "No such file",
               (source, tree / "real"): "not a regular file",
               (tmp_path / "source-dir-link" / "gitfile", target): "symlink"}
    for (src, dst), reason in refused.items():
        proc = subprocess.run(base + ["--ro-file", str(src), str(dst), "--", "true"],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 126 and reason in proc.stderr, (src, dst, proc.stderr)
    assert target.read_text() == "original\n" and not (tree / "real" / "missing").exists()


@namespaces
def test_a_symlinked_worktree_gitfile_falls_back_to_a_phase_without_the_view(sandbox,
                                                                           monkeypatch):
    engine, worktree, logs = _engine(sandbox, {"enabled": True, "git_view": True})
    copy = sandbox.tmp / "gitfile-copy"
    copy.write_text((worktree / ".git").read_text())
    (worktree / ".git").unlink()
    (worktree / ".git").symlink_to(copy)
    captured = {}

    def start(cmd, **kwargs):
        captured.update(cmd=cmd)
        return subprocess.Popen(["true"], start_new_session=True)

    outcome = _launch(engine, monkeypatch, start)
    assert "is not a plain gitfile" in outcome.warning
    command = captured["cmd"]
    assert not _options(command, "--ro-file") and [str(worktree / ".git")] not in _options(
        command, "--mask")
    launcher = command.index(engine.sandbox.launcher)
    sandboxed = command[launcher - 1:command.index("--", launcher) + 1] + ["true"]
    proc = subprocess.run(sandboxed, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr  # not the launcher's 126
    assert views_left(engine.sandbox) == []


def test_identity_comes_from_git_so_a_tampered_run_json_cannot_sweep_another_run(sandbox,
                                                                                monkeypatch):
    from rsibench.orchestrator.engine import RunEngine

    victim, victim_wt, _ = _engine(sandbox, {"enabled": True, "git_view": True})
    active = victim.sandbox.views / "tmp-active-phase"
    (active / "git").mkdir(parents=True)
    _, intruder = new_run_for(sandbox, seed=43)
    run_json = intruder / "records" / "run.json"
    committed = run_json.read_text()
    tampered = json.loads(committed)
    tampered["branch"] = victim.branch  # left behind by a phase of the intruding run
    tampered["sandbox"] = {"enabled": False}
    run_json.write_text(json.dumps(tampered))
    logs = []
    engine = RunEngine(intruder, log=logs.append)
    assert engine.branch != victim.branch and engine.sandbox.git_view
    assert engine.sandbox.views != victim.sandbox.views
    assert run_json.read_text() == committed
    assert "records/run.json differed from its committed version; restored it" in logs
    own_stale = engine.sandbox.views / "tmp-killed-phase"
    own_stale.mkdir(parents=True)

    def stop():
        raise RuntimeError("stop after the sweep")

    monkeypatch.setattr(engine, "_rewind_partial_round", stop)
    with pytest.raises(RuntimeError, match="stop after the sweep"):
        engine.start()
    assert active.is_dir() and not own_stale.exists()


IN_PHASE = """cd "$RSIB_WORKTREE" && {
  echo "GITFILE=$(cat .git)"
  echo "LOG<<"; git log --format=%s; echo ">>LOG"
  echo "REFS<<"; git for-each-ref --format='%(refname)'; echo ">>REFS"
  git status --short > /dev/null; echo "STATUS_RC=$?"
  env | grep '^GIT_' || echo NO_GIT_ENV
  o="$RSIB_CACHE/other"; git init -q "$o"
  git -C "$o" -c user.name=a -c user.email=a@b commit -q --allow-empty -m other-repository-commit
  echo "OTHER=$(git -C "$o" log --format=%s)"
  git clone -q "$o" "$RSIB_CACHE/clone"; echo "CLONE_RC=$?"
  echo "CLONED=$(git -C "$RSIB_CACHE/clone" log --format=%s)"
  git update-ref refs/heads/forbidden HEAD 2> /dev/null; echo "REF_RC=$?"
} > records/notes/toy-regression.md 2>&1"""


@namespaces
def test_sandboxed_phase_reads_its_own_history_and_uses_git_normally_elsewhere(sandbox):
    from rsibench.orchestrator.engine import RunEngine
    from rsibench.records import Records
    from rsibench.runs import new_run

    _, foreign = new_run(["toy-regression"], "mock:finish", mode="base", seed=7,
                         repo=sandbox.repo, log=lambda *a: None)
    (Path(foreign) / "workspace" / "toy-regression" / "secret.py").write_text("x = 1\n")
    git(foreign, "add", "-A")
    git(foreign, "commit", "-qm", "FOREIGN-RUN-SECRET")
    script = {"executor": {"1": [{"tool_calls": [call("bash", command=IN_PHASE)]},
                                 {"tool_calls": [call("finish", summary="probed git")]}]}}
    branch, wt = new_run(["toy-regression"], "mock:scripted", mode="base", seed=8,
                         model_params={"script": str(write_script(sandbox, script, "git.json"))},
                         protocol_overrides={"max_rounds": 1},
                         sandbox={"enabled": True, "git_view": True},
                         repo=sandbox.repo, log=lambda *a: None)
    engine = RunEngine(wt, log=lambda *a: None)
    engine.start()
    notes = (Path(wt) / "records" / "notes" / "toy-regression.md").read_text()
    lines = notes.splitlines()
    log = notes.split("LOG<<\n")[1].split(">>LOG")[0].splitlines()
    refs = notes.split("REFS<<\n")[1].split(">>REFS")[0].splitlines()
    assert lines[0].startswith("GITFILE=gitdir: ") and "/_phase-git/" in lines[0]
    assert f"init {branch}" in log and "test base" in log
    assert set(log) <= set(git(wt, "log", "--format=%s", branch).stdout.splitlines())
    assert "FOREIGN-RUN-SECRET" not in notes
    assert f"refs/heads/{branch}" in refs and f"refs/tags/{branch}/toy-regression/r0" in refs
    assert all(r == f"refs/heads/{branch}" or r.startswith(f"refs/tags/{branch}/") for r in refs)
    for expected in ("STATUS_RC=0", "NO_GIT_ENV", "OTHER=other-repository-commit", "CLONE_RC=0",
                     "CLONED=other-repository-commit"):
        assert expected in lines, notes
    assert "REF_RC=0" not in lines
    executor = Records(wt).read("rounds")[1]["phases"]["executor"]
    assert executor["status"] == "finished" and "warning" not in executor
    assert views_left(engine.sandbox) == []
