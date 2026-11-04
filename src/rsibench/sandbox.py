"""Filesystem sandbox for everything that runs agent-written code.

The sandbox is on by default (`rsib run new --no-sandbox` turns it off; the setting is recorded in
run.json at creation). With it on:

    phases        the harness process and all its tools see the run's worktree, the dev data,
                  the run's scratch directory (HOME, cache, tmp) and the interpreter; nothing else
                  under the hidden roots (other runs, held-out splits, other users' files).
    experiments   container jobs mount only the workspace, the run's artifact and scratch
                  directories and the dev data; jobs without a container run under the same
                  launcher as phases.
    evaluations   the task's evaluator gets $RSIB_SANDBOX, a launcher prefix it must put in front
                  of the solution's command; the solution then sees its own snapshot, the run cache
                  and the I/O directory the evaluator hands it, never the labels.

Hidden roots are the top-level directories that hold the runs, data, artifacts and repository
(e.g. /lustre), plus /home and any site-listed extras. `read_only` in the site profile lists
shared resources agents may use (e.g. a public model cache); everything a solution needs at
evaluation time must be in the workspace or in the run cache ($RSIB_CACHE).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .envs import scrubbed_env

LAUNCHER = Path(__file__).with_name("sandbox_exec.py")
# Most processes (threads included) one sandboxed launch may hold at once: a phase, an experiment,
# a solution under evaluation. Counted per launch (each is its own user namespace), inherited by
# every descendant, so code that forks without end gets EAGAIN instead of taking the machine down.
# A site setting ([sandbox] max_procs, 0: no limit), not part of the run's protocol.
DEFAULT_MAX_PROCS = 8192


class GitViewError(RuntimeError):
    """The read-only Git view of a phase could not be built."""


def _read_gitfile(path: Path) -> str:
    """The Git directory a worktree's .git file names; the file must be a plain file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise GitViewError(f"{path} is not a plain gitfile: {exc.strerror}") from exc
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise GitViewError(f"{path} is not a plain gitfile")
        content = handle.read(4097).decode(errors="replace")
    target = content.removeprefix("gitdir: ").strip()
    if not content.startswith("gitdir: ") or "\n" in target or not os.path.isabs(target) \
            or not os.path.isdir(target):
        raise GitViewError(f"{path} does not name a Git directory")
    return target


def _write_ref(directory: Path, name: str, sha: str) -> None:
    path = directory / name
    if not (name.startswith("refs/") and path.resolve().is_relative_to(directory.resolve())):
        raise GitViewError(f"refusing ref {name!r} outside the view")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sha + "\n")


def top_level(path: Path | str) -> str:
    parts = Path(os.path.abspath(path)).parts
    return os.path.join("/", parts[1]) if len(parts) > 1 else "/"


def python_roots() -> list[str]:
    """Paths the sandbox allows so that the orchestrator's interpreter runs inside it."""
    return interpreter_paths(sys.executable, sys.prefix, sys.base_prefix)


def interpreter_paths(executable: str, prefix: str, base_prefix: str) -> list[str]:
    """The directories an interpreter needs (its venv and the base installation) and the paths
    that reach them through symlinks. A link whose target is one of those directories, or inside
    one, is allowed as it is (uv links e.g. cpython-3.11 to cpython-3.11.15). A link to a
    directory above them (a cluster that reaches the venv through a linked parent, such as
    /fsw/projects -> /fs12/projects) is allowed only as the path through it to each directory
    below: allowing the link itself would bind its whole target tree, held-out data included."""
    roots = {os.path.realpath(prefix), os.path.realpath(base_prefix),
             os.path.dirname(os.path.dirname(os.path.realpath(executable)))}
    paths = set(roots) | {os.path.abspath(prefix), os.path.abspath(base_prefix)}
    for link in links_on_the_way(executable) | links_on_the_way(prefix) | links_on_the_way(base_prefix):
        target = os.path.realpath(link)
        if any(_within(target, r) for r in roots):
            paths.add(link)
        else:
            paths |= {os.path.join(link, os.path.relpath(r, target)) for r in roots
                      if _within(r, target)}
    return sorted(paths)


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def links_on_the_way(path: str) -> set[str]:
    """Every symlink met while resolving `path`, its parent directories included."""
    links, todo, seen = set(), [os.path.abspath(path)], set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        for p in [current, *map(str, Path(current).parents)]:
            if os.path.islink(p):
                links.add(p)
                todo.append(os.path.abspath(os.path.join(os.path.dirname(p), os.readlink(p))))
    return links


def probe(python: str | None = None) -> str | None:
    """None if the launcher works on this machine, else the error it printed."""
    try:
        proc = subprocess.run([python or sys.executable, str(LAUNCHER), "--hide", "/home", "--",
                               "true"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return None if proc.returncode == 0 else (proc.stderr.strip() or f"exit {proc.returncode}")


def run_config(site: dict | None, enabled: bool | None = None) -> dict:
    """The sandbox settings recorded in run.json (part of the protocol fingerprint). On unless the
    caller or the site profile turns it off."""
    cfg = dict((site or {}).get("sandbox") or {})
    on = bool(cfg.get("enabled", True)) if enabled is None else bool(enabled)
    return {"enabled": on,
            "read_only": [str(p) for p in cfg.get("read_only", [])] if on else [],
            "hide": [str(p) for p in cfg.get("hide", [])] if on else [],
            "mask": [str(p) for p in cfg.get("mask", [])] if on else [],
            "git_view": bool(cfg.get("git_view", True)) if on else False}


class Sandbox:
    """Per-run sandbox: directories, environment and launcher command lines."""

    def __init__(self, config: dict | None, scratch_dir: Path, roots: list[Path],
                 max_procs: int = DEFAULT_MAX_PROCS):
        config = config or {}
        self.max_procs = int(max_procs or 0)
        self.enabled = bool(config.get("enabled"))
        # recorded by `rsib run new`; runs created before it keep their phases as they were
        self.git_view = self.enabled and bool(config.get("git_view"))
        self.read_only = [p for p in config.get("read_only", []) if os.path.exists(p)]
        self.mask = list(config.get("mask", []))
        hide = {top_level(p) for p in roots} | {"/home"} | set(config.get("hide", []))
        self.hide = sorted(h for h in hide if h != "/" and os.path.isdir(h))
        self.dir = Path(scratch_dir)  # this run's scratch: home, cache, tmp, bin
        self.home, self.cache, self.tmp, self.bin = (self.dir / n for n in ("home", "cache", "tmp", "bin"))

    def prepare(self) -> None:
        if not self.enabled:
            return
        for d in (self.home, self.cache, self.tmp, self.bin):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LAUNCHER, self.bin / LAUNCHER.name)  # visible to container jobs via a mount

    @property
    def launcher(self) -> str:
        return str(self.bin / LAUNCHER.name)

    def env(self) -> dict:
        """Environment for sandboxed phases and experiments (downloads land in the run cache)."""
        if not self.enabled:
            return {}
        c = self.cache
        return {"HOME": str(self.home), "RSIB_CACHE": str(c), "HF_HOME": str(c / "huggingface"),
                "TORCH_HOME": str(c / "torch"), "XDG_CACHE_HOME": str(c / "xdg"),
                "PIP_CACHE_DIR": str(c / "pip"), "TMPDIR": "/tmp"}

    def prefix(self, ro: list, rw: list, python: str | None = None, home: bool = True,
               tmp: bool = True, cwd: Path | None = None, close: bool = True,
               mask: list | None = None, ro_files: list | None = None) -> list[str]:
        """Launcher argv up to (and with `close`, including) the `--` before the command."""
        args = [python or sys.executable, self.launcher]
        for h in self.hide:
            args += ["--hide", h]
        for p in dict.fromkeys(str(x) for x in list(ro) + self.read_only):
            args += ["--ro", p]
        for p in dict.fromkeys(str(x) for x in rw):
            args += ["--rw", p]
        for m in self.mask + list(mask or []):
            args += ["--mask", str(m)]
        for src, dst in ro_files or []:
            args += ["--ro-file", str(src), str(dst)]
        if home:
            args += ["--home", str(self.home)]
        if tmp:
            args += ["--tmp", str(self.tmp)]
        if cwd:
            args += ["--cwd", str(cwd)]
        if self.max_procs > 0:
            args += ["--max-procs", str(self.max_procs)]
        return args + (["--"] if close else [])

    @property
    def views(self) -> Path:
        """Where this run's phase Git views live: beside the scratch roots, under the hidden
        scratch root, never inside self.dir (experiments mount that read-write)."""
        return self.dir.parent / "_phase-git" / self.dir.name

    def sweep_views(self) -> int:
        """Remove views an orchestrator that was killed mid-phase left behind. Only at the start
        of `rsib run start`: a view is in use while its phase runs."""
        stale = list(self.views.iterdir()) if self.views.is_dir() else []
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
        return len(stale)

    @contextmanager
    def phase_git(self, worktree: Path, expected_branch: str | None = None):
        """A disposable, self-contained Git view of HEAD and this run's ancestor tags. Yields the
        view's Git directory and a gitfile pointing at it, to mount read-only over <wt>/.git.

        Never clone or copy the shared object database: its packs, refs, configuration and
        unreachable objects can contain other runs. Export only reachable objects into a fresh
        repository, with no templates, alternates, remotes, hooks or reflogs. The branch comes from
        the worktree's own Git metadata (hidden from agents), not from the agent-writable records.
        Raises GitViewError when the view cannot be built.
        """
        env = scrubbed_env()
        # Site pass-through settings must not select a different repository/object store or
        # inject Git configuration. Source-local config is read, but never copied into the view.
        env = {k: v for k, v in env.items() if not k.startswith("GIT_")}
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"})

        # Read <wt>/.git once, as the plain file the launcher will cover, and select the source
        # repository explicitly from it: nothing below resolves <wt>/.git again.
        source = _read_gitfile(worktree / ".git")

        def git(args, cwd=worktree, view=False, **kwargs):
            options = {"check": True, "stdout": subprocess.PIPE, **kwargs}
            try:
                return subprocess.run(["git", *args], cwd=cwd,
                                      env=env if view else {**env, "GIT_DIR": source},
                                      stderr=subprocess.PIPE, **options)
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or b"").decode(errors="replace").strip()[-1000:]
                raise GitViewError(f"git {args[0]} failed (exit {exc.returncode}): "
                                   f"{detail}") from exc

        branch = git(["symbolic-ref", "--short", "HEAD"]).stdout.decode().strip()
        git(["check-ref-format", "--branch", branch])
        if expected_branch is not None and branch != expected_branch:
            raise GitViewError(f"the worktree is on {branch!r}, the run records "
                               f"{expected_branch!r}")
        head = git(["rev-parse", "HEAD^{commit}"]).stdout.decode().strip()
        tags = [line.split() for line in git(
            ["for-each-ref", "--merged", head, "--format=%(refname) %(objectname)",
             f"refs/tags/{branch}/"]).stdout.decode().splitlines()]
        self.views.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.views) as temp:
            directory = Path(temp) / "git"
            git(["init", "--bare", "--template=", str(directory)], view=True)
            packs = directory / "objects" / "pack"
            packs.mkdir(parents=True, exist_ok=True)
            # Without --thin no object outside the exported set can serve as a delta base.
            # Streamed and indexed inside the view: with a base name, pack-objects writes its temp
            # file into the source object directory and renames it, which fails (EXDEV) when the
            # scratch root is on another filesystem.
            pack = packs / "pack-view.pack"
            with open(pack, "wb") as out:
                git(["pack-objects", "--revs", "--threads=2", "-q", "--stdout"], stdout=out,
                    input=("\n".join([head, *(sha for _, sha in tags)]) + "\n").encode())
            git(["index-pack", "--threads=2", str(pack)], cwd=directory, view=True)
            _write_ref(directory, f"refs/heads/{branch}", head)
            for name, sha in tags:
                _write_ref(directory, name, sha)
            (directory / "HEAD").write_text(f"ref: refs/heads/{branch}\n")
            # A private index makes status/diff useful without touching the real worktree index.
            git(["read-tree", "HEAD"], cwd=directory, view=True)
            # Found through <wt>/.git, a non-bare view has the worktree as its work tree.
            git(["config", "--file", str(directory / "config"), "core.bare", "false"], view=True)
            gitfile = Path(temp) / "gitfile"
            gitfile.write_text(f"gitdir: {directory}\n")
            yield directory, gitfile

    def child_prefix_json(self, workspace: Path, python: str, extra_ro: list | None = None) -> str:
        """$RSIB_SANDBOX for evaluators: the solution sees its snapshot and the run cache only.
        The evaluator appends `--rw <io dir> [--cwd <io dir>] --` and the command."""
        ro = [workspace, self.cache] + list(extra_ro or [])
        return json.dumps(self.prefix(ro, [], python=python, home=False, tmp=False, close=False))

    def container_mounts(self, ro: list, rw: list) -> list[str]:
        mounts = [f"{os.path.realpath(p)}:{os.path.realpath(p)}:ro" for p in list(ro) + self.read_only]
        mounts += [f"{os.path.realpath(p)}:{os.path.realpath(p)}" for p in rw]
        return list(dict.fromkeys(mounts))


def kill_leftovers(token: str) -> int:
    """Stop processes of this user that carry RSIB_SANDBOX_TOKEN=<token> (background jobs an
    agent left behind when its phase ended). Returns how many were signalled."""
    import signal

    needle = f"RSIB_SANDBOX_TOKEN={token}".encode()
    uid, me, n = os.getuid(), os.getpid(), 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            if needle in (entry / "environ").read_bytes().split(b"\0"):
                os.kill(int(entry.name), signal.SIGKILL)
                n += 1
        except (OSError, ProcessLookupError):
            continue
    return n
