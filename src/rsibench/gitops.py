"""Thin wrappers around the git CLI.

Run branches are written only by the orchestrator. Its commits use a fixed bot identity and
carry machine-readable trailers (Rsib-Round, Rsib-Scope, ...), so `git log` on a run branch is
the lineage of the run: every candidate, every harness edit, every revert.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

BOT_NAME = os.environ.get("RSIB_GIT_NAME", "rsib-orchestrator")
BOT_EMAIL = os.environ.get("RSIB_GIT_EMAIL", "rsib-orchestrator@users.noreply.github.com")


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: Path, check: bool = True, env: dict | None = None,
        input_text: str | None = None) -> str:
    full_env = dict(os.environ)
    full_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        full_env.update(env)
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), env=full_env, text=True, input=input_text,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def bot_env() -> dict:
    return {
        "GIT_AUTHOR_NAME": BOT_NAME, "GIT_AUTHOR_EMAIL": BOT_EMAIL,
        "GIT_COMMITTER_NAME": BOT_NAME, "GIT_COMMITTER_EMAIL": BOT_EMAIL,
    }


def rev_parse(cwd: Path, ref: str = "HEAD") -> str:
    return git(["rev-parse", "--verify", f"{ref}^{{commit}}"], cwd).strip()


def current_branch(cwd: Path) -> str:
    return git(["symbolic-ref", "--short", "HEAD"], cwd).strip()


def ref_exists(cwd: Path, ref: str) -> bool:
    return git(["rev-parse", "--verify", "--quiet", ref], cwd, check=False).strip() != ""


def tree_hash(cwd: Path, ref: str, path: str) -> str:
    out = git(["rev-parse", f"{ref}:{path}"], cwd, check=False).strip()
    return out or ""


def latest_tag(cwd: Path, pattern: str = "harness-v*", merged: str | None = None) -> str | None:
    """Newest tag matching `pattern` (version order); only tags contained in `merged` if given."""
    extra = ["--merged", merged] if merged else []
    out = git(["tag", "--list", pattern, "--sort=-v:refname", *extra], cwd).split()
    return out[0] if out else None


def count_commits(cwd: Path, rev_range: str, paths: list[str] | None = None) -> int:
    return int(git(["rev-list", "--count", rev_range, "--", *(paths or [])], cwd).strip() or 0)


def add_worktree(repo: Path, path: Path, branch: str, base_commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "-b", branch, str(path), base_commit], repo)


# ---- working-tree state ----------------------------------------------------------------------

@dataclass
class Change:
    status: str  # two-letter porcelain code, e.g. " M", "??", " D"
    path: str


def changes(cwd: Path, pathspecs: list[str] | None = None) -> list[Change]:
    args = ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    if pathspecs:
        args += ["--", *pathspecs]
    raw = git(args, cwd)
    out: list[Change] = []
    parts = raw.split("\0")
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if not entry:
            continue
        status, path = entry[:2], entry[3:]
        if status[0] in "RC":  # rename/copy: next field is the original path
            i += 1
        out.append(Change(status, path))
    return out


def restore(cwd: Path, paths: list[str]) -> None:
    """Return paths to their HEAD state: tracked files are checked out, untracked are deleted."""
    if not paths:
        return
    tracked = set(git(["ls-files", "-z", "--", *paths], cwd).split("\0")) - {""}
    in_head = set(git(["ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *paths], cwd)
                  .split("\0")) - {""}
    for path in paths:
        if path in in_head:
            git(["checkout", "HEAD", "--", path], cwd)
        else:
            if path in tracked:
                git(["rm", "--cached", "-q", "--", path], cwd, check=False)
            full = cwd / path
            if full.is_file() or full.is_symlink():
                full.unlink()


def numstat(cwd: Path, paths: list[str]) -> int:
    """Changed lines (added + deleted) for paths relative to HEAD, counting untracked files."""
    if not paths:
        return 0
    total = 0
    out = git(["diff", "--numstat", "HEAD", "--", *paths], cwd, check=False)
    for line in out.splitlines():
        added, deleted, _ = (line.split("\t") + ["", ""])[:3]
        total += int(added) if added.isdigit() else 0
        total += int(deleted) if deleted.isdigit() else 0
    untracked = git(["ls-files", "--others", "--exclude-standard", "-z", "--", *paths], cwd)
    for path in filter(None, untracked.split("\0")):
        try:
            total += (cwd / path).read_text(errors="replace").count("\n") + 1
        except (OSError, UnicodeDecodeError):
            total += 1
    return total


def commit(cwd: Path, subject: str, trailers: dict, paths: list[str], body: str = "") -> str | None:
    """Stage `paths` (additions, edits, deletions) and commit. Returns the sha, or None."""
    if not paths:
        return None
    git(["add", "-A", "--", *paths], cwd)
    staged = git(["diff", "--cached", "--name-only", "--", *paths], cwd).strip()
    if not staged:
        return None
    message = subject.strip() + "\n\n"
    if body.strip():
        message += body.strip() + "\n\n"
    message += "\n".join(f"{k}: {v}" for k, v in trailers.items()) + "\n"
    git(["commit", "-q", "-F", "-", "--", *paths], cwd, env=bot_env(), input_text=message)
    return rev_parse(cwd)


def revert(cwd: Path, sha: str, trailers: dict) -> str:
    git(["revert", "--no-commit", sha], cwd, env=bot_env())
    subject = f"Revert {sha[:10]}"
    message = subject + "\n\n" + "\n".join(f"{k}: {v}" for k, v in trailers.items()) + "\n"
    git(["commit", "-q", "-F", "-"], cwd, env=bot_env(), input_text=message)
    return rev_parse(cwd)


def tag(cwd: Path, name: str, sha: str) -> None:
    git(["tag", "-f", name, sha], cwd)


def reset_hard(cwd: Path) -> None:
    git(["reset", "-q", "--hard", "HEAD"], cwd)
    git(["clean", "-q", "-fd", "--", "workspace", "harness", "records"], cwd)


def archive_extract(repo: Path, ref: str, path: str, dest: Path) -> None:
    """Extract `ref:path` into dest (dest/<path>), without touching any worktree."""
    import io
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["git", "archive", "--format=tar", ref, "--", path], cwd=str(repo),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise GitError(f"git archive {ref} {path}: {proc.stderr.decode().strip()}")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest, filter="data")


def show(cwd: Path, ref: str, path: str) -> str | None:
    proc = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=str(cwd), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout if proc.returncode == 0 else None


def log_commits(cwd: Path, rev_range: str) -> list[dict]:
    """Commits in rev_range (oldest first) with their parsed trailers."""
    raw = git(["log", "--reverse", "--format=%H%x1f%s%x1f%B%x1e", rev_range], cwd)
    out = []
    for entry in raw.split("\x1e"):
        entry = entry.strip("\n")
        if not entry:
            continue
        sha, subject, body = (entry.split("\x1f") + ["", ""])[:3]
        trailers = {}
        for line in body.splitlines():
            if line.startswith("Rsib-") and ": " in line:
                key, value = line.split(": ", 1)
                trailers[key] = value.strip()
        out.append({"sha": sha, "subject": subject, "trailers": trailers})
    return out


def changed_files(cwd: Path, base: str, ref: str) -> list[str]:
    out = git(["diff", "--name-only", f"{base}..{ref}"], cwd)
    return [p for p in out.splitlines() if p]


def push(cwd: Path, remote: str, branch: str) -> None:
    git(["push", "-q", remote, f"refs/heads/{branch}:refs/heads/{branch}"], cwd)
    tags = git(["tag", "--list", f"{branch}/*"], cwd).split()
    if tags:
        git(["push", "-q", "-f", remote, *[f"refs/tags/{t}:refs/tags/{t}" for t in tags]], cwd)
