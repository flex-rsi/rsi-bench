"""Where things live on disk.

Repository (main branch)   -> framework code, harness template, public task packages
Runs root                  -> one git worktree per run branch, plus the private store (task
                              copies, val and test evaluation outputs)
Artifact root              -> large outputs (checkpoints, media, transcripts); never in git
Scratch root               -> one directory per run for agent-made state (HOME, download cache,
                              tmp); moved to the archive root when the run ends
Archive root               -> scratch and large artifacts of finished runs, out of every agent's
                              reach, deleted after the retention period
Data root                  -> task data splits (dev / val / test); val and test stay private

Every location can be overridden with an environment variable so the same checkout works on a
laptop, a workstation and a cluster with a shared filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("RSIB_REPO")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "protocol").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate the agent-rsi-benchmarking checkout; set RSIB_REPO to its path."
    )


def runs_root() -> Path:
    env = os.environ.get("RSIB_RUNS_ROOT")
    if env:
        return Path(env).resolve()
    repo = repo_root()
    return repo.parent / f"{repo.name}-runs"


def data_root() -> Path:
    env = os.environ.get("RSIB_DATA_ROOT")
    if env:
        return Path(env).resolve()
    repo = repo_root()
    return repo.parent / f"{repo.name}-data"


def artifact_root() -> Path:
    env = os.environ.get("RSIB_ARTIFACT_ROOT")
    if env:
        return Path(env).resolve()
    return runs_root() / "_artifacts"


def scratch_root() -> Path:
    """Per-run scratch for everything agent code creates (HOME, downloaded weights and packages,
    tmp). Only the owning run can see its directory."""
    env = os.environ.get("RSIB_SCRATCH_ROOT")
    if env:
        return Path(env).resolve()
    return runs_root() / "_scratch"


def archive_root() -> Path:
    """Scratch and large artifacts of finished runs, kept for review and then deleted."""
    env = os.environ.get("RSIB_ARCHIVE_ROOT")
    if env:
        return Path(env).resolve()
    return runs_root() / "_archive"


def private_root() -> Path:
    """The private store: evaluator copies of task packages and val/test evaluation outputs,
    outside every worktree the agent can edit."""
    return runs_root() / "_private"


def site_config_path() -> Path | None:
    env = os.environ.get("RSIB_SITE")
    if env:
        return Path(env).expanduser().resolve()
    default = Path("~/.config/rsib/site.toml").expanduser()
    return default if default.exists() else None
