"""Compute backends: where experiments and evaluations actually run.

Every backend implements `run(job) -> JobResult`, which may be called from several threads at once
(the candidates of one dev evaluation call), and never starts a job whose `cancelled()` became true
while it waited. The harness never talks to a scheduler directly; it asks the gateway, which
charges the GPU budget and records the job.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .base import ComputeBackend, Job, JobResult
from .local import LocalBackend
from .slurm import SlurmBackend


def load_site(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def make_backend(name: str, site: dict | None = None) -> ComputeBackend:
    site = site or {}
    if name == "local":
        return LocalBackend(site.get("local", {}))
    if name == "slurm":
        return SlurmBackend(site.get("slurm", {}))
    raise ValueError(f"unknown compute backend {name!r}; use local or slurm")


__all__ = ["ComputeBackend", "Job", "JobResult", "LocalBackend", "SlurmBackend", "load_site",
           "make_backend"]
