"""CPU limits for phases.

Phases run on the orchestrator's machine, often a shared login node, and anything an agent runs
from its shell runs there too. The `[phase]` table of the site profile caps what that can take:

    [phase]
    cpus = 8      # pin each phase, and every process it starts, to this many cores
    nice = 10     # lower the scheduling priority of phase processes
    threads = 8   # size of the usual thread pools (OpenMP, MKL, OpenBLAS, numexpr, Rayon)

Heavy work belongs in experiments, which the backend runs as jobs (a CPU-only job goes to the
SLURM `cpu_partition` when the site names one). The limits are applied with `nice` and `taskset`
in front of the phase command, so they hold for every child and need no privileges.
"""

from __future__ import annotations

import os
import shutil

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS",
               "RAYON_NUM_THREADS")


class PhaseLimits:
    def __init__(self, site: dict | None):
        cfg = dict((site or {}).get("phase") or {})
        self.nice = int(cfg.get("nice", 0) or 0)
        self.threads = int(cfg.get("threads", 0) or 0)
        n = int(cfg.get("cpus", 0) or 0)
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        # the last cores of the machine: the first ones tend to take system and interactive load
        self.cores = available[-n:] if 0 < n < len(available) else []

    def env(self) -> dict[str, str]:
        return {var: str(self.threads) for var in THREAD_VARS} if self.threads else {}

    def prefix(self) -> list[str]:
        """Command prefix that applies the limits to the phase and all its children."""
        out: list[str] = []
        if self.nice and shutil.which("nice"):
            out += ["nice", "-n", str(self.nice)]
        if self.cores and shutil.which("taskset"):
            out += ["taskset", "-c", ",".join(map(str, self.cores))]
        return out

    def describe(self) -> dict:
        """What the phase context tells the harness."""
        return {"phase_cpus": len(self.cores) or None, "phase_threads": self.threads or None}
