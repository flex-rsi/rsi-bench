from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    command: str              # run with `bash -c`
    cwd: Path
    env: dict                 # complete environment for the job (already scrubbed)
    gpus: int = 0
    timeout_s: float = 3600.0
    log_path: Path | None = None
    cpus: int | None = None
    mem_gb: float | None = None
    container: str | None = None   # optional image (slurm + pyxis/enroot)
    mounts: list | None = None     # container mounts for this job ("src:dst[:ro]"); None = site default
    # true once nobody waits for the job any more: if it has not started yet, it never does
    cancelled: Callable[[], bool] | None = None
    # told the job's queue wait while it lasts: (submitted, None) while it is pending, then
    # (submitted, started) once it started or left the queue (Budget.waiting)
    on_queue: Callable[[float, float | None], None] | None = None

    def wanted(self) -> bool:
        return self.cancelled is None or not self.cancelled()

    def report_queue(self, start: float, end: float | None = None) -> tuple[float, float | None]:
        if self.on_queue is not None:
            self.on_queue(start, end)
        return start, end


@dataclass
class JobResult:
    status: str               # ok | failed | timeout | rejected | cancelled | error
    exit_code: int | None
    wall_s: float
    gpu_hours: float
    job_id: str | None = None
    detail: str = ""
    extra: dict = field(default_factory=dict)
    queued: tuple | None = None   # (start, end) in time.monotonic(): the job waited in a queue

    @property
    def queue_s(self) -> float:
        return max(0.0, self.queued[1] - self.queued[0]) if self.queued else 0.0


class ComputeBackend(ABC):
    name = "base"

    @abstractmethod
    def run(self, job: Job) -> JobResult:
        ...

    def describe(self) -> dict:
        return {"backend": self.name}
