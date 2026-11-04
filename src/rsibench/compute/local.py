"""Run jobs as local subprocesses, at most `max_parallel` at once (site [local] max_parallel,
default 4). GPUs come from a fixed pool (site [local] gpus or $RSIB_LOCAL_GPUS, e.g. "0,1,2,3"):
a job asking for n GPUs waits until n devices are free and runs on exactly those, so concurrent
jobs never share one; a job asking for more GPUs than the pool holds is rejected, and a job
cancelled while it waits never starts."""

from __future__ import annotations

import os
import subprocess
import threading
import time

from ..util import kill_group, wait_group
from .base import ComputeBackend, Job, JobResult


class LocalBackend(ComputeBackend):
    name = "local"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        pool = cfg.get("gpus", os.environ.get("RSIB_LOCAL_GPUS", ""))
        if isinstance(pool, str):
            pool = [g.strip() for g in pool.split(",") if g.strip()]
        self.gpu_pool = [str(g) for g in pool]
        self.max_parallel = max(1, int(cfg.get("max_parallel", 4)))
        self._free = set(range(len(self.gpu_pool)))  # pool indices of the devices not leased
        self._running = 0
        self._cond = threading.Condition()

    def describe(self) -> dict:
        return {"backend": self.name, "gpu_pool_size": len(self.gpu_pool),
                "max_parallel": self.max_parallel}

    def _lease(self, gpus: int) -> list[int]:
        """Wait for a free job slot and `gpus` free devices, and take them."""
        with self._cond:
            self._cond.wait_for(lambda: self._running < self.max_parallel
                                and len(self._free) >= gpus)
            self._running += 1
            lease = sorted(self._free)[:gpus]
            self._free.difference_update(lease)
            return lease

    def _release(self, lease: list[int]) -> None:
        with self._cond:
            self._running -= 1
            self._free.update(lease)
            self._cond.notify_all()

    def run(self, job: Job) -> JobResult:
        if job.gpus > len(self.gpu_pool):
            return JobResult("rejected", None, 0.0, 0.0,
                             detail=f"asked for {job.gpus} GPUs, local pool has {len(self.gpu_pool)}")
        lease = self._lease(job.gpus)
        try:
            if not job.wanted():
                return JobResult("cancelled", None, 0.0, 0.0, detail="cancelled before it started")
            return self._run(job, [self.gpu_pool[i] for i in lease])
        finally:
            self._release(lease)

    def _run(self, job: Job, devices: list[str]) -> JobResult:
        env = dict(job.env)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        log = open(job.log_path, "w") if job.log_path else subprocess.DEVNULL
        start = time.monotonic()  # GPU hours count the running time, not the wait for a lease
        try:
            proc = subprocess.Popen(["bash", "-c", job.command], cwd=str(job.cwd), env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=job.timeout_s)
                status = "ok" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                code, status = None, "timeout"
            finally:
                kill_group(proc.pid)  # a job ends with its processes, as on SLURM
                proc.wait()
            wall = time.monotonic() - start
            wait_group(proc.pid)  # devices go back once its processes are gone, or after 5 s
        except OSError as exc:
            return JobResult("error", None, time.monotonic() - start, 0.0, detail=str(exc))
        finally:
            if job.log_path:
                log.close()
        return JobResult(status, code, wall, len(devices) * wall / 3600.0, job_id=str(proc.pid))
