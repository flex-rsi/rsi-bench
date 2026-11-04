"""Run jobs through SLURM (`sbatch` + `sacct`), optionally inside a pyxis/enroot container.

Settings come from the [slurm] and [slurm.containers] tables of the site profile, a local,
untracked TOML file (docs/operations.md, "Site profile"; examples/site.example.toml).

GPU-hours are charged from sacct ElapsedRaw x requested GPUs. A job cancelled before it starts is
not submitted, or is removed from the queue while still pending.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from .base import ComputeBackend, Job, JobResult

# Host variables that would break a container's own environment.
HOST_ONLY = {"PATH", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
             "CUDA_HOME", "CUDA_PATH", "PYTHONPATH"}

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def _hhmmss(seconds: float) -> str:
    seconds = max(60, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _write_private(path: Path, text: str) -> None:
    """Write `text` to `path`, readable by its owner only: a job script exports the job's gateway
    token."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        os.fchmod(f.fileno(), 0o600)  # open keeps the mode of a script an earlier job left here
        f.write(text)


class SlurmBackend(ComputeBackend):
    name = "slurm"

    def __init__(self, cfg: dict | None = None, runner=subprocess.run):
        self.cfg = dict(cfg or {})
        self._run = runner

    def describe(self) -> dict:
        safe = {k: v for k, v in self.cfg.items() if k not in ("account",)}
        return {"backend": self.name, **safe}

    def script(self, job: Job) -> str:
        c = self.cfg
        # A CPU-only job goes to `cpu_partition` when the site names one (with its own CPU,
        # memory and time defaults), so CPU-heavy work leaves the orchestrator's machine.
        cpu_job = not job.gpus and bool(c.get("cpu_partition"))
        time_s = job.timeout_s
        if cpu_job and c.get("cpu_max_minutes"):
            time_s = min(time_s, 60 * float(c["cpu_max_minutes"]))
        lines = ["#!/bin/bash", f"#SBATCH --job-name=rsib-{job.name[:40]}",
                 f"#SBATCH --time={_hhmmss(time_s)}", "#SBATCH --nodes=1",
                 "#SBATCH --ntasks=1"]
        if c.get("account"):
            lines.append(f"#SBATCH --account={c['account']}")
        partition = c.get("cpu_partition") if cpu_job else c.get("partition")
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if job.gpus:
            if c.get("gres"):
                lines.append(f"#SBATCH --gres=gpu:{job.gpus}")
            else:
                lines.append(f"#SBATCH {c.get('gpus_flag', '--gpus-per-node')}={job.gpus}")
        if cpu_job:
            cpus = job.cpus or c.get("cpu_cpus", 8)
            mem = job.mem_gb or c.get("cpu_mem_gb", 32)
        else:
            cpus = job.cpus or (c.get("cpus_per_gpu", 4) * max(1, job.gpus))
            mem = job.mem_gb or (c.get("mem_per_gpu_gb", 16) * max(1, job.gpus))
        lines.append(f"#SBATCH --cpus-per-task={cpus}")
        lines.append(f"#SBATCH --mem={int(mem)}G")
        if job.log_path:
            lines.append(f"#SBATCH --output={job.log_path}")
        for extra in c.get("extra_sbatch", []):
            lines.append(f"#SBATCH {extra}")
        # Start from an empty environment so orchestrator secrets (API keys) never reach the
        # job; the variables the job needs are exported explicitly below.
        lines.append("#SBATCH --export=NONE")
        lines.append("set -o pipefail")
        # A stale ~/.local site-packages must never shadow the job's environment.
        lines.append("export PYTHONNOUSERSITE=1")
        lines.append("export SLURM_EXPORT_ENV=ALL")
        lines.append(f"cd {shlex.quote(str(job.cwd))}")
        image = job.container or c.get("container_image")
        for key, value in sorted(job.env.items()):
            if image and key in HOST_ONLY:
                continue  # inside a container, the image's own PATH / python / CUDA apply
            if image and key == "RSIB_PYTHON":
                value = "python"  # the image's interpreter; the orchestrator's path is a host path
            lines.append(f"export {key}={shlex.quote(str(value))}")
        if image:
            own = job.mounts is not None  # a sandboxed job brings an exact mount list
            mounts = ",".join(job.mounts if own else c.get("container_mounts", []))
            srun = f"srun --container-image={shlex.quote(image)}"
            if mounts:
                srun += f" --container-mounts={shlex.quote(mounts)}"
            srun += f" --container-workdir={shlex.quote(str(job.cwd))}"
            flags = list(c.get("container_flags", []))
            if own and "--no-container-mount-home" not in flags:
                flags.append("--no-container-mount-home")
            for flag in flags:
                srun += f" {flag}"
            command = job.command
            if own and job.env.get("HOME"):
                # the container runtime resets HOME to the passwd entry; a sandboxed job keeps its own
                command = f"export HOME={shlex.quote(str(job.env['HOME']))}; {command}"
            lines.append(f"{srun} bash -c {shlex.quote(command)}")
        else:
            lines.append(f"bash -c {shlex.quote(job.command)}")
        return "\n".join(lines) + "\n"

    def _sacct(self, job_id: str) -> tuple[str, int | None, float]:
        proc = self._run(["sacct", "-j", job_id, "-n", "-P", "-X", "-o", "State,ExitCode,ElapsedRaw"],
                         capture_output=True, text=True)
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return "PENDING", None, 0.0
        state, exit_code, elapsed = (line[0].split("|") + ["", "", "0"])[:3]
        state = state.split()[0] if state else "PENDING"
        code = int(exit_code.split(":")[0]) if exit_code and exit_code.split(":")[0].isdigit() else None
        return state, code, float(elapsed or 0)

    def run(self, job: Job) -> JobResult:
        if not job.wanted():
            return JobResult("cancelled", None, 0.0, 0.0, detail="cancelled before it started")
        script_path = Path(job.log_path or Path(job.cwd) / f"{job.name}.log").with_suffix(".sbatch")
        _write_private(script_path, self.script(job))
        start = time.monotonic()
        proc = self._run(["sbatch", "--parsable", str(script_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            return JobResult("error", None, 0.0, 0.0, detail=f"sbatch failed: {proc.stderr.strip()}")
        job_id = proc.stdout.strip().split(";")[0]
        submitted, began = time.monotonic(), None
        job.report_queue(submitted)  # a phase waiting on the job is not timed while it is pending
        poll = float(self.cfg.get("poll_s", 15))
        # Queue time does not count against the experiment timeout; the job's own --time does.
        deadline = start + job.timeout_s + float(self.cfg.get("max_queue_s", 6 * 3600))
        state, code, elapsed = "PENDING", None, 0.0
        while True:
            state, code, elapsed = self._sacct(job_id)
            if state != "PENDING" and began is None:
                began = time.monotonic()
                job.report_queue(submitted, began)
            if state in TERMINAL:
                break
            if state == "PENDING" and not job.wanted():
                self._run(["scancel", job_id], capture_output=True, text=True)
                return JobResult("cancelled", None, time.monotonic() - start, 0.0, job_id=job_id,
                                 detail="cancelled before it started",
                                 queued=job.report_queue(submitted, time.monotonic()))
            if time.monotonic() > deadline:
                self._run(["scancel", job_id], capture_output=True, text=True)
                return JobResult("timeout", None, time.monotonic() - start,
                                 job.gpus * elapsed / 3600.0, job_id=job_id,
                                 detail="cancelled after queue + run deadline",
                                 queued=job.report_queue(submitted, began or time.monotonic()))
            time.sleep(poll)
        status = {"COMPLETED": "ok", "TIMEOUT": "timeout"}.get(state, "failed")
        waited = self._queue_seconds(job_id)
        queued = (submitted, submitted + waited) if waited is not None else (submitted, began)
        return JobResult(status, code, elapsed, job.gpus * elapsed / 3600.0, job_id=job_id,
                         detail=state, queued=job.report_queue(*queued))

    def _queue_seconds(self, job_id: str) -> float | None:
        """Start - Submit from sacct (exact); None when sacct cannot tell (the polling estimate,
        accurate to one poll interval, is used instead)."""
        from datetime import datetime
        try:
            proc = self._run(["sacct", "-j", job_id, "-n", "-P", "-X", "-o", "Submit,Start"],
                             capture_output=True, text=True)
            submit, begin = (proc.stdout or "").strip().splitlines()[0].split("|")[:2]
            return max(0.0, (datetime.fromisoformat(begin) - datetime.fromisoformat(submit))
                       .total_seconds())
        except Exception:  # noqa: BLE001 - unknown start, odd output, a test double
            return None
