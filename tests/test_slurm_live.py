"""Live SLURM round trip; skipped unless RSIB_TEST_SLURM=1 and a site file is configured."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rsibench.compute import Job, load_site
from rsibench.compute.slurm import SlurmBackend
from rsibench.envs import scrubbed_env
from rsibench.paths import site_config_path

pytestmark = pytest.mark.slurm


@pytest.mark.skipif(os.environ.get("RSIB_TEST_SLURM") != "1", reason="set RSIB_TEST_SLURM=1")
def test_cpu_job_round_trip(tmp_path_factory):
    site = load_site(site_config_path())
    cfg = dict(site.get("slurm", {}))
    cfg.update({"partition": os.environ.get("RSIB_TEST_SLURM_PARTITION", "cpu"),
                "cpus_per_gpu": 2, "mem_per_gpu_gb": 4, "poll_s": 5})
    # The job writes to a directory the compute node must see (set RSIB_TEST_SLURM_DIR to a
    # shared filesystem path when /tmp is node-local).
    base = os.environ.get("RSIB_TEST_SLURM_DIR")
    work = Path(base) if base else tmp_path_factory.mktemp("slurm")
    work.mkdir(parents=True, exist_ok=True)
    metrics = work / "metrics.json"
    job = Job(name="rsib-live-test", cwd=work, gpus=0, timeout_s=120, log_path=work / "job.log",
              command="python3 -c \"import json,os; json.dump({'ok': 1}, open(os.environ['RSIB_METRICS_FILE'], 'w'))\"",
              env=scrubbed_env({"RSIB_METRICS_FILE": str(metrics)}))
    result = SlurmBackend(cfg).run(job)
    assert result.status == "ok", result
    assert metrics.read_text().strip() == '{"ok": 1}'
