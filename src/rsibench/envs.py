"""Environment scrubbing for every process the orchestrator starts on the agent's behalf.

Harness phases, experiments and evaluators get a whitelisted environment plus the RSIB_*
variables of their role: no API keys or provider credentials, and no val or test data path except
an evaluator's own split (RSIB_DATA_DIR). Sites can pass extra variable names through with
RSIB_PASS_ENV="OMP_NUM_THREADS,NCCL_DEBUG" (names only; values come from the orchestrator).
"""

from __future__ import annotations

import os

BASE_NAMES = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR",
    "TZ", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "VIRTUAL_ENV", "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV", "HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NCCL_SOCKET_IFNAME",
)


def scrubbed_env(extra: dict | None = None) -> dict:
    names = list(BASE_NAMES)
    names += [n.strip() for n in os.environ.get("RSIB_PASS_ENV", "").split(",") if n.strip()]
    env = {n: os.environ[n] for n in names if n in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env
