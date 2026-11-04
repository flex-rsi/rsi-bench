"""Install a frozen agent-as-policy panel (operator/agent_panels.py) into a task's data dir.

Panels are selected once, from official metadata, for all four agent tasks together; task
preparation only copies them, so separate preparation cannot place a physical state twice.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

PROFILE = "libero-agent-20261017"
SPLITS = ("dev", "val", "test")


def install(task_id: str) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    default = os.environ.get("LIBERO_RSI_FROZEN_PANELS")
    parser.add_argument("--frozen-root", type=Path, default=default, required=default is None)
    args = parser.parse_args()
    for split in SPLITS:
        origin = args.frozen_root / task_id / split
        panel = json.loads((origin / "cases.json").read_text())
        if panel.get("task_id") != task_id or panel.get("profile") != PROFILE or not panel["cases"]:
            raise ValueError(f"{task_id} {split}: not a frozen {PROFILE} panel")
    for split in SPLITS:
        destination = args.out / split
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("cases.json", "runtime.json"):
            shutil.copyfile(args.frozen_root / task_id / split / name, destination / name)
        print(f"{task_id} {split}: installed")
    return 0
