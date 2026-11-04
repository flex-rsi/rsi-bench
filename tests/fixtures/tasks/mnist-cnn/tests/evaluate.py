"""Evaluator: train + predict in a child process on the GPU, score accuracy here."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def solution_cmd(python, io_dir):
    """The solution's process. When the orchestrator gives a sandbox launcher ($RSIB_SANDBOX),
    the solution runs through it and sees its own code, the run cache and io_dir, never labels."""
    child = Path(io_dir) / "_child_predict.py"
    shutil.copy(Path(__file__).with_name("child_predict.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


split_dir = Path(os.environ["RSIB_DATA_DIR"])
train_dir = Path(os.environ["RSIB_TRAIN_DIR"])
workspace = Path(os.environ["RSIB_WORKSPACE"])
out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
python = os.environ.get("RSIB_PYTHON", sys.executable)
TIME_LIMIT_S = 900

with tempfile.TemporaryDirectory(prefix="mnist-") as tmp:
    io = Path(tmp)
    shutil.copy(train_dir / "train_images.npy", io / "train_images.npy")
    shutil.copy(train_dir / "train_labels.npy", io / "train_labels.npy")
    shutil.copy(split_dir / "images.npy", io / "query_images.npy")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("RSIB_") or k in ("RSIB_PYTHON",)}
    env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    try:
        proc = subprocess.run(solution_cmd(python, io),
                              env=env, cwd=tmp, capture_output=True, text=True,
                              timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        sys.exit(f"training + prediction exceeded {TIME_LIMIT_S}s")
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        sys.exit("model failed")
    preds = np.load(io / "pred.npy")

labels = np.load(split_dir / "labels.npy")
if preds.shape != labels.shape:
    sys.exit(f"expected {labels.shape[0]} predictions, got shape {preds.shape}")
accuracy = float((preds == labels).mean())
reward = {"accuracy": accuracy, "errors": int((preds != labels).sum()), "n": int(labels.shape[0])}
(out_dir / "reward.json").write_text(json.dumps(reward))
print(json.dumps(reward))
