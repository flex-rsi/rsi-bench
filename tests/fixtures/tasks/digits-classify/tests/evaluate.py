"""Evaluator: run the classifier on the split's images in a child process, score accuracy here."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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

with tempfile.TemporaryDirectory(prefix="digits-") as tmp:
    io = Path(tmp)
    shutil.copy(train_dir / "train_inputs.json", io / "train_inputs.json")
    shutil.copy(train_dir / "train_labels.json", io / "train_labels.json")
    shutil.copy(split_dir / "inputs.json", io / "query_inputs.json")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "RSIB_IO_DIR": tmp,
           "RSIB_SOLVER_DIR": str(workspace), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "4"}
    proc = subprocess.run(solution_cmd(python, io), env=env,
                          cwd=tmp, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        sys.exit("classifier failed")
    preds = json.loads((io / "pred.json").read_text())

labels = json.loads((split_dir / "labels.json").read_text())["y"]
if len(preds) != len(labels):
    sys.exit(f"expected {len(labels)} predictions, got {len(preds)}")
accuracy = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
reward = {"accuracy": accuracy, "errors": sum(int(p != y) for p, y in zip(preds, labels)),
          "n": len(labels)}
(out_dir / "reward.json").write_text(json.dumps(reward))
print(json.dumps(reward))
