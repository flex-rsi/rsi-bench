"""Evaluator: run the segmenter on the split's images in a child process, score mIoU here."""

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

with tempfile.TemporaryDirectory(prefix="toyseg-") as tmp:
    io = Path(tmp)
    shutil.copy(train_dir / "train_inputs.json", io / "train_inputs.json")
    shutil.copy(train_dir / "train_labels.json", io / "train_labels.json")
    shutil.copy(split_dir / "inputs.json", io / "query_inputs.json")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "RSIB_IO_DIR": tmp,
           "RSIB_SOLVER_DIR": str(workspace), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(solution_cmd(python, io), env=env,
                          cwd=tmp, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        sys.exit("segmenter failed")
    preds = json.loads((io / "pred.json").read_text())

truth = json.loads((split_dir / "labels.json").read_text())["masks"]
if len(preds) != len(truth):
    sys.exit(f"expected {len(truth)} masks, got {len(preds)}")
inter = {0: 0, 1: 0}
union = {0: 0, 1: 0}
for pm, tm in zip(preds, truth):
    if len(pm) != len(tm) or any(len(a) != len(b) for a, b in zip(pm, tm)):
        sys.exit("mask shape mismatch")
    for prow, trow in zip(pm, tm):
        for p, t in zip(prow, trow):
            p = 1 if p else 0
            for cls in (0, 1):
                a, b = p == cls, t == cls
                inter[cls] += a and b
                union[cls] += a or b
iou = {cls: inter[cls] / union[cls] if union[cls] else 1.0 for cls in (0, 1)}
reward = {"miou": (iou[0] + iou[1]) / 2, "iou_fg": iou[1], "iou_bg": iou[0], "n": len(truth)}
(out_dir / "reward.json").write_text(json.dumps(reward))
print(json.dumps(reward))
