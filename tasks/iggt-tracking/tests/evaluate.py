"""Evaluator: run the tracker on the split's frames in a child process, score here against GT."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scorer import score_clip  # noqa: E402



def solution_cmd(python, io_dir):
    """The tracker's process. When the orchestrator gives a sandbox launcher ($RSIB_SANDBOX),
    the tracker runs through it and sees its own code, the run cache and io_dir, never labels."""
    child = Path(io_dir) / "_child_predict.py"
    shutil.copy(Path(__file__).with_name("child_predict.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


split_dir = Path(os.environ["RSIB_DATA_DIR"])
workspace = Path(os.environ["RSIB_WORKSPACE"])
out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
python = os.environ.get("RSIB_PYTHON", sys.executable)
runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE when sandboxed)
if runtime.exists() and runtime.read_text().strip():
    python = runtime.read_text().strip().splitlines()[0].strip()
TIME_LIMIT_S = 2700

with tempfile.TemporaryDirectory(prefix="iggt-", dir=out_dir) as tmp:
    io = Path(tmp)
    shutil.copytree(split_dir / "inputs", io / "inputs")
    env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
    env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    try:
        proc = subprocess.run(solution_cmd(python, io), env=env,
                              cwd=tmp, capture_output=True, text=True, timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        sys.exit(f"tracker exceeded {TIME_LIMIT_S}s on this split")
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-6000:])
        sys.exit("tracker failed")
    scores = {}
    for lab in sorted((split_dir / "labels").iterdir()):
        gt = np.load(lab / "gt.npz")["inst_ids"]
        pred_path = io / "pred" / f"{lab.name}.npy"
        if not pred_path.exists():
            sys.exit(f"no prediction for {lab.name}")
        pred = np.load(pred_path)
        if pred.shape != gt.shape:
            sys.exit(f"{lab.name}: prediction shape {pred.shape} != {gt.shape}")
        r = score_clip(pred, gt)
        if r is not None:
            scores[lab.name] = r

keys = ["t_miou", "t_sr", "J", "F", "JF"]
reward = {k: float(np.mean([s[k] for s in scores.values()])) for k in keys}
reward["clips"] = len(scores)
(out_dir / "reward.json").write_text(json.dumps(reward))
(out_dir / "per_clip.json").write_text(json.dumps(scores, indent=1))
print(json.dumps(reward))
