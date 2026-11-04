"""Evaluator: estimate the split's item weights in a child process that sees the photos, the
dimensions and a limited proxy of the fixed backbone, never the weights; score MRA here.

The child's only route to the backbone is backbone_proxy.Proxy (see its docstring). It never sees
the backbone server's own address."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backbone_proxy import Proxy  # noqa: E402

TIME_LIMIT_S = 2700            # the whole split
CALLS_PER_QUESTION = 4         # on average: a split of n items may make 4n backbone calls
MAX_IMAGES = 8                 # images of one call
MAX_PIXELS = 8 * 1024 * 1024   # all images of one call together
MAX_TOKENS = 16384             # completion tokens of one call
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BINS = {"light": (0, 0.5), "medium": (0.5, 5), "heavy": (5, math.inf)}  # kg


def solution_cmd(python, io_dir):
    """The solver's process. When the orchestrator gives a sandbox launcher ($RSIB_SANDBOX), the
    solver runs through it and sees its own code, the run cache and io_dir, never the labels."""
    child = Path(io_dir) / "_child_answer.py"
    shutil.copy(Path(__file__).with_name("child_answer.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def relative_accuracy(pred, label: float) -> float:
    """QuantiPhy's per-question MRA: the share of thresholds t with |pred - y| / |y| < 1 - t."""
    if pred is None or not math.isfinite(pred):
        return 0.0
    err = abs(pred - label) / abs(label)
    return sum(err < 1 - t for t in THRESHOLDS) / len(THRESHOLDS)


split_dir = Path(os.environ["RSIB_DATA_DIR"])
workspace = Path(os.environ["RSIB_WORKSPACE"])
out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
python = os.environ.get("RSIB_PYTHON", sys.executable)
runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE when sandboxed)
if runtime.exists() and runtime.read_text().strip():
    python = runtime.read_text().strip().splitlines()[0].strip()
if not os.environ.get("FIXED_VLM_URL"):
    sys.exit("FIXED_VLM_URL is not set: the fixed backbone server must be running")

questions = read_jsonl(split_dir / "questions.jsonl")
max_calls = CALLS_PER_QUESTION * len(questions)
proxy = Proxy(os.environ["FIXED_VLM_URL"], os.environ.get("FIXED_VLM_MODEL", "Qwen/Qwen3.8-27B"),
              max_calls, MAX_IMAGES, MAX_PIXELS, MAX_TOKENS)
with tempfile.TemporaryDirectory(prefix="qmass-", dir=out_dir) as tmp:
    io_dir = Path(tmp)
    shutil.copytree(split_dir / "images", io_dir / "images")
    items = [{**q, "images": [str(io_dir / p) for p in q["images"]]} for q in questions]
    (io_dir / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in items))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("RSIB_") and not k.startswith("FIXED_VLM_")}
    env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace),
                "VLM_URL": proxy.url, "VLM_KEY": proxy.key,
                "VLM_MODEL": proxy.model, "VLM_MAX_CALLS": str(max_calls),
                "VLM_MAX_IMAGES": str(MAX_IMAGES), "VLM_MAX_PIXELS": str(MAX_PIXELS),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    try:
        proc = subprocess.run(solution_cmd(python, io_dir), env=env, cwd=tmp, capture_output=True,
                              text=True, timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        sys.exit(f"the solver exceeded {TIME_LIMIT_S}s on this split")
    print(proc.stdout[-4000:])
    if proc.stderr.strip():
        print("[solver stderr]\n" + proc.stderr[-6000:])
    if proc.returncode != 0:
        sys.exit("solver failed")
    predictions = json.loads((io_dir / "predictions.json").read_text())

labels = {row["id"]: row["answer"] for row in read_jsonl(split_dir / "labels.jsonl")}
results = []
for q in questions:
    pred, y = predictions.get(q["id"]), labels[q["id"]]
    results.append({"id": q["id"], "product_type": q["product_type"], "answer": pred, "label": y,
                    "log_ratio": math.log(pred / y) if pred and pred > 0 else None,
                    "mra": relative_accuracy(pred, y)})


def mra(rows: list[dict]) -> float:
    return sum(r["mra"] for r in rows) / len(rows) if rows else 0.0


logs = sorted(abs(r["log_ratio"]) for r in results if r["log_ratio"] is not None)
reward = {"mra": mra(results),
          **{f"mra_{b}": mra([r for r in results if lo <= r["label"] < hi])
             for b, (lo, hi) in BINS.items()},
          "median_abs_log_ratio": logs[len(logs) // 2] if logs else None,
          "n": len(results), "answered": sum(r["answer"] is not None for r in results),
          "model_calls": proxy.calls, "refused_calls": proxy.refused,
          "prompt_tokens": proxy.prompt_tokens, "completion_tokens": proxy.completion_tokens,
          "artifacts": [{"path": "per_question.json", "role": "diagnosis",
                         "note": "answer, label and relative accuracy of every item"}]}
(out_dir / "per_question.json").write_text(json.dumps(results, indent=1))
(out_dir / "reward.json").write_text(json.dumps(reward))
print(json.dumps({k: v for k, v in reward.items() if k != "artifacts"}))
