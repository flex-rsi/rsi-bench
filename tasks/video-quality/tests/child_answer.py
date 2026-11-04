"""Runs the submitted solver on every item of one split, several at a time. It sees the videos,
their frames, the items and the gateway of the model under test; never the labels."""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import solver  # noqa: E402

items = [json.loads(line) for line in (io / "items.jsonl").read_text().splitlines() if line]
workers = max(1, min(int(getattr(solver, "WORKERS", 8)), 16))  # the solver may set WORKERS


def as_score(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def as_choices(value) -> dict:
    """{"alignment": "A" or "B", ...}: the first letter of each criterion's answer."""
    out = {}
    for c in ("alignment", "artifacts", "preference"):
        letter = str((value or {}).get(c) or "").strip().upper()[:1]
        out[c] = letter if letter in ("A", "B") else None
    return out


def run(item: dict):
    t0 = time.time()
    try:
        got = as_score(solver.rate(item)) if item["kind"] == "rate" else as_choices(solver.compare(item))
    except Exception:  # one failed item scores as missing; the others keep their answers
        traceback.print_exc()
        got = None
    print(f"{item['id']}: {got} {time.time() - t0:.1f}s", flush=True)
    return item["id"], got


with ThreadPoolExecutor(workers) as pool:
    predictions = dict(pool.map(run, items))
(io / "predictions.json").write_text(json.dumps(predictions))
