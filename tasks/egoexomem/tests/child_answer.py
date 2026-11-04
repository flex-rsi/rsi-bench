"""Runs the submitted solver on every question of one split, several at a time. It sees the
videos, their frames, the questions and the gateway of the model under test; never the answers."""

import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import solver  # noqa: E402

items = [json.loads(line) for line in (io / "questions.jsonl").read_text().splitlines() if line]
workers = max(1, min(int(getattr(solver, "WORKERS", 8)), 16))  # the solver may set WORKERS


def letter(reply, n_options: int) -> str | None:
    """The first stand-alone option letter in the solver's reply ("B", "(B)", "B) red" -> "B")."""
    valid = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]
    return next((m.group(1) for m in re.finditer(r"\b([A-Z])\b", str(reply or ""))
                 if m.group(1) in valid), None)


def run(item: dict) -> tuple[str, str | None]:
    t0 = time.time()
    try:
        got = letter(solver.answer(item), len(item["options"]))
    except Exception:  # one failed question scores 0; the others keep their answers
        traceback.print_exc()
        got = None
    print(f"{item['id']}: {got} {time.time() - t0:.1f}s", flush=True)
    return item["id"], got


with ThreadPoolExecutor(workers) as pool:
    predictions = dict(pool.map(run, items))
(io / "predictions.json").write_text(json.dumps(predictions))
