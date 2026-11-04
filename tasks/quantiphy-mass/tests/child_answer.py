"""Runs the submitted solver on every question of one split, several at a time. It sees the
photos, the questions and the backbone proxy; never the answers."""

import json
import math
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
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def number(reply) -> float | None:
    """A float as is; otherwise the first number in the reply's text ("about 3.2 kg" -> 3.2)."""
    if isinstance(reply, (int, float)) and not isinstance(reply, bool):
        value = float(reply)
    else:
        m = NUMBER.search(str(reply or "").replace(",", ""))
        value = float(m.group(0)) if m else None
    return value if value is not None and math.isfinite(value) else None


def run(item: dict) -> tuple[str, float | None]:
    t0 = time.time()
    try:
        got = number(solver.answer(item))
    except Exception:  # one failed question scores 0; the others keep their answers
        traceback.print_exc()
        got = None
    print(f"{item['id']}: {got} {time.time() - t0:.1f}s", flush=True)
    return item["id"], got


with ThreadPoolExecutor(workers) as pool:
    predictions = dict(pool.map(run, items))
(io / "predictions.json").write_text(json.dumps(predictions))
