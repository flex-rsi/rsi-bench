"""Runs the submitted editor in a process that never sees the labels.

Reads $RSIB_IO_DIR/inputs.jsonl (id, image, instruction, lang), starts one worker per group of
GPUS_PER_WORKER visible GPUs (a module-level integer in editor.py, default 1), each importing
editor.py from $RSIB_SOLVER_DIR, and writes outputs/<id>.png. A sample that raises is logged to
outputs/<id>.err and skipped (it scores 0).
"""

import ast
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path


def worker(gpu, items, io, solver):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.path.insert(0, solver)
    out = io / "outputs"
    try:
        import editor
        ed = editor.Editor("cuda" if gpu is not None else "cpu")
    except Exception:
        (out / f"_worker{gpu}.err").write_text(traceback.format_exc())
        return
    for it in items:
        t = time.time()
        try:
            ed.edit(str(io / it["image"]), it["instruction"], str(out / f"{it['id']}.png"),
                    {"lang": it["lang"], "id": it["id"]})
        except Exception:
            (out / f"{it['id']}.err").write_text(traceback.format_exc())
        print(f"{it['id']} {time.time() - t:.1f}s", flush=True)


def gpus_per_worker(solver, n_gpus):
    """GPUS_PER_WORKER from editor.py, read without importing it (no CUDA in the parent)."""
    try:
        tree = ast.parse((Path(solver) / "editor.py").read_text())
    except (OSError, SyntaxError):
        return 1
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "GPUS_PER_WORKER" for t in node.targets):
            try:
                k = int(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                return 1
            return max(1, min(k, n_gpus))
    return 1


def main():
    io = Path(os.environ["RSIB_IO_DIR"])
    solver = os.environ["RSIB_SOLVER_DIR"]
    items = [json.loads(l) for l in (io / "inputs.jsonl").read_text().splitlines() if l.strip()]
    (io / "outputs").mkdir(exist_ok=True)
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    k = gpus_per_worker(solver, len(gpus)) if gpus else 1
    gpus = [",".join(gpus[i:i + k]) for i in range(0, len(gpus) - k + 1, k)] or [None]
    shards = [items[i::len(gpus)] for i in range(len(gpus))]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(g, s, io, solver)) for g, s in zip(gpus, shards) if s]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
