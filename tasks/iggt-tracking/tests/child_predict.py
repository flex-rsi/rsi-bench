"""Runs the submitted tracker on every clip of one split. It sees only the frames."""

import os
import sys
import time
from pathlib import Path

import numpy as np

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import tracker  # noqa: E402

out = io / "pred"
out.mkdir(exist_ok=True)
for clip in sorted(p.name for p in (io / "inputs").iterdir()):
    t0 = time.time()
    res = tracker.predict(str(io / "inputs" / clip / "frames"))
    idmap = res["idmap"] if isinstance(res, dict) else res
    np.save(out / f"{clip}.npy", np.asarray(idmap).astype(np.int32))
    print(f"{clip}: {time.time() - t0:.1f}s", flush=True)
