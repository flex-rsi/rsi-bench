"""Runs the submitted reconstruct() on every photo of one split, one photo at a time. It sees only
the inputs (photo.png); each call writes scene.blend into its own output directory."""

import os
import sys
import time
import traceback
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import reconstruct  # noqa: E402

for scene in sorted(p.name for p in (io / "inputs").iterdir() if p.is_dir()):
    out = io / "out" / scene
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        reconstruct.reconstruct(str(io / "inputs" / scene), str(out))
        status = "ok"
    except Exception:  # this scene stays empty; the others keep their results
        traceback.print_exc()
        status = "error"
    print(f"{scene}: {status} {time.time() - t0:.1f}s", flush=True)
