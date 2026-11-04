import os
import sys

# Inside apptainer --nv the driver libraries (libcuda, nvml, ...) are reachable only through
# LD_LIBRARY_PATH=/.singularity.d/libs. EmbodiedSWE's boot check starts its child interpreter with an
# environment of its own (PATH=/usr/bin:/bin, no LD_LIBRARY_PATH), and torch then finds "no NVIDIA driver".
# Re-exec once with the path added.
_L = "/.singularity.d/libs"
if os.path.isdir(_L) and _L not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in (_L, os.environ.get("LD_LIBRARY_PATH")) if p)
    os.execv(sys.executable, sys.orig_argv)

# Warp's precompiled CUDA headers crash NVRTC (SIGBUS) on this cluster; compile without them.
try:
    import warp.config
    warp.config.use_precompiled_headers = False
except Exception:
    pass
