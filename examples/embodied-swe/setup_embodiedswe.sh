#!/usr/bin/env bash
# EmbodiedSWE at the commit tasks/embodied-swe pins (branch isaacsim-6: Isaac Sim 6.2 / Isaac Lab 3.0), its venv
# built inside the official Isaac Sim 6.2 image (hosts with glibc < 2.35, e.g. RHEL 9, cannot install Isaac Sim 6.2
# natively). Needs git, uv >= 0.8, apptainer, an NVIDIA driver >= r580 and network. CPU only; then, on a GPU node:
#   EMBODIEDSWE_ROOT=<dir> rsib task prepare embodied-swe
set -euo pipefail
dir=${1:?usage: setup_embodiedswe.sh <dir> <isaac-sim-6.2.0.sif>}
sif=${2:?usage: setup_embodiedswe.sh <dir> <isaac-sim-6.2.0.sif>  (apptainer pull docker://nvcr.io/nvidia/isaac-sim:6.2.0)}
cfg="$(dirname "$0")/../../tasks/embodied-swe/embodiedswe.json"
commit=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['commit'])" "$cfg")
[[ -d $dir/.git ]] || git clone https://github.com/EmbodiedSWE/EmbodiedSWE.git "$dir"
git -C "$dir" fetch -q origin "$commit" && git -C "$dir" checkout -q "$commit"
lab=$dir/externals/IsaacLab; labc=$(sed -n 's/^ISAACLAB_COMMIT="\(.*\)"/\1/p' "$dir/scripts/bootstrap_isaaclab_6.sh")
[[ -d $lab/.git ]] || git clone -q --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git "$lab"
git -C "$lab" fetch -q origin "$labc" 2>/dev/null || true; git -C "$lab" checkout -q "$labc"
# the image has no git: run the bootstrap's two install steps inside it
apptainer exec --containall --bind "$dir" --bind "$(command -v uv):/usr/local/bin/uv:ro" --env UV_PROJECT_ENVIRONMENT="$dir/.venv" \
  "$sif" bash -euc "cd $lab && uv sync --frozen --no-dev --inexact --python 3.12 --extra isaacsim && cd $dir && uv pip install --python .venv/bin/python -e ."
# container fixes for every interpreter run (driver libs for scrubbed child envs; Warp PCH off)
cp "$(dirname "$0")/sitecustomize.py" "$dir/.venv/lib/python3.12/site-packages/sitecustomize.py"
echo "ready: EMBODIEDSWE_ROOT=$dir rsib task prepare embodied-swe   (on a GPU node)"
