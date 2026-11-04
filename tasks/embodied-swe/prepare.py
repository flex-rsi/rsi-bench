"""Build the splits of embodied-swe from an EmbodiedSWE checkout and its Isaac Lab venv.

    EMBODIEDSWE_ROOT=/path/to/EmbodiedSWE rsib task prepare embodied-swe

$EMBODIEDSWE_ROOT is a checkout of github.com/EmbodiedSWE/EmbodiedSWE at the commit of
embodiedswe.json (branch isaacsim-6: Isaac Sim 6.2 / Isaac Lab 3.0) with its <root>/.venv. On hosts whose
glibc is too old for Isaac Sim 6.2, the interpreter is a wrapper that runs it in the official Isaac Sim 6.2 image
(embodiedswe.json "runtime", examples/embodied-swe/esw-python), so prepare.py runs on a GPU node. Needs network (asset download). Needs an RTX-class GPU: the boot check builds the scene in Isaac Sim.

The bench tree is EmbodiedSWE's own: eval/scripts/build_env.py under the `no_tools` condition
(minimal robobench tree for the stage, graders removed, control mode frozen, boot-validated at
seed 0, describe() harvested). The boot fetches the binary assets the preset needs from Hugging
Face into the tree itself (robobench resolves assets next to the package it is imported from),
so a rollout later runs offline from the tree alone.

    <out>/graders/     the suite's grader package: the privileged scorer, beside the splits,
                       outside every split, so no phase sees it
    <out>/dev/bench/   the agent's robobench tree (on PYTHONPATH), task.md, rules and skills
    <out>/<split>/     cases.json (the rollouts of one evaluation), sim.json (interpreter, preset)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESW_PYTHON = json.loads((HERE / "embodiedswe.json").read_text())["runtime"]["python"]


def git_head(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def build_bench(root: Path, cfg: dict, work: Path) -> Path:
    """EmbodiedSWE's build_env.py: one stage, the cfg's condition. Returns the stage dir."""
    if work.exists():
        shutil.rmtree(work)
    s = cfg["stage"]
    subprocess.run([ESW_PYTHON, str(root / "eval/scripts/build_env.py"),
                    "--name", "embodied-swe", "--stage", f"{s['suite']}.{s['scene']}:{s['robot']}:"
                    f"{s['controller']}", "--config", cfg["condition"], "--seed", "0",
                    "--out", str(work)], check=True, cwd=root,
                   env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "ESW_BIND_ALL_PROJ": "1"})
    stages = sorted((work / "embodied-swe" / "stages").iterdir())
    assert len(stages) == 1, stages
    return stages[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.environ.get("EMBODIEDSWE_ROOT")
    if not root:
        raise SystemExit("set EMBODIEDSWE_ROOT to the EmbodiedSWE checkout (see the docstring)")
    root = Path(root).resolve()
    cfg = json.loads((HERE / "embodiedswe.json").read_text())
    if git_head(root) != cfg["commit"]:
        raise SystemExit(f"{root} is at {git_head(root)}, the task pins {cfg['commit']}")
    if not (root / ".venv" / "bin" / "python").exists():
        raise SystemExit(f"missing {root}/.venv: run setup_embodiedswe.sh first")

    out = Path(args.out).resolve()
    stage = build_bench(root, cfg, out / "_build")
    s = cfg["stage"]
    grader_src = root / "robobench" / "suites" / s["suite"] / "grader"
    if (stage / "bench" / "robobench" / "suites" / s["suite"] / "grader").exists():
        raise SystemExit("LEAK: the built bench tree carries the grader")
    if (out / "graders").exists():
        shutil.rmtree(out / "graders")
    shutil.copytree(grader_src, out / "graders", ignore=shutil.ignore_patterns("__pycache__"))

    dev = out / "dev"
    if dev.exists():
        shutil.rmtree(dev)
    dev.mkdir(parents=True)
    shutil.move(str(stage / "bench"), dev / "bench")
    # robobench's asset check takes a lock in bench/.cache, which rollouts and experiments see
    # read-only; every asset is already in the tree, so the lock goes to each process's own /tmp
    shutil.rmtree(dev / "bench" / ".cache", ignore_errors=True)
    (dev / "bench" / ".cache").symlink_to("/tmp")
    # the agent-facing task text: EmbodiedSWE's describe() harvest, as their task.md has it
    preset = json.loads((out / "_build" / "embodied-swe" / "resolved.json").read_text())[
        "stages"][0]["preset"]
    (dev / "task.md").write_text(
        "# The task\n\n" + (stage / "describe.md").read_text().strip() + "\n\n"
        "## Building the environment\n\n```python\nimport robobench; robobench.discover()\n"
        "from robobench.core.registries import ENVS\n"
        f"env = ENVS.get('{preset}')().build(num_envs=1)\n```\n")
    for sub, names in (("rules", cfg["rules"]), ("skills", cfg["skills"])):
        (dev / sub).mkdir()
        for name in names:
            shutil.copy2(root / "eval" / "prompts" / sub / f"{name}.md", dev / sub / f"{name}.md")
    shutil.copy2(out / "_build" / "embodied-swe" / "resolved.json", dev / "resolved.json")
    shutil.rmtree(out / "_build")

    sim = {"python": ESW_PYTHON, "preset": preset, "scene": s["scene"],
           "rollout_budget_s": cfg["rollout_budget_s"]}
    # dev: the build seed 0, which the agent also develops on. val and test: seeds drawn here and
    # kept only in their split directories, so no file of the worktree names them
    rng = random.SystemRandom()
    drawn = rng.sample(range(1, 2**31), cfg["splits"]["val"] + cfg["splits"]["test"])
    split_seeds = {"dev": list(range(cfg["splits"]["dev"])),
                   "val": drawn[:cfg["splits"]["val"]], "test": drawn[cfg["splits"]["val"]:]}
    for split in ("val", "test"):  # a re-prepare keeps the seeds a run may already be scored on
        if (out / split / "cases.json").exists():
            split_seeds[split] = [c["seed"] for c in
                                  json.loads((out / split / "cases.json").read_text())["cases"]]
    for split, seeds in split_seeds.items():
        cases = [{"id": f"seed{k}", "seed": k, "num_envs": cfg["num_envs"]} for k in seeds]
        (out / split).mkdir(exist_ok=True)
        (out / split / "cases.json").write_text(json.dumps({"cases": cases}, indent=2))
        (out / split / "sim.json").write_text(json.dumps(sim, indent=2))
        print(f"{split}: {len(cases)} rollouts x {cfg['num_envs']} envs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
