"""Evaluator of embodied-swe: grade the workspace's solution/solve.py like EmbodiedSWE does.

One rollout per seed of the split (cases.json), each a fresh process of the Isaac Lab
interpreter running tests/grade.py (EmbodiedSWE's eval/grader/grade.py, paths made
overridable): build the preset from the bench tree, reset it at the seed, wrap it in GradedEnv
(reset and set_states blocked), call solve(env), read the grader's verdict. The grader package
(the privileged scorer) lives in $RSIB_DATA_ROOT/embodied-swe/graders/, beside the splits, so
no phase sees it.

Each rollout runs in the Isaac Sim 6.2 image (embodiedswe.json "runtime") with --containall: it
sees the image, the venv, the workspace snapshot, the bench tree (with its assets), the grader
package and its own output directory, all but the last read-only; not the seeds of other splits,
the run's records or the agent's worktree. rsib's sandbox cannot host apptainer, so this container
is the rollout's sandbox. As in EmbodiedSWE's grading container, the grader is
readable from the solution's process: what protects the score is a fresh process per rollout
and a GradedEnv, plus the hack audit of accepted solutions (instruction.md), not secrecy.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["RSIB_DATA_DIR"])
BENCH = Path(os.environ["RSIB_TRAIN_DIR"]) / "bench"  # the agent's tree: dev holds the one copy
GRADERS = DATA_DIR.parent / "graders"
WORKSPACE = Path(os.environ["RSIB_WORKSPACE"]).resolve()
OUT = Path(os.environ["RSIB_OUTPUT_DIR"]).resolve()
SPLIT = os.environ.get("RSIB_SPLIT", "dev")


def log(msg: str) -> None:
    print(f"[embodied-swe] {msg}", flush=True)


def tail(path: Path, chars: int = 4000) -> str:
    try:
        return path.read_text(errors="replace")[-chars:]
    except OSError:
        return ""


def rollout_cmd(sim: dict, case: dict, io: Path) -> tuple[list[str], dict]:
    """The rollout's command and the extra environment of its container. sim["python"] runs the
    interpreter in the Isaac Sim image with --containall: the rollout sees the image, the venv, its io
    directory and the paths bound here (bench, grader and workspace read-only), nothing else of the host.
    rsib's own sandbox cannot host apptainer, so the container is the rollout's sandbox."""
    grade = io / "_grade.py"
    shutil.copy(HERE / "grade.py", grade)
    cmd = [sim["python"], "-u", str(grade), "--preset", sim["preset"], "--scene", sim["scene"],
           "--seed", str(case["seed"]), "--num-envs", str(case["num_envs"])]
    binds = [f"{BENCH}:ro", f"{GRADERS}:ro", f"{WORKSPACE}:ro", str(io)]
    return cmd, {"ESW_BIND": ",".join(binds), "ESW_HOME": str(io / "home")}


def run_case(sim: dict, case: dict, budget_s: float) -> dict:
    out = OUT / "rollouts" / case["id"]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="esw-") as tmp:
        io = Path(tmp)
        (io / "out").mkdir()
        env = {
            "PATH": str(Path(sim["python"]).parent) + ":/usr/bin:/bin",
            "HOME": str(io), "XDG_CACHE_HOME": str(io / "cache"), "TMPDIR": str(io),
            "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(BENCH),
            "ESW_BENCH": str(BENCH), "ESW_GRADERS": str(GRADERS),
            "ESW_SOLUTION": str(WORKSPACE / "solution" / "solve.py"), "ESW_OUT": str(io / "out"),
            "HF_HUB_OFFLINE": "1",  # every asset is in the bench tree (prepare.py booted it)
            "OMNI_KIT_ACCEPT_EULA": "YES", "OMNI_KIT_ALLOW_ROOT": "1",
            **{k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES",) if k in os.environ},
        }
        icd = "/etc/vulkan/icd.d/nvidia_icd.json"
        if Path(icd).exists():
            env.update(VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd)
        logf = out / "grade.log"
        t0 = time.monotonic()
        with logf.open("w") as lf:
            cmd, extra = rollout_cmd(sim, case, io)
            env.update(extra)
            proc = subprocess.Popen(cmd, env=env, cwd=io, stdout=lf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=budget_s)
            except subprocess.TimeoutExpired:
                # grade.py turns SIGTERM into a verdict from the state at that moment
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        verdict_file = io / "out" / "verdict.json"
        if not verdict_file.exists():
            raise RuntimeError(f"{case['id']}: no verdict (exit {proc.returncode}); log tail:\n"
                               + tail(logf))
        v = json.loads(verdict_file.read_text())
        shutil.copytree(io / "out", out, dirs_exist_ok=True)
    per = v.get("per_traj") or []
    return {"case": case["id"], "seed": case["seed"], "num_envs": case["num_envs"],
            "score": float(v.get("score") or 0.0),
            "successes": sum(bool(t.get("success")) for t in per),
            "timed_out": bool(v.get("timed_out")), "error": (v.get("error") or "")[-2000:],
            "solve_wall_s": v.get("solve_wall_s"), "solve_sim_steps": v.get("solve_sim_steps"),
            "wall_s": round(time.monotonic() - t0, 1)}


def main() -> int:
    sim = json.loads((DATA_DIR / "sim.json").read_text())
    cases = json.loads((DATA_DIR / "cases.json").read_text())["cases"]
    if not (WORKSPACE / "solution" / "solve.py").is_file():
        print("workspace has no solution/solve.py", file=sys.stderr)
        return 1
    log(f"{sim['preset']} {SPLIT}: {len(cases)} rollouts, {sim['rollout_budget_s']} s each")
    rows = []
    for case in cases:  # one GPU per evaluation: rollouts run one after another
        r = run_case(sim, case, float(sim["rollout_budget_s"]))
        log(f"{r['case']}: score {r['score']:.3f} success {r['successes']}/{r['num_envs']}"
            + (" timed out" if r["timed_out"] else "") + (" ERROR" if r["error"] else ""))
        rows.append(r)
    trajs = sum(r["num_envs"] for r in rows)
    reward = {
        "score": sum(r["score"] * r["num_envs"] for r in rows) / trajs,
        "success_rate": sum(r["successes"] for r in rows) / trajs,
        "trajectories": trajs,
        "crashed": sum(bool(r["error"]) for r in rows),
        "timed_out": sum(r["timed_out"] for r in rows),
        "metrics": {"per_rollout": {r["case"]: r for r in rows}},
        "artifacts": [{"path": f"rollouts/{r['case']}/grade.log", "role": "log", "note": r["case"]}
                      for r in rows],
    }
    assert all(math.isfinite(reward[k]) for k in ("score", "success_rate"))
    (OUT / "reward.json").write_text(json.dumps(reward, indent=2))
    log(f"score {reward['score']:.3f} success_rate {reward['success_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
