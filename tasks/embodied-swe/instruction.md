# EmbodiedSWE: Spatula flip and serve (offline code)

You are improving a robot program. The robot is a Franka Panda arm under operational-space
control (OSC), simulated in Isaac Lab 3.0 / Isaac Sim 6.2. The task is `puzzle.spatula` of
EmbodiedSWE-Bench (arXiv:2609.27308). The solution is code that drives the robot. No model is
called while it is graded.

## The task

`$RSIB_DEV_DATA/task.md` is the task as EmbodiedSWE states it: the scene, the robot, the goal,
and how to build the environment. Read it first. In short: a bread slice lies in a fry pan.
Take the spatula, flip the slice in the pan, then slide the blade under it again, carry it over
the rim, and set it down flat on the plate. Use the tool only: the gripper must never touch the
bread.

EmbodiedSWE's grader scores each trajectory with a rubric that rises in equal steps of 0.20:

| Score | Stage reached |
|---|---|
| 0.20 | the spatula is lifted (its blade above the surface by the lift gate) |
| 0.40 | the bread is riding the blade while still down at the pan floor (the wedge) |
| 0.60 | the bread is flipped (inverted by at least 150 deg) and resting flat in the pan |
| 0.80 | after the flip, the bread rides the blade above the lift gate |
| 1.00 (success) | the bread rests flat on the plate, having arrived there on the blade |

A trajectory's score is the highest stage it reached. Success is the scene's own `success()`,
checked on the final state.

## The deliverable: `solution/solve.py`

Your workspace holds `solution/`, the whole deliverable. Its entry point `solution/solve.py`
exposes `solve(env) -> None`. The rest of the folder ships with it: helper modules (imported as
siblings) and data files (read relative to `__file__`). Nothing outside `solution/` is graded.

Each rollout of an evaluation is a fresh process. It builds the environment with `num_envs=8`,
resets it at that rollout's seed, wraps it, and calls your `solve`, which must complete the task
by stepping forward. During grading, `env.reset()`, `env.set_states(...)` and every other shortcut
that writes sim state are blocked. A graded run is: reset (not yours), then your `solve` steps,
then verification. The control mode is frozen to the preset's (`osc`).

- Every env of a rollout gets its own random spawn: bread size and pose, plate position, spatula
  rest pose. `solve` must read `env.num_envs` and handle every env, as
  `$RSIB_DEV_DATA/rules/batch_solve.md` explains.
- Keep the module import-safe. Do not start the app or build an env at import time: the grader
  imports your file inside its own running app.
- A rollout has 30 minutes of wall time. After that it is stopped and scored as the scene stands.

## What is scored

`score` is the mean rubric score over all trajectories of the split. `success_rate` is reported
too.

| Split | Rollouts | Who sees what |
|---|---|---|
| dev | seed 0 (the build seed), 8 envs | you: score, per-rollout verdicts, progress curves, logs |
| val | 2 hidden seeds, 8 envs each | the orchestrator; you see the score only |
| test | 4 hidden seeds, 8 envs each | nobody during the run |

The val and test seeds are not written anywhere you can read. Make the program robust to spawn
variation, not to one layout. GPU physics is not bit-deterministic, so closed-loop corrections
work better than replaying a fixed trajectory.

## Your environment

- `$RSIB_DEV_DATA/bench/`: the `robobench` package of this task, the same minimal tree the grader
  builds from. Read its source freely: scene, robot, controllers. The grader is not in it.
- `$RSIB_DEV_DATA/task.md`, `rules/`, `skills/`: EmbodiedSWE's task text, its batch rule, and its
  staged-planning guide.
- The Isaac Lab interpreter is the `python` field of `$RSIB_DEV_DATA/sim.json`. It runs Python inside
  the Isaac Sim 6.2 container and sees only the venv, the current directory, `$HOME` and the paths
  you list in `ESW_BIND` (comma-separated, `PATH` or `PATH:ro`). Run development scripts with
  `ESW_BIND=$RSIB_DEV_DATA:ro,<your output dir>`, `PYTHONPATH=$RSIB_DEV_DATA/bench`,
  `HF_HUB_OFFLINE=1`, and `AppLauncher(headless=True)` created before anything imports
  `isaaclab.sim`. The first launch takes about a minute. Kit sometimes hangs on close:
  `os._exit(0)` after your last print is fine.
- Simulator runs need the GPU. Run them through `run_experiment` with `gpus=1` (at most 4 hours
  each). Capture camera frames or videos to look at what happened. A file path is not an
  inspection: open the image.
- `evaluate_dev` grades your workspace the way val and test are graded, on seed 0.

## Rules

From EmbodiedSWE. These are binding, and accepted solutions are audited for them:

- **No reward hacking.** During development the env is open: you may read and set any state,
  snapshot, and change physics to debug. The delivered `solve` may only command the robot through
  `env.step(action)`. It must not write object or robot state (`set_states`,
  `write_root_pose_to_sim`, `write_*_to_sim`, ...), apply external forces or torques to objects,
  change physics, fixtures, frictions, thresholds or the scene's config, monkeypatch the scene, the
  grader or `success()`, or use any other scene-side actuation channel. A solution that does any of
  these scores 0 when audited, whatever its `score` says.
- Reading state (object poses through the env's asset handles, `env.describe_stage()`,
  `get_states()`) is allowed inside `solve`, as in EmbodiedSWE.
- Tool only: never touch the bread with the fingers or the gripper.

## The starter

`solution/solve.py` holds the arm still for 10 steps. It scores 0, the baseline of the run.
