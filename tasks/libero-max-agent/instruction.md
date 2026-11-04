# LIBERO-MAX: agent-as-policy around frozen X-VLA

Improve `controller.py`, the robot policy evaluated on official LIBERO-MAX cases. The
primary metric is Dynamic success rate over every planned pair. Nothing is trained. The controller decides every action of a
Franka Panda in LIBERO's MuJoCo kitchen scenes, and it may use:

- the frozen pretrained **X-VLA** policy through `query_policy` (as the starter does);
- the **run's own model** (the model you are), called through the episode's gateway;
- **calibrated cameras and metric depth** for both views;
- any computation of its own: scripted motions, geometry, state machines, memory of the episode.

## The benchmark

LIBERO-MAX changes the world after execution has begun. Each case is a pair: a **Base** episode in the source configuration and a **Dynamic** episode that replays the same start and receives one online event during execution: target relocation, receptacle relocation, camera shift, sensor-noise onset, illumination switch, visual-theme switch, distractor burst or obstacle insertion. The primary metric is Dynamic success; Base success, the paired delta and repaired/regressed pairs are diagnostics.

Every split (dev, validation, test) uses the same three LIBERO tasks ("pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate", "pick up the bbq sauce and place it in the basket", "push the plate to the front of the stove") and the same eight event types, in
equal numbers per event type. Validation and test cases are other initial states of the same
tasks; they are kept private. `RSIB_DEV_DATA/cases.json` lists the dev cases.

## Controller interface

The workspace exports `class Controller`:

```python
class Controller:
    def reset(self):
        ...

    def step(self, observation, query_policy):
        # Return one np.float32 array with shape (7,).
        ...
```

Each episode (each Base or Dynamic arm) starts a fresh CPU worker, creates one controller and calls `reset()` once.
Python and NumPy start with seed 0. `step` is called once per control step until the task succeeds
or the native horizon ends.

`observation` is a dict:

- `images`: `"observation.images.image"` (agent view) and `"observation.images.image2"` (wrist
  view), each a float32 array `[1, 3, H, W]` in `[0, 1]`, as rendered (see Geometry for the
  orientation). The dots are part of the key strings.
- `depth`: `"agentview"` and `"wrist"`, float32 `[H, W]` metric depth (meters along the optical
  axis), pixel-aligned with the matching image.
- `cameras`: `"agentview"` and `"wrist"`, each `{"K", "camera_to_world", "width", "height"}`.
- `instruction`: the native task instruction.
- `step`: zero-based control step after the native ten-action warmup.
- `proprio`: `eef_pos` (3), `eef_rot6d` (6), and `gripper_qpos`, `joint_qpos` where present;
  finite float32 vectors.
- `last_action`: the seven-element action executed on the previous step, or `None` at step 0.

`query_policy(images=None)` returns X-VLA's absolute action chunk, float32 `[30, 7]`, for the
current step (seed `case.policy_seed + step`). With `None` it uses the current images; you may
pass transformed images as a dict with both image keys, same shapes, float32 in `[0, 1]`. One
query attempt per step. The controller decides what to execute; querying executes nothing.

Return a finite float32 action `(7,)`: absolute end-effector position (world frame, meters),
absolute axis-angle rotation (radians), gripper command (+1 close, -1 open, 0 keeps the current
target). `eef_rot6d` is the first two columns of the end-effector rotation matrix. Joint limits,
contact and the controller's finite tracking can keep a pose from being reached; compute a held
target once and compare later measured poses with it. `gripper_qpos` is the finger joints, not a
grasp flag. The starter queries X-VLA when its queue is empty and executes all 30 actions; its
score on validation is the baseline of the run.

## Geometry

`workspace/geometry.py` implements these conventions (verified on the simulator):

- Public image and depth arrays are the upright camera view **flipped vertically**: row 0 of an
  array is the bottom row of the upright view; columns are unchanged. `geometry.upright_rgb`
  gives the upright uint8 view, e.g. to show the model.
- `K` is the pinhole intrinsic matrix of the upright view; `camera_to_world` is the camera pose
  (4x4) with OpenCV axes (x right, y down, z forward). World +z is up.
- `geometry.project(points, camera)` gives public-array rows/cols and depth;
  `geometry.unproject(rows, cols, depth_map, camera)` gives world points.
- The agentview calibration is measured once at the start of the episode (a camera_shift event moves the camera but not this calibration; the depth map always comes from the actual camera). The wrist
  camera follows the arm and is measured every step.

## Calling the model

`workspace/model_client.py` wraps the gateway: `model_client.ask(system, content,
max_tokens=..., session=...)` posts `{system, messages, tools, max_tokens, session}` to
`$RSIB_GATEWAY_URL/v1/complete` and returns `{text, tool_calls, usage}`; content parts are
`text_part(...)` and `image_part(uint8_rgb)`. The model thinks before answering, so leave
`max_tokens` room for its reasoning (a reply cut off by `max_tokens` may have no answer).

- One episode (one arm) may make at most **16** model calls; the 17th gets HTTP 402.
- An episode still running after **20 minutes** of wall time is stopped and scored as the scene
  stands. Model calls are slow compared with control steps: budget them per episode.
- Model calls made during your dev evaluations count toward this run's budget; validation and
  test calls do not.
- **Paired replay.** Before the event, the Dynamic arm re-runs the Base arm's shared prefix: a fresh controller receives the recorded Base observations and X-VLA replies, and its model calls receive the Base arm's recorded replies in order. Each prefix request must equal the Base request (same messages, images and parameters), otherwise the call returns HTTP 409 and the evaluation fails. So make the prefix deterministic: no wall-clock time or randomness outside the seeded generators in requests or decisions. After the event, calls are live.

## Rules

- **Models.** The policy may use X-VLA through `query_policy` and the run's model through the
  gateway, and no other model: no detectors, segmenters, depth or pose estimators, other VLAs or
  online services, whether downloaded weights or a library that bundles them.
- **Everything else is yours to choose**: what the model is shown and asked, prompts, scripted
  motions, your own geometry and planning code, knowledge files in the workspace. What you learn
  on dev (geometry, calibration, what works) may go into the policy. Simulator state, case labels
  and the validation/test cases are not available, and your code must not look for them.
- The simulator, its physics, the scenes, perturbations, success checks and action semantics are
  fixed.

## Feedback and resources

Use `evaluate_dev` for the dev panel. Dev `feedback.json` gives per-case outcomes, actual
action/eef/query trajectories (`trajectory_paths`), frame sequences around queries
(`frame_sequence_paths`), command-tracking summaries and each episode's model-call statistics.
Everything the controller prints to stderr is saved per episode as `adapter-io/adapter.log`
next to its trajectory: log your decisions there. Validation reports aggregate metrics only;
the test panel runs once after the search.

Episodes of an evaluation run in parallel, so a dev evaluation of a model-calling policy takes
roughly (slowest episode) x (episodes / 3). Measure early, keep the workspace runnable, and keep
measured numbers, failure modes and lessons in your notes file.
