# Humanoid visual search in 360-degree scenes (H*Bench)

A humanoid robot stands in a 360-degree scene (an airport, a supermarket, a subway station, a
street) but sees only what is in front of its head camera. Given an instruction, it turns its head
to search, and ends by choosing a direction:

- **Object search (`hos`)**: "Find the TCL TV". Success: the chosen direction is on the object,
  within the target's yaw range widened to at least +-30 degrees and its pitch range widened to at
  least +-20 degrees around their centers. As in H*Bench's environment, when the yaw range crosses
  0/360 degrees only the yaw is checked.
- **Path search (`hps`)**: "You need to get out of the subway station, where'd you go?". Success:
  the chosen yaw points the way to go, within the target range widened to at least +-10 degrees
  (pitch does not matter).

Most targets start out of view, so the search has to reason about where things are in a scene it
has only partly seen. The scenes are real panoramas from transportation hubs, shops, streets and
public buildings in many countries; the test scenes are a separate in-the-wild set (street views
and generated scenes).

**The camera.** A view is a 100-degree field of view rendered at 960 x 720 pixels, with a green
cross at its center. An episode starts at a given yaw with pitch 0. The agent may take at most 10
actions: turn by a relative angle, or submit an absolute direction, which ends the episode. An
episode that has not submitted after 10 actions fails.

**Fewer steps score more.** A search is scored by its success weighted by the actions it took, so
looking all around before answering costs as much as it would cost a robot: a target in the first
view is best found by submitting at once, one behind the robot by one turn and a submission (see
Metric).

## The interface: `policy.py`

`policy.py` must define

```python
def search(env) -> None:
```

The evaluator calls it once per episode, in a separate process, with `WORKERS` episodes at a time
(a module constant of `policy.py`, default 8, at most 16), so `search` must be safe to call from
several threads. `env` provides:

| | |
|---|---|
| `env.kind` | `"hos"` or `"hps"` |
| `env.task` | the instruction |
| `env.view` | path of the current view (JPEG) |
| `env.yaw`, `env.pitch` | current direction in degrees: yaw in [0, 360), pitch in [-90, 90] |
| `env.step`, `env.steps_left`, `env.max_steps`, `env.done` | the step budget |
| `env.rotate(dyaw, dpitch)` | turn relative to the current direction (yaw > 0 turns right, pitch > 0 looks up; pitch is clamped to [-90, 90]); returns the new view's path, or `None` when that was the last action |
| `env.submit(yaw, pitch)` | end the episode with this absolute direction |

Angles are rounded to whole degrees. An episode whose `search` raises keeps whatever it submitted
(nothing, if it did not), and the others go on. A whole split must finish within 45 minutes.

Each episode is solved on its own, from its instruction and its own views: the evaluator starts an
episode (and renders its first view) only when `search` is called for it, knows the episodes by
opaque ids, and keeps at most 16 in progress. A policy must not use anything from other episodes
(their views, instructions or outcomes) or drive the environment's HTTP interface itself.

## Model access

The policy calls the model under test while it is evaluated: `model.py` in the workspace sends a
conversation through the evaluator's gateway (`$RSIB_GATEWAY_URL`, `$RSIB_GATEWAY_TOKEN`); you may
change or replace it. A request is the gateway's `/v1/complete` body: `system`, `messages` (a whole
conversation; images go in user messages), `tools`, `max_tokens`. **The call budget:** a split of n
episodes may make at most 20n model calls in total, 20 per episode on average (`MAX_CALLS` in
`model.py`, from `$HSTAR_MAX_CALLS`); the evaluator refuses every call after that (HTTP 400, error
`call_budget`). Model calls on dev are charged to your budget; on val and test they are metered
only.

The starter is the H*Bench baseline: one model call per step on the whole conversation, with the
benchmark's own prompts.

The evaluation runs on one GPU with Python, numpy, scipy, Pillow and PyTorch. Open models, code and
Python packages you need at evaluation time must be downloaded into `$RSIB_CACHE` while you work
(`HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` point there); the evaluated policy must not use the
network, except the model gateway. If it needs another Python interpreter, that must live in
`$RSIB_CACHE`; write its absolute path in `runtime.txt` in the workspace.

## Metric

`spl`: success weighted by steps, the mean of the object-search and path-search averages (higher is
better). An episode scores

    success x fewest / max(fewest, steps)

where `steps` counts its actions (turns and the submission) and `fewest` is the fewest actions that
can succeed: 1 when the target is in the first view (its center within 50 degrees of the starting
yaw), else 2 (one turn, then submit). A success after a full look around (3 turns and a
submission) scores 0.25 or 0.5; a failure scores 0.

Reported too: `success` (the mean of the plain success rates), `hos_spl`, `hps_spl`, `hos_sr`,
`hps_sr`, the success rate per difficulty level (`hos_sr_l0` ...), `avg_steps`, `hos_avg_steps`,
`hps_avg_steps`, `avg_steps_success` (over the successful episodes), `submitted` (share of
episodes that submitted), `model_calls` and `refused_calls`.
Difficulty: for object search, level 0 means the target is visible in the first view, 1 partly
visible and 2 not visible; for path search, levels 0 to 3 go from clear, aligned cues to scenes
without them or with misleading ones.

## Data and rules

The dev data directory holds `episodes.jsonl` (`{"id", "kind", "task", "start_yaw", "scene",
"panorama"}`), `labels.jsonl` (`{"id", "kind", "level", "yaw": [lo, hi], "pitch": [lo, hi] or
null}`, in degrees, the annotated target before widening) and `panoramas/` (the equirectangular
panoramas, for your own analysis: the policy never gets them). It holds 80 episodes (40 object
search, 40 path search), drawn the same way as the hidden val split (100 episodes: 50 and 50); the
two splits share no scene, and each episode has one fixed starting direction. The hidden test split
is a separate set of in-the-wild scenes. `evaluate_dev` returns the metrics and
`per_episode.json` with every episode's actions, submitted direction, outcome, `fewest_steps` and
score.

H*Bench is a public benchmark. Do not download, search for or use the benchmark, its scenes, its
annotations or its training data from anywhere (Hugging Face, GitHub, papers, other copies), and do
not look for the validation or test data. Use only the dev data, your own work and publicly
released models and libraries; publicly released model weights trained for this kind of search are
allowed.
