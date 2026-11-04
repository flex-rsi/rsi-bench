# Authoring a task

A task package tells the benchmark what the agent must improve, where its data lives, how a
solution is scored, and what "good" means (anchors). The format is Harbor's task layout with an
extra `[rsi]` section, so the same directory can be read by Harbor-based tooling.

## Layout

```
tasks/<id>/
├── task.toml
├── instruction.md
├── environment/workspace/      starter solution, copied to workspace/<id>/ in every run
├── prepare.py                  builds the data splits
└── tests/
    └── test.sh                 evaluator entry point
```

Start from `tasks/_template/`. Directories whose name begins with `_` are ignored by `rsib task list`.

## task.toml: the [rsi] section

| Key | Meaning |
|---|---|
| `id` | must equal the directory name |
| `domain` | free text such as `vision/segmentation`, `robotics/sim` or `generation/video`, recorded in `run.json` and passed to the harness; use it to split meta-train / meta-test tasks |
| `tier` | budget tier from `protocol/default.toml` (`XS`, `S`, `M`, `L`) |
| `artifact` | what the agent ultimately produces: `code`, `checkpoint`, `predictions` or `samples` |
| `verifier_tier` | trust in the metric: `exact` (computed from labels), `learned` (FID, FVD, CLIP...), `judge` (a model grades it) |
| `metric.primary`, `metric.direction` | key in `reward.json`; `maximize` or `minimize` |
| `metric.secondary` | other metrics of `reward.json` worth watching |
| `metric.public` | optional list of the scalar metrics val and test records keep besides the primary metric (default: all; see [the evaluator contract](#the-evaluator-contract)) |
| `anchors.baseline`, `anchors.reference` | scores that map to 0 and 1 in normalized results |
| `noise.sigma`, `noise.repeats` | standard deviation of the baseline score over repeated evaluations, and their number |
| `splits.dev/val/test` | sub-directories of `$RSIB_DATA_ROOT/<id>/` |
| `entrypoints.prepare/evaluate/workspace` | paths relative to the task directory |
| `resources.experiment_gpus_max` | GPUs one `run_experiment` call may request (0 = CPU only) |
| `resources.eval_gpus` | GPUs the evaluator needs; >0 sends evaluations to the configured backend |
| `resources.container` | logical image name (e.g. `pytorch`); the site profile maps it to an image, and the task's SLURM jobs run inside it |
| `resources.model_access` | `true`: the solution may call the run's model when it is evaluated ([below](#solutions-that-call-the-model)); default `false` |
| `resources.eval_model_calls_max` | with `model_access`: the most model calls one evaluation may make; default the protocol's `max_model_calls_per_phase` (80) |

`artifact`, `verifier_tier`, `metric.secondary` and `noise.repeats` are informational: no command
reads them. The Harbor fields (`[task]`, `[agent]`, `[environment]`, `[verifier]`, `[scoring]`) are
kept for compatibility. Of these, `[verifier].timeout_sec` is the evaluator timeout (default 600 s),
and `[scoring].direction` applies when `metric.direction` is missing.

## Data splits

`rsib task prepare <id>` runs `python prepare.py --out $RSIB_DATA_ROOT/<id>`. Produce:

- `dev/`: the agent's data. It may include labels and a training set. Its path is given to the
  executor and to experiments as `$RSIB_DEV_DATA`.
- `val/`: used by the orchestrator to accept or reject candidates. The agent sees the scores,
  never the data.
- `test/`: used after the run and by `rsib posthoc`, including explicit
  [redos of accepted rounds](records.md#posthocjsonl). Nobody sees anything during
  the run.

Keep val and test large enough that the noise sigma is small relative to the improvements you
expect, and draw them from the same distribution as the dev queries. If the domain shifts between
val and test, say so in the task README.

## The evaluator contract

`tests/test.sh` runs under bash in `$RSIB_OUTPUT_DIR`, with a scrubbed environment (no credentials;
[`RSIB_PASS_ENV`](operations.md#environment-variables) passes extra variables) plus:

| Variable | Meaning |
|---|---|
| `RSIB_SPLIT` | `dev`, `val` or `test` |
| `RSIB_DATA_DIR` | that split's directory (inputs and labels) |
| `RSIB_TRAIN_DIR` | the dev directory (training data the solution may use) |
| `RSIB_WORKSPACE` | the solution to score (treat it as read-only); in a run, without its files over `max_file_mb` or caches |
| `RSIB_OUTPUT_DIR` | write `reward.json` here |
| `RSIB_TASK_DIR` | the private copy of the task package |
| `RSIB_PYTHON` | the orchestrator's Python interpreter; `python` inside a container image |
| `RSIB_SANDBOX` | JSON list, empty when the run has no sandbox: launcher prefix for the solution's process |
| `RSIB_GATEWAY_URL`, `RSIB_GATEWAY_TOKEN` | only with `resources.model_access`: the gateway and a token for `/v1/complete` until the evaluation ends, to hand to the solution |

Several evaluations of one task can run at once (the candidates of one dev evaluation call), each
with its own `RSIB_WORKSPACE` and `RSIB_OUTPUT_DIR`, so an evaluator writes only to its output
directory and to temporary directories it creates.

`reward.json` must contain the primary metric as a finite number. A plain Harbor file
(`{"accuracy": 0.93, "n": 1000}`) is valid. Three keys are reserved for richer results (the
Result contract; schema: `src/rsibench/schemas/result.schema.json`):

```json
{"accuracy": 0.93,
 "metrics":   {"per_class": {"0": 0.99, "7": 0.81}},
 "artifacts": [{"path": "failures/7.png", "role": "failure", "note": "a 7 read as 1"}],
 "info":      {"confusions": [[7, 1, 12]]}}
```

`metrics` holds more metrics (nested values are allowed), `artifacts` lists files the evaluator
wrote under `$RSIB_OUTPUT_DIR` (images, videos, trajectories; `mime` is guessed from the name if
missing), and `info` is a free-form object. A non-zero exit status, a timeout, a missing
`reward.json` or one that is not a JSON object, a missing or non-finite primary metric, or a
malformed reserved key counts as a failed evaluation, and the candidate is rejected.

Visibility depends on the split. On dev, the agent receives the whole result, and the artifacts
are readable from its tools, so the evaluator can explain failures. On val and test, the records
keep only the score and the scalar metrics; nested metrics, artifacts and info stay in the private
store (`$RSIB_RUNS_ROOT/_private/<run>/evals/`), because per-example outputs would leak the split.
For the same reason a failed val or test evaluation is reported by where `reward.json` breaks the
contract (`info: breaks type "object"`), never by its values.

Which scalar metrics of val and test are public is part of the task:

```toml
[rsi.metric]
primary = "accuracy"
public = ["n"]        # val and test records keep accuracy and n, no other metric
```

With `public`, the val and test records (`rounds.jsonl`, the `eval-val` and `eval-test` lines of
`experiments.jsonl`, `final.jsonl`, `posthoc.jsonl`) keep only these keys plus the primary metric;
without it they keep every scalar metric. Leave out scalars that tell too much about a hidden split,
such as error counts per class. Dev results reach the agent in full either way. The list lives in
`task.toml`, so it is covered by the task's version hash and by the `protocol_id` of its runs.

Recommended structure (see `tests/fixtures/tasks/toy-regression/tests/`): copy the split's inputs to a temporary
directory, run the solution there in a child process that cannot see the labels, then score the
predictions in the parent. This keeps a submitted solution from reading the labels it is scored on.

When the run is sandboxed, `RSIB_SANDBOX` holds a launcher prefix. Start the solution's process as
`prefix + ["--rw", io_dir, "--cwd", io_dir, "--"] + command`: it then sees its own snapshot, the
run's download cache and `io_dir`, and nothing else (not the labels, not other runs). Copy any
helper script the child needs into `io_dir` first. `solution_cmd()` in
`tests/fixtures/tasks/toy-regression/tests/evaluate.py` is a ready-made version; with an empty `RSIB_SANDBOX` it
returns the command unchanged.

The evaluator runs from a private copy of the task package under `$RSIB_RUNS_ROOT/_private/`, so
edits the agent makes to `tasks/` in its worktree have no effect, and they are reverted anyway.

## How agents may solve a task

The benchmark does not restrict *how* an agent solves a task. It may look at the dev data itself
(`view_image`), study failures, write rules, tune parameters, write code, train small parts, or
combine pretrained models; everything it does goes through the harness's tools. One rule is hard:
during its phases the agent never sees val or test inputs, so whatever is scored is the solution
in the workspace, run by the evaluator.

Whether the solution may use the model under test while it is evaluated is a property of the task,
declared below (`model_access`). Decide it by the kind of output:

- **on** when a model can produce the answer by looking at the input: visual and spatial QA,
  multimodal reasoning, embodied QA and planning, the planning side of unified multimodal tasks,
  code-as-policy with a model in the loop. Letting the model look and answer is then a legitimate
  solution, often the strongest first one;
- **off** when the output is dense or geometric (masks, tracks, depth, poses, point clouds,
  generated media scored by fixed metrics): the solution is self-contained code that may load
  open models from its workspace or the run cache.

For a task with model access, make the **starter the direct pipeline**: send the input to the
model, parse its answer. The baseline anchor then measures what the model does by looking, and
the normalized score and the RSI gains measure only what iteration adds on top (tools,
verification, decomposition, better prompts). A task that instead wants a named open model to do
the core inference uses a fixed backbone (as `quantiphy-*` and `edit-multiturn` do) and leaves
`model_access` off.

## Solutions that call the model

A solution may use the model under test at inference time, for example a prompting pipeline or a
small agent. The task then declares

```toml
[rsi.resources]
model_access = true     # a boolean; default false
```

Like every key of `task.toml`, it is part of the task's version hash. Every evaluation of such a
task gets a [grant](harness.md#grants) of its own: the evaluator receives `RSIB_GATEWAY_URL` and
`RSIB_GATEWAY_TOKEN`, which allow `POST /v1/complete` (the request of the
[gateway API](harness.md#gateway-api)) and nothing else, until the evaluation ends. The evaluator
hands them to the process that runs the solution, as it hands it the split's inputs. Without
`model_access`, the evaluator gets no gateway variables. An evaluator on another node (SLURM)
reaches the gateway only if the site profile's `[gateway]` settings allow it
([operations.md](operations.md#site-profile)).

| Evaluation | `phase` of its usage rows | Charged to the task budget | Transcripts |
|---|---|---|---|
| dev (`evaluate_dev`, the reference score of the accepted solution) | `eval-dev` | yes | with the evaluation's outputs, readable by the agent |
| val | `eval-val` | no: metered (`charged: false`) | in the private store |
| final test, `rsib posthoc` | `eval-test`, `posthoc` | no: metered | in the private store |

Because val and test calls are not charged, a spent budget never fails a validation or a test
evaluation; reports sum up the calls of the val and final test evaluations as `eval_model`
([configuration.md](configuration.md#metrics)). The records keep how many calls an evaluation made
and their token counts, never their content. One evaluation may make at most
`resources.eval_model_calls_max` calls, by default the protocol's `max_model_calls_per_phase` (80);
a solution that calls the model per example or per step needs a cap above the calls one split
takes. Every val and test evaluation may spend that many calls without charge, so set it from a
measured evaluation, not generously. Experiments can always call
the model ([harness.md](harness.md#experiments)). `rsib task calibrate` runs evaluators without a
gateway.

## Anchors and noise

```bash
rsib task calibrate <id> --repeats 5      # scores the starter workspace on val 5 times
rsib task calibrate <id> --workspace /path/outside/the/repo/strong_solution   # reference score
rsib task calibrate <id> --backend slurm  # GPU evaluators (uses the site profile and container)
```

The command prints a suggested `anchors.baseline` and `noise.sigma`. Set `anchors.reference` to
the val score of a strong known solution: published state of the art, an expert implementation,
or a solution you wrote for the purpose. **Do not commit reference solutions or hidden labels to
this repository.** The agent can read every file in its worktree, and the worktree contains
`tasks/`.

The noise sigma matters: the default acceptance rule requires a gain of `2 × sigma`, and
`noise.sigma = 0` means any strictly better val score is accepted. For stochastic training,
sampling-based generation metrics or simulated rollouts, measure sigma with several seeds.

## Choosing the tier

A run needs enough budget for several rounds, within every limit of its tier. A rough sizing rule is
`budget ≈ rounds × (executor experiments per round × cost per experiment + val evaluation)`,
in GPU-hours and in wall-clock hours alike (the starting executor runs one experiment at a time,
and the model calls take time too). If a training run takes 10 minutes on 4 GPUs and you want
about 15 rounds with 3 experiments each, that is about 30 GPU-hours and 7.5 hours: tier `M`
(64 GPU-hours, 12 hours). Tiers are defined in `protocol/default.toml`
([configuration.md](configuration.md#budget-tiers-tiers-per-task)).

## Domain notes

**Segmentation, depth estimation.** Exact metrics (mIoU; AbsRel and δ1). The artifact is usually a
training recipe plus inference code. Experiments train, and the evaluator runs inference on val or
test with the checkpoint the solution points to. Put checkpoints under `$RSIB_ARTIFACT_DIR` and
let the workspace record where they are. Files over `max_file_mb` are never committed.

**Scene reasoning / VQA.** Prefer multiple-choice or numeric answers (exact match). Open-ended
answers need a judge model; freeze it, choose it from a different model family than the model
under test, and measure sigma, because judge scores are noisy.

**3D reconstruction.** Chamfer distance or F-score at a fixed threshold, evaluated against
held-out geometry. Outputs are large; keep them in the artifact store.

**Robotics (simulation).** Isaac Lab, ManiSkill or similar. Score the success rate over N episodes
with hidden reset seeds for val and test, and set `eval_gpus >= 1`. Reward-design and
code-as-policy tasks keep the artifact small (a reward file or a policy script); policy-learning
tasks produce a checkpoint. Budget tier `L`.

**Image, video and audio generation.** FID, FVD or FAD, plus a text-alignment score (CLIP, CLAP or
a frozen VLM judge). Fix the number of samples and the seeds in the evaluator. These metrics are
noisy and expensive, so measure sigma with at least three repeats and keep `val_max_calls` low.

## Porting Harbor tasks (RSI-Exam, OpenRSI Index)

Those tasks evaluate inside their own containers. To port one:

1. Copy `task.toml`, `instruction.md`, the starter under `environment/` and `tests/`, then add an
   `[rsi]` section.
2. Map container paths in the evaluator to the `RSIB_*` variables: the solution directory (often
   `/app/methods/main`) becomes `$RSIB_WORKSPACE`, the reward file (often
   `/logs/verifier/reward.json`) becomes `$RSIB_OUTPUT_DIR/reward.json`, and the sealed data
   becomes `$RSIB_DATA_DIR`.
3. If the evaluator needs the task's image, set `resources.container` to a name for it, map that
   name to the image under `[slurm.containers]` in the site profile, and make the directories the
   `RSIB_*` variables name (data, runs and artifact roots) visible at the same paths with
   `container_mounts` ([operations.md](operations.md#site-profile)). Only SLURM jobs run in
   containers, so set `eval_gpus >= 1`: CPU-only evaluations run locally.
4. Run `rsib task calibrate` and fill in the anchors and sigma.

## Checklist

- [ ] `rsib task check <id>` passes and all three splits exist
- [ ] the starter workspace scores without errors on every split
- [ ] a strong solution scores clearly better; its val score is `anchors.reference`
- [ ] `noise.sigma` is measured, not guessed
- [ ] no reference solution, hidden labels or test data in the repository
- [ ] `instruction.md` states the exact artifact interface and resource limits
- [ ] with `model_access`, the evaluator passes `RSIB_GATEWAY_URL` and `RSIB_GATEWAY_TOKEN` to the
  solution, and `instruction.md` says how to use them
- [ ] the task is committed in the release your runs start from: `rsib run new` reads task packages
  from that commit (`--base`), not from the working tree ([README](../README.md#install))
