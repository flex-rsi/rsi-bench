# Testing guide

From a fresh clone to a scored run: evaluating one agent on one task with the standard run.

## Before you start

- The repository installed ([README](../README.md#install)) on the machine that will run the
  orchestrator: a workstation or a cluster login node. The orchestrator is a light process; it must
  stay up for the whole run (use `tmux` or `nohup`), and it needs network access to the model.
- Compute for the task's experiments and evaluations: local GPUs or a SLURM cluster
  ([operations](operations.md#where-the-orchestrator-runs)). The [task details](tasks.md) list what each task
  needs.
- Unprivileged user namespaces, which the filesystem sandbox uses (on by default). The smoke test
  below tells you whether they work.

## 1. Pick a task and prepare its data

Choose a task from the [task list](tasks.md); its entry says where its data comes from and how to
prepare it. Data lives outside git, under `$RSIB_DATA_ROOT/<task>/{dev,val,test}`:

```bash
export RSIB_DATA_ROOT=/path/with/space/rsib-data
```

Keep the data root, and any token you used to download it, out of reach of agent code: the sandbox
hides everything outside the run's own directories, but a run made with `--no-sandbox` could read
them.

## 2. Configure the site

Copy `examples/site.example.toml` to `~/.config/rsib/site.toml` (or point `$RSIB_SITE` at it) and fill
in the compute backend: the SLURM account and partitions, the container images that task names map
to (for example `pytorch`), mounts, the local GPU pool, and the gateway address when jobs run on
other nodes. Every setting is explained in [operations](operations.md#site-profile).

## 3. Connect the model

Claude and GPT models can run through a local Claude Code or Codex login; other models need an API
key or a self-hosted OpenAI-compatible server. [models.md](models.md) covers each way, including
what a self-hosted server must support and how to keep it up for a 10-hour run.

## 4. Smoke-test the setup

Before spending a real budget, make a short run with the mock model: it
checks that the task package loads, the data is in place, the backend and container work, the
sandbox starts and the starter scores on validation and test.

```bash
rsib run new --task <task> --model mock:finish --mode base \
    --max-rounds 1 --budget wall_hours=0.5
rsib run start <worktree> --backend slurm      # or --backend local
rsib report <worktree>
```

Then repeat with your real model and a tiny budget (`--max-rounds 1
--budget wall_hours=0.5`) to check that its tool calls go through.

## 5. Run

Create the run with the standard flags of the task from its [details](tasks.md), adding your
model, effort and seed:

```bash
rsib run new --task <task> --model <model> --effort <effort> --seed 1 <standard flags of the task>
rsib run start <worktree> --backend slurm      # or --backend local
```

- **Interruptions.** `rsib run start <worktree>` resumes a stopped run from its last complete round
  ([operations](operations.md#where-the-orchestrator-runs)). Time the model is unreachable and time
  jobs wait in a queue are not charged to the wall budget.
- **Do not change** the standard flags, the task package or the starting harness for a run you want
  to compare with others: such runs get a different `protocol_id` or release and are grouped apart.

## 6. After the run

When the budget runs out the orchestrator scores the final solution on test, scores every accepted
version on test, inventories and archives the run's scratch directory, and commits the records to
the run's branch. Look at the results:

```bash
rsib report <worktree>                          # one run: scores, rounds, harness edits, cost
rsib validate --ref <branch>                    # integrity checks of the run branch
rsib results table --branches 'run/*'           # every local run, grouped and averaged
rsib results report --output RESULTS.md         # a full report of every run your clone knows
```

A run counts as a result when it was made from a release tag, passes validation, has a test score
for every task and was sandboxed ([results](results.md#what-a-result-is)).

**Submitting results.** The channel for submitting runs and having them verified will open as soon as possible.
Keep your run branches (do not rebase or delete them): they hold everything a later submission
needs.

## Modes and run shapes

The mode decides what the improver may change:

| Mode | Executor | Improver edits E | Improver edits I |
|---|---|---|---|
| `base` | yes | no (no improver phase) | no |
| `fixed-i` | yes | yes | no (reverted) |
| `full` | yes | yes | yes, plus meta reviews |

Run shapes that build on the modes:

- **Stream**: repeat `--task` to run several tasks in order on one branch. The improver and its
  memory carry over between tasks; each task has its own workspace and budget; in mode `full` a
  meta review runs at every task boundary.
- **Improver import**: `--improver-from <ref>` replaces `harness/improver/` with the one at `<ref>`:
  your own improver on a branch, or the final improver of another run (`<branch>/final`)
  ([harness](harness.md#writing-your-own-improver-method-track)).
- **Executor import**: `--executor-from <ref>` replaces E (`harness/` except `harness/improver/`)
  with the one at `<ref>`.

`rsib results table` groups runs by task, model, effort, condition, release and `protocol_id`,
reports mean ± standard error per group and the gains between conditions
([configuration](configuration.md#collecting-results-across-runs)). When you compare conditions,
keep the model, the task and the budget the same and run several seeds of each.
