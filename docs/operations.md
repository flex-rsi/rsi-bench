# Operations: paths, compute backends, pushing, CI

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RSIB_REPO` | the checkout containing the installed package | repository with `tasks/`, `protocol/`, `harness/` and the release tags |
| `RSIB_RUNS_ROOT` | `<repo>/../<repo-name>-runs` | run worktrees; the private store (`_private/<run>/`: task copies, val and test evaluation outputs) |
| `RSIB_DATA_ROOT` | `<repo>/../<repo-name>-data` | task data splits |
| `RSIB_ARTIFACT_ROOT` | `$RSIB_RUNS_ROOT/_artifacts` | large outputs, logs, transcripts |
| `RSIB_SCRATCH_ROOT` | `$RSIB_RUNS_ROOT/_scratch` | one directory per sandboxed run for what agent code creates: `HOME`, the download cache, `/tmp` |
| `RSIB_ARCHIVE_ROOT` | `$RSIB_RUNS_ROOT/_archive` | scratch and large artifacts of finished runs, kept for review, then deleted |
| `RSIB_SITE` | `~/.config/rsib/site.toml` if present | site profile for compute backends |
| `RSIB_LOCAL_GPUS` | empty | GPU pool of the local backend, e.g. `0,1,2,3`, unless the site profile sets `[local] gpus` |
| `RSIB_PASS_ENV` | empty | extra variable names passed to phases and jobs, e.g. `HF_HUB_OFFLINE,NLTK_DATA`, besides the `BASE_NAMES` of `src/rsibench/envs.py` (`PATH`, `HOME`, `CUDA_HOME`, `HF_HOME`, `TORCH_HOME`, ...) |
| `RSIB_PRICING` | bundled `pricing.toml` | override the token price table |
| `RSIB_ENV_FILE` | `~/.config/rsib/.env`, then `<repo>/.env` | the only dotenv file to read (API keys and the four settings below) |
| `RSIB_RUN_NAMESPACE` | – | prefix of new run branches: `robo` -> `robo/run/...` (`--namespace`) |
| `RSIB_CLAUDE_CONFIG_DIR` | the CLI's default profile | Claude Code profile for `claude-cli` (`CLAUDE_CONFIG_DIR`) |
| `RSIB_CODEX_HOME` | `~/.codex` | Codex profile for `codex-cli` (`CODEX_HOME`) |
| `RSIB_METRICS` | – | metric plugins of `rsib report` and `rsib results` (`--metrics`; [configuration.md](configuration.md#metrics)) |
| `RSIB_GIT_NAME`, `RSIB_GIT_EMAIL` | `rsib-orchestrator`, `rsib-orchestrator@users.noreply.github.com` | identity of orchestrator commits |
| `RSIB_MOCK_SCRIPT` | – | default script for `mock:scripted` |
| `RSIB_STOP_AT` | empty | epoch seconds at which the hosting job ends; the run pauses between rounds before it |
| `RSIB_STOP_MARGIN_S` | `3600` | do not start a new round within this many seconds of `RSIB_STOP_AT` |

Model credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ARK_API_KEY`, ...)
are read by the orchestrator only, from the environment or else from a dotenv file:
`$RSIB_ENV_FILE`, or `~/.config/rsib/.env` and then `<repo>/.env` (the first file that has the
variable wins). `RSIB_RUN_NAMESPACE`, `RSIB_CLAUDE_CONFIG_DIR`, `RSIB_CODEX_HOME` and
`RSIB_METRICS` can be set there too. Values from the file are handed to the SDK client and never
put into the process environment. Phases, experiments and evaluators run with a scrubbed
environment and never see them. Phases do run shell commands as your user, though, so keep the
file outside the repository and the runs root for real runs.

With the SLURM backend, jobs run on other machines: the runs, data and artifact roots, and the
orchestrator's Python environment (the `RSIB_PYTHON` of jobs outside a container image), must be
on storage those machines can read at the same paths.

## Site profile

Backend settings live outside the repository, in a file such as `~/.config/rsib/site.toml`
(`*.local.toml` and `site.toml` are git-ignored). Start from the template
`examples/site.example.toml`. A filled-in profile for a SLURM cluster:

```toml
[slurm]
account = "my_account"
partition = "gpu_a,gpu_b"           # several partitions: SLURM picks the first with room
gpus_flag = "--gpus-per-node"       # the default; or set gres = true for --gres=gpu:N
cpus_per_gpu = 16                   # default 4
mem_per_gpu_gb = 200                # default 16
poll_s = 15                         # the default; seconds between sacct polls
max_queue_s = 21600                 # the default; cancel a job not done by submission + timeout + this
container_image = "/path/to/default.sqsh"     # optional default image for every job
container_mounts = ["/data:/data"]            # srun --container-mounts
container_flags = []                          # extra srun flags for container jobs
extra_sbatch = []                             # extra #SBATCH lines, e.g. ["--qos=normal"]
cpu_partition = "cpu_short"         # optional: CPU-only experiments and evaluations run here
cpu_cpus = 16                       # default 8; CPUs of such a job
cpu_mem_gb = 64                     # default 32
cpu_max_minutes = 240               # optional: clamp their time limit to the partition's

[slurm.containers]                  # names that tasks use in [rsi.resources] container
pytorch = "/path/to/pytorch.sqsh"

[local]
gpus = "0,1"
max_parallel = 4                    # jobs the local backend runs at once

[phase]                             # limits for phases, which run on this machine
cpus = 8                            # pin each phase and its children to this many cores
nice = 10                           # lower their scheduling priority
threads = 8                         # OpenMP / MKL / OpenBLAS / numexpr / Rayon thread pools

[gateway]
host = "0.0.0.0"                    # bind address (default 127.0.0.1: this machine only)
advertise = "login01.example.org"   # host name in RSIB_GATEWAY_URL (default: see below)
```

The SLURM backend writes an sbatch script next to the job log, starts the job with
`--export=NONE` plus explicit exports, polls `sacct`, and charges
`gpus × ElapsedRaw` GPU-hours. An experiment or dev evaluation whose phase ended while it was
still pending is cancelled with `scancel`. Container jobs do not inherit the host's `PATH`,
Python or CUDA variables, so the image's own environment applies, and `RSIB_PYTHON` is `python`,
in a task's own image (`[slurm.containers]`) as in the default `container_image`.
GPU jobs go through SLURM: experiments with `gpus > 0` and evaluations of tasks with
`resources.eval_gpus > 0`. CPU-only experiments and evaluations run locally, unless
`cpu_partition` names a partition for them: then they are SLURM jobs too, with `cpu_cpus`,
`cpu_mem_gb` and a time limit clamped to `cpu_max_minutes`. Their queue time is left out of the
wall budget like that of GPU jobs.

Phases always run on the orchestrator's machine, and so does every command an agent runs from its
shell. On a shared login node, set `[phase]`: each phase and all its children are pinned to the
last `cpus` cores (`taskset`), run at `nice`, and get the common thread-pool variables
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, ...) set to `threads`. The phase context tells the harness
both settings, and the starting executor asks the agent to run heavy work with `run_experiment`.

Phases, experiments and the evaluators of tasks with model access reach the orchestrator's gateway
at `RSIB_GATEWAY_URL`. It listens on `[gateway] host`, by default `127.0.0.1`, which only this
machine can reach, on a port the OS picks each time the gateway starts (not configurable). The URL
names `[gateway] advertise`, by default `host` itself, or this machine's host name when `host` is
`0.0.0.0`. A SLURM job on another node that calls the model (an experiment, or an evaluation of a
task with model access) needs a `host` that node can reach and an `advertise` name that resolves
to this machine there. The gateway speaks plain HTTP, so requests and tokens then travel
unencrypted over the cluster network: whoever can read that traffic can use a token until its
grant is revoked. Open it only on a network you trust. The SLURM backend writes a job's token into
its sbatch script next to the job log, like every variable it exports, and makes that script
readable by its owner only (mode 0600).

## Local backend

The local backend runs jobs (experiments and evaluations, also the CPU-only ones of a SLURM run)
as subprocesses of the orchestrator, at most `[local] max_parallel` at once (default 4); further
jobs wait for a free slot. GPUs come from a fixed pool, `[local] gpus` or else `RSIB_LOCAL_GPUS`.
A job asking for N GPUs waits until N devices of the pool are free and runs with exactly those in
`CUDA_VISIBLE_DEVICES`, so concurrent jobs never share a GPU; a job asking for more than the pool
holds is rejected. When a job ends, by exit, error or timeout, its process group is killed and its
devices return to the pool once its processes are gone, waiting at most 5 s for them (a process
stuck in uninterruptible sleep may outlive the kill). GPU hours are `gpus × running time`: the
wait for a slot or for devices is not charged, and the job's timeout counts from its start. An
experiment or dev evaluation whose phase ended while it waited is not started.

## Filesystem sandbox

Agents run code with the same user as the orchestrator, so without a sandbox they can read anything
that user can: other runs, the held-out splits, unrelated data, credentials in `$HOME`, and the
weights, environments and code that earlier agents left behind. **Every run is sandboxed by
default.** The site profile can adjust it:

```toml
[sandbox]
enabled = true          # the default; false makes unsandboxed runs (left out of results tables)
read_only = []          # shared resources agents may use, e.g. a public model cache
mask = []               # paths to hide inside allowed trees, e.g. "/run/munge" to block SLURM
hide = []               # extra roots to hide (the defaults are listed below)
git_view = true         # phases can read the run's own Git history (see below)
archive_days = 7        # how long a finished run's archived scratch is kept
max_procs = 8192        # processes (threads included) one sandboxed launch may hold; 0: no limit
```

`rsib run new` records the setting in `run.json` and checks that the machine supports it; if it
does not, the command stops, and only an explicit `--no-sandbox` creates an unsandboxed run. With
the sandbox on:

- **phases** run under `sandbox_exec.py`, which uses unprivileged user and mount namespaces (no
  root, no setuid helper). The top-level directories that hold the runs, data, artifacts and the
  repository, plus `/home` and any `hide` entries, are covered; only the run's worktree, its
  artifact directory, its scratch directory, the dev data, the interpreter and the `read_only`
  paths come back. `HOME`, `/tmp` and the download cache (`$RSIB_CACHE`, with `HF_HOME`,
  `TORCH_HOME`, `PIP_CACHE_DIR` and `XDG_CACHE_HOME` inside it) live in the run's scratch
  directory, `$RSIB_SCRATCH_ROOT/<run>/`, which no other run can see.
- **experiments** in a container mount exactly the workspace, the run's artifact and scratch
  directories and the dev data (`--no-container-mount-home`); jobs without a container run under
  the launcher.
- **evaluations** pass `RSIB_SANDBOX` to the task's evaluator, which starts the solution through it
  (see the task-authoring guide). The solution sees its snapshot, `$RSIB_CACHE` and its I/O
  directory, never the labels.
- processes a phase leaves running are stopped when the phase ends.
- every sandboxed launch (a phase, an experiment without a container, a solution under evaluation)
  may hold at most `max_procs` processes, threads included, counted in its own user namespace and
  inherited by everything it starts, also by processes that leave its process group. Code that
  forks without end gets `EAGAIN` instead of exhausting the machine. The limit is a site setting:
  it is not recorded in `run.json` and does not change the protocol fingerprint.
- val and test evaluation outputs (per-example scores, logs) are written under the private root,
  which no sandboxed process can see; only the scores reach the agent, through the records.

So a solution can use what it brings in its workspace or downloads into `$RSIB_CACHE` during a
phase, plus the `read_only` resources.

With `git_view` (recorded in `run.json` by `rsib run new`, on by default), each sandboxed phase
also receives a fresh read-only Git view: the run's current commit and its ancestors, its branch
ref, and only its own tags that point into that history. The branch is read from the worktree's
Git metadata, not from `records/run.json`, and must match the one the run records. Objects are
exported into a new repository; the shared object store, other runs' refs and objects, remotes,
hooks, reflogs and configuration are never mounted or copied. Inside the phase, the worktree's
`.git` file is replaced, read-only, by one that points at the view, so Git finds the view only
from within the worktree; no `GIT_*` variable is set (and none is passed through), so `git clone`
and Git in other directories behave as usual. `git log`, `git show` and `git diff` work; commits
and index writes are reserved to the orchestrator. The view lives under
`$RSIB_SCRATCH_ROOT/_phase-git/<run>/`, outside the scratch directory experiments mount; it is
removed after the phase's processes are stopped and rebuilt for the next phase, and
`rsib run start` removes views an interrupted orchestrator left behind. Experiments and evaluated
solutions do not receive it. If the view cannot be built, the phase runs without it (its `.git`
masked), the orchestrator logs why, and the round record carries the reason as the phase's
`warning`. Runs created before `git_view` was recorded run their phases as before, and the setting
is part of the `protocol_id`.

### When a run ends

With the protocol's `posthoc_at_end` and `archive_at_end` (both on by default), finishing a run
takes three steps after the final test:

1. every accepted version is scored on test (`records/posthoc.jsonl`), while the solutions can
   still load what they downloaded;
2. the scratch is inventoried into `records/environment.jsonl`: the Hugging Face models (with
   revisions), Python packages, git checkouts (remote and commit) and sizes the run depended on;
3. the scratch, and every artifact larger than `max_file_mb` (transcripts excepted), move to
   `$RSIB_ARCHIVE_ROOT/<run>__<time>/` with an `ARCHIVE.json` that gives its `delete_after` date.

The archive is hidden from every sandboxed process, so a later run, by the same or another agent,
starts without the weights, environments and code of earlier runs. What stays is what the run is
measured by: the run branch (records, notes, harness edits, the solution), the transcripts and
experiment metadata in the artifact directory, and the val/test outputs in the private store.
`rsib run new` and `rsib run start` delete archives whose retention is over; by hand:

```bash
rsib archive list
rsib archive purge                 # expired archives
rsib archive purge --run <run-id> --all   # one run's archive now
rsib run archive <worktree>        # archive a finished run that did not archive itself
```

A solution that loads weights from `$RSIB_CACHE` cannot be evaluated again once its scratch is
archived; to re-run `rsib run finalize` or `rsib posthoc` on it, move `<archive>/scratch` back to
`$RSIB_SCRATCH_ROOT/<run>` first. The sandbox needs unprivileged user namespaces
(`/proc/sys/user/max_user_namespaces` > 0); the tests skip when they are unavailable. A sandboxed
run's protocol fingerprint includes the sandbox, so it is never averaged with unsandboxed runs.

To replace invalid posthoc scores of accepted rounds in a finished run:

```bash
rsib posthoc <worktree> --redo 1,6,10 --redo-reason 'Earlier model calls failed'
```

Only the listed accepted rounds are evaluated. The command appends rows with `redo: true`,
`supersedes` (the earlier row's `evaluated_at`) and `redo_reason`; it never rewrites old rows.
Each evaluation writes to a fresh hidden directory such as `posthoc-r010-redo1/` so earlier
outputs and transcripts remain available. Reports and summaries use the latest appended row
for each task and round. A normal `rsib posthoc <worktree>` still skips rounds with existing
rows. See [posthoc records](records.md#posthocjsonl).

## Where the orchestrator runs

The orchestrator calls the model, so it runs wherever the model adapter can reach the model (for
`claude-cli` or `codex-cli`, a machine where `claude` or `codex` is signed in). It is a light,
long-lived process (up to about the sum of its tasks' wall-clock budgets); the harness phases and
CPU-only jobs run on the same machine, and GPU jobs go to the configured backend. Run it under a
process manager or detached, for example:

```bash
nohup rsib run start "$WORKTREE" --backend slurm > orchestrator.log 2>&1 &
```

`rsib run start` resumes an interrupted run. By default it discards the partial round
(uncommitted changes and commits after the last `records` commit, logged as `resume_discard`) and
runs that round again. `--close-partial-round` instead finishes that round with the work done so
far, provided the round has committed nothing but notes (typically, it was interrupted in its
executor phase); otherwise `rsib run start` stops with an error. If one of the round's dev evaluation calls
beat the accepted solution, the state the submit rule picked is submitted: from the first such
call, its best candidate. Either way the records the partial round wrote are kept, and its charged
model calls, GPU hours and wall time up to its last recorded activity count against the budget.
Time the orchestrator was down is not charged.

When the orchestrator itself lives in a time-limited job (for example next to a self-hosted model
server), set `RSIB_STOP_AT` to the job's end time. Before each round the orchestrator checks whether
a round still fits (`RSIB_STOP_MARGIN_S`); if not, it stops cleanly at the round boundary and prints
`paused`. Start the next job with the same `rsib run start` to continue; nothing is discarded and
the time between jobs is not charged to the wall-clock budget.

## Pushing run branches

```bash
rsib run start <worktree> --push origin     # push after every round (branch + its tags)
rsib push <worktree> --remote origin        # push once, e.g. after the run
```

The orchestrator pushes with plain `git push`, so the machine running it needs git credentials
for the remote: an SSH remote, or HTTPS with a credential helper (for example `gh auth setup-git`).
A failed push is logged as a warning and the run continues; `rsib push` can retry later.
`rsib run finalize` and `rsib posthoc` never push: run `rsib push` after them.

Pushing is optional: a run is complete in your local clone. Push to your own fork or remote if you
want a copy elsewhere; the run branch's `<branch>/...` tags are force-pushed, because
`rsib run finalize` can move `<branch>/final`.

## CI

`.github/workflows/ci.yml` runs the unit and end-to-end tests on pushes to `main` and on pull
requests, except those that change only documentation (`docs/**`, `*.md`).

## Housekeeping

```bash
git worktree list                     # all runs on this machine
git worktree remove <worktree>        # the branch and tags remain
git worktree prune
```

Scratch and large artifacts of finished runs are archived and later deleted automatically (see
[When a run ends](#when-a-run-ends)). The rest of `$RSIB_ARTIFACT_ROOT/<run>` stays: remove it
when the transcripts and logs are no longer needed; `records/artifacts.jsonl` keeps their paths and sizes, and
hashes of experiment outputs ([records.md](records.md#artifactsjsonl)). Val and test outputs, with
a workspace snapshot per evaluation, collect in `$RSIB_RUNS_ROOT/_private/<run>/evals/`; remove it
once the run is final and post-hoc scored. Keep `_private/<run>/tasks/`: `rsib run finalize` and
`rsib posthoc` evaluate from it.
