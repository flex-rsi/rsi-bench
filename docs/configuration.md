# Configuration reference

Everything a user can set, with defaults, grouped by how it affects comparability:

| Layer | Where | Changes comparability? |
|---|---|---|
| Run options | `rsib run new` flags | recorded in `records/run.json`; compare only runs with equal values |
| Protocol constants | `protocol/default.toml` (frozen per release tag) | yes: every run in a comparison must use the same release |
| Harness hyperparameters | `harness/executor/config.toml`, `harness/improver/config.toml` | they are the starting harness (E0/I0); the agent may change them during a run |
| Task settings | `tasks/<id>/task.toml` | part of the task definition (hashed into `run.json`) |
| Site settings | site TOML, environment variables, `rsib run start` flags | no (where things run, not what is measured), except `RSIB_PRICING`: the token prices the `usd` budget is charged at |
| Metric plugins | `--metrics`, `$RSIB_METRICS` ([Metrics](#metrics)) | no (computed from the records afterwards) |

## When does a run stop?

A **task** ends at the first of:

| Reason (`ended_reason`) | Condition | Default | Set with |
|---|---|---|---|
| `budget_wall` | wall-clock hours used ≥ limit (waits on infrastructure excluded, see below) | by tier | `--budget wall_hours=H` or `--tier` |
| `budget_usd` | USD used ≥ limit (if the model is priced and the limit is > 0) | by tier | `--budget usd=X` |
| `budget_tokens` | input tokens (cache reads and writes included) + output tokens ≥ limit | by tier | `--budget tokens=N` |
| `budget_gpu` | GPU hours used ≥ limit (only if the limit is > 0) | by tier | `--budget gpu_hours=H` |
| `val_calls` | candidates scored on val ≥ limit | by tier | `--budget val_max_calls=N` |
| `max_rounds` | round number > limit (a safety cap) | 200 | `--max-rounds N` |
| `idle` | consecutive rounds with no candidate and no harness change | 5 | protocol `max_idle_rounds` |

Runs end on **budget**, not on a round count: one round costs minutes on a segmentation task and
hours on a robotics or video task, so a fixed number of rounds would mean very different amounts of
work on different tasks. `max_rounds` is only a safety cap.

Budgets are checked before every charged model call, experiment and dev evaluation call (once for
a call with several candidates), and each is charged when it ends, so the calls in flight at once
(of the phase, its experiments and its evaluators) can overshoot a limit by at most their combined
cost. A candidate's val evaluation runs even on a spent budget and is charged its GPU hours; the
model calls of val and test evaluations (tasks with `model_access`) are metered but never
charged, so they never stop a task.

Time the model is unavailable is not charged as wall time either: a CLI adapter waits out a provider's session or usage limit (`outage_wait_s`), the orchestrator starts no phase or evaluation while the model is known to refuse calls, a phase stopped by such an outage is run again when the model is back (twice at most), and an evaluation during which the model became unavailable is run again. The waits are reported as `queue_s`. A round in which every phase still ended with a model error (a provider message the adapter does not recognize, or a limit that outlasts `outage_wait_s`) pauses the run at the round boundary rather than counting toward `idle`: `rsib run start` prints the error, and running it again continues with the next round.

An **executor round** also ends as soon as a dev evaluation beats the dev score of the accepted
solution by a clear margin (the submit rule, `submit_on_dev_improvement` and `submit_min_gain`;
[harness.md](harness.md#gateway-api)): exactly that evaluated state (of a call with several
candidates, the best of those past the line) is submitted to validation, and the improver reviews
after it. The bar is the dev score of the accepted solution (measured by the orchestrator when it
is not known); the margin is `submit_min_gain` times the task's anchor span
|reference − baseline|. A smaller gain leaves the round running: the executor keeps working
toward the line, or submits the workspace as it is by calling `finish`. Every submission spends
one validation call, and the executor's round prompt says how many it has used. An executor cut
off before `finish` (`unfinished_submit = "best_dev"`) submits only the best state one of its dev
evaluations showed to beat the accepted solution; with none, its changes are discarded.

A **phase** (executor, improver or meta) ends at the first of: the agent calls `finish`; the
harness step limit (`max_steps`); too many replies without a tool call (`max_nudges`); the
protocol call cap `max_model_calls_per_phase` (default 80, `--max-calls-per-phase`); budget
exhaustion; `phase_timeout_min` (180); a model refusal or error.

So the number of RSI rounds is bounded by `max_rounds`, and in practice by the budget. Self-review
of the improver (meta phase) happens only in mode `full`: every `meta_every_rounds` rounds
(default 4) and at every task boundary of a stream; its depth is bounded by the same round and
budget limits.

For smoke tests, tighten everything at creation time:

```bash
rsib run new --task vsearch --model claude-cli:claude-opus-5-5 --effort high --mode full \
    --max-rounds 3 --max-calls-per-phase 30 \
    --budget usd=5 --budget tokens=3000000 --budget wall_hours=0.5
```

## Run options

### `rsib run new`

| Flag | Default | Meaning |
|---|---|---|
| `--task ID` (repeatable) | required | one task, or several in order (a stream) |
| `--model MODEL` | required | a model name (`claude-opus-5-5`, `gpt-6.1-sol`, `gemini-…`, `doubao-…`, `NAME@URL`) or a full spec `claude-cli:…`, `codex-cli:…`, `anthropic:…`, `openai:…[@URL]`, `mock:scripted`, `mock:finish` |
| `--auth cli\|api` | `cli` | how a model name is reached: `cli` = local CLI login (`claude-*` via Claude Code, `gpt-*` via Codex), `api` = API key (see below); a full spec implies it |
| `--effort LEVEL` | model default | reasoning effort, fixed for the whole run |
| `--model-params JSON` | `{}` | adapter parameters (below) |
| `--mode` | `full` | `base` (no improver), `fixed-i` (improver edits E only), `full` (E and I, plus meta reviews) |
| `--seed N` | `0` | recorded; use several seeds per condition |
| `--base TAG` | newest `harness-v*` in the checked-out branch | release the run branches from |
| `--improver-from REF` | – | import `harness/improver/` from another run (probe runs) |
| `--executor-from REF` | – | import E (`harness/` except `harness/improver/`) from another run; in mode `base`, a run's final E against its base release measures what its harness edits gained |
| `--tier XS\|S\|M\|L` | the task's tier | budget tier for every task of the run |
| `--budget KEY=VALUE` (repeatable) | the tier's values | override `wall_hours`, `gpu_hours`, `usd`, `tokens`, `val_max_calls` |
| `--max-rounds N` | protocol (200) | override `max_rounds` |
| `--max-calls-per-phase N` | protocol (80) | override `max_model_calls_per_phase` |
| `--exact-context` | protocol (`false`) | set `exact_context`: refuse model calls whose context the adapter cannot deliver as given |
| `--submit-min-gain X` | protocol (0.02) | override `submit_min_gain`: the dev gain, as a share of the anchor span, that ends a round and submits its state |
| `--namespace NAME` | `$RSIB_RUN_NAMESPACE`, else none | run branch prefix: `robo` -> `robo/run/...` (a branch named `robo` must not exist) |
| `--notes TEXT` | – | free text stored in `run.json` |

### `rsib run start WORKTREE`

| Flag | Default | Meaning |
|---|---|---|
| `--backend local\|slurm` | `local` | where GPU jobs (experiments, evaluations) run; CPU-only jobs always run locally |
| `--site FILE` | `$RSIB_SITE` or `~/.config/rsib/site.toml` | site profile (SLURM account, partitions, containers) |
| `--push REMOTE` | – | push the branch and its tags after every round |
| `--close-partial-round` | off | finish an interrupted round with the work done so far (its executor counts as stopped; an improving dev evaluation is submitted) instead of discarding it |

`rsib run start` also resumes an interrupted run.

### Other commands

| Command | Options |
|---|---|
| `rsib task list` / `check ID` / `prepare ID` | – |
| `rsib task calibrate ID` | `--repeats N` (3), `--split` (`val`), `--workspace DIR`, `--backend`, `--site` |
| `rsib run status WORKTREE` | – |
| `rsib run finalize WORKTREE` | `--backend`, `--site` |
| `rsib posthoc WORKTREE` | `--every K` (1), `--redo N[,N...]`, `--redo-reason TEXT`, `--backend`, `--site` |
| `rsib report WORKTREE` | `--format md\|json`, `--output FILE`, `--metrics PLUGINS` |
| `rsib validate --ref BRANCH` | `--repo DIR` |
| `rsib results append --ref BRANCH` | `--repo DIR`, `--results-branch` (`results`), `--push REMOTE`, `--metrics PLUGINS` |
| `rsib results table` | `--repo DIR`, `--branches PATTERN`, `--results-branch`, `--task`, `--model`, `--base`, `--column NAME` (repeatable), `--metrics PLUGINS` (with `--branches`), `--format md\|csv\|json`, `--output FILE` |
| `rsib push WORKTREE` | `--remote` (`origin`) |

`rsib posthoc` normally scores accepted versions that have no posthoc row. `--redo 1,6,10`
instead re-scores only those accepted rounds and appends new rows; it rejects non-accepted
rounds. `--redo-reason TEXT` records why the scores were redone and requires `--redo`. The
latest appended row for each task and round wins. See the [record fields and preserved
evaluation outputs](records.md#posthocjsonl).

### Models and access modes

| `--model` | `--auth cli` | `--auth api` (adapter, endpoint, key variable) |
|---|---|---|
| `claude-*` | `claude-cli` (Claude Code login) | `anthropic`, Anthropic API, `ANTHROPIC_API_KEY` (else `ANTHROPIC_AUTH_TOKEN`, else the SDK's login profile) |
| `gpt-*` | `codex-cli` (Codex login) | `openai`, OpenAI API, `OPENAI_API_KEY` |
| `gemini-*` | – | `openai`, `https://generativelanguage.googleapis.com/v1beta/openai/`, `GEMINI_API_KEY` |
| `doubao-*` | – | `openai`, `https://ark.cn-beijing.volces.com/api/v3`, `ARK_API_KEY` |
| any other name, or `NAME@URL` | – | `openai`, OpenAI or `URL`, `OPENAI_API_KEY` |

The `gemini-*` and `doubao-*` endpoint and key variable are recorded in `run.json` under
`model.params` (other names use the adapter defaults below); `--model-params` (`base_url`,
`api_key_env`) and `NAME@URL` override them. Keys come from the environment or a dotenv file
([operations.md](operations.md#environment-variables)). Only variable names are recorded, never
values; do not put a key itself into `--model-params`, which is committed with the run.

### Model parameters (`--model-params`)

| Adapter | Key | Default | Meaning |
|---|---|---|---|
| `claude-cli` | `timeout_s` | 1800 | timeout of one CLI call |
| | `retries`, `backoff_s` | 3, 60 | retries on rate limits / overload, exponential backoff |
| | `outage_wait_s`, `limit_poll_s` | 21600, 300 | a session or usage limit is waited out (until the reset time the error names, else polling) for at most this long; the wait is left out of the wall-time budget |
| | `max_turns` | 4 | internal turns the CLI may take per call |
| | `binary`, `workdir` | `claude` on PATH, a temp dir | CLI executable; empty working directory |
| | `config_dir` | `$RSIB_CLAUDE_CONFIG_DIR`, else the CLI default | Claude Code profile (`CLAUDE_CONFIG_DIR`) |
| `codex-cli` | `timeout_s` | 1800 | timeout of one CLI call |
| | `retries`, `backoff_s` | 3, 60 | retries on rate limits / overload, exponential backoff |
| | `outage_wait_s`, `limit_poll_s` | 21600, 300 | a session or usage limit is waited out (until the reset time the error names, else polling) for at most this long; the wait is left out of the wall-time budget |
| | `binary`, `workdir` | `codex` on PATH, a temp dir | CLI executable; empty working directory |
| | `codex_home` | `$RSIB_CODEX_HOME`, else `~/.codex` | Codex profile (`CODEX_HOME`) |
| `anthropic` | `api_key_env` | `ANTHROPIC_API_KEY` | variable holding the key |
| | `max_retries`, `timeout_s` | 4, 1800 | SDK retries and timeout |
| | `prompt_cache` | true | automatic prompt caching |
| `openai` | `base_url` | OpenAI | any OpenAI-compatible server (also `openai:MODEL@URL`) |
| | `api_key_env` | `OPENAI_API_KEY` | variable holding the key |
| | `max_tokens_param` | `max_completion_tokens` | use `max_tokens` for servers that need it |
| | `max_retries`, `timeout_s` | 4, 1800 | |
| | `outage_wait_s`, `outage_backoff_s` | 0, 30 | while the server cannot be reached, retry every `outage_backoff_s` for at most `outage_wait_s` (0: fail at once); the wait is left out of the wall-time budget ([models](models.md#self-hosted-models)) |
| | `extra_body` | – | extra fields for every request, e.g. `{"chat_template_kwargs": {"enable_thinking": true}}` |
| `mock` | `script` | `$RSIB_MOCK_SCRIPT` | script for `mock:scripted` |

## Protocol constants (`protocol/default.toml`)

### `[run]`

| Key | Default | Meaning | Per-run override |
|---|---|---|---|
| `max_rounds` | 200 | safety cap on rounds per task | `--max-rounds` |
| `max_idle_rounds` | 5 | end a task after this many idle rounds (0 disables) | – |
| `max_model_calls_per_phase` | 80 | hard cap on the model calls of one grant: a phase, an experiment, an evaluation ([harness.md](harness.md#grants)); a task with model access may set its own for its evaluations (`resources.eval_model_calls_max`) | `--max-calls-per-phase` |
| `max_tokens_per_call` | 64000 | cap on `max_tokens` per model call | – |
| `submit_on_dev_improvement` | true | a round ends as soon as a dev evaluation beats the accepted solution; that evaluated state is submitted | – |
| `submit_grace_calls` | 3 | model calls left after the improving evaluation (notes, `finish`); experiments and evaluations are refused | – |
| `submit_min_gain` | 0.02 | dev gain over the accepted solution that submits on its own, as a share of the task's anchor span; 0: any gain (also the value of a release without the key) | `--submit-min-gain` |
| `unfinished_submit` | `best_dev` | an executor that ends without calling `finish` (call limit, timeout, budget, crash): `best_dev` submits its best dev evaluation that beat the accepted solution, or nothing (its unmeasured changes are discarded, no validation call spent); `workspace` submits the workspace as it was left (a release without the key) | – |
| `posthoc_at_end` | true | when the run is over, score every accepted version on test (`records/posthoc.jsonl`) | – |
| `archive_at_end` | true | then inventory the run's scratch into `records/environment.jsonl` and move it, with every artifact larger than `max_file_mb`, to the archive ([operations.md](operations.md#when-a-run-ends)) | – |
| `max_eval_candidates` | 8 | most candidate directories one dev evaluation call scores, concurrently ([harness.md](harness.md#evaluating-candidates)) | – |
| `exact_context` | false | `true`: refuse a model call (HTTP 400 `inexact_context`) that the adapter would have to lower, e.g. reorder images ([harness.md](harness.md#context-fidelity)); `false`: send it and record the lowerings in `usage.jsonl` | `--exact-context` |
| `phase_timeout_min` | 180 | wall-clock limit of one phase | – |
| `experiment_timeout_min` | 240 | limit of one `run_experiment` | – |
| `max_harness_diff_lines` | 400 | changed harness lines per improver/meta phase; larger edits are reverted | – |
| `max_file_mb` | 5 | larger files are never committed | – |
| `consecutive_crash_rollback` | 2 | crashed rounds in a row before `harness/` is rolled back | – |
| `default_accept_sigma_k` | 2.0 | acceptance rule of mode `base`, and the fallback when the harness rule fails: gain > k × noise sigma | – |
| `default_accept_min_rel` | 0.0 | … and gain > this × \|best\| | – |

Constants without a per-run override change only through a new release (edit the file, commit,
tag `harness-vX.Y.Z`), so that every run made from one tag follows the same protocol. A release
whose protocol file lacks `exact_context` or `max_eval_candidates` runs with its default (`false`,
8; its `run.json` and `protocol_id` stay as that release defines them), and `--exact-context`
applies to it. A release without `posthoc_at_end` / `archive_at_end` does neither at the end of a
run (`rsib posthoc` and `rsib run archive` do them by hand).

### Budget tiers (`[tiers.*]`, per task)

Wall time counts the time the agent can work. Waiting on infrastructure is left out and reported
separately as `queue_s` (in `cost`, `cumulative` and `totals`, and per job in
`experiments.jsonl`): a SLURM job of an experiment or an evaluation pending in the queue (from
`sacct` Submit to Start), and a model endpoint that is unreachable while the adapter waits for it
(`outage_wait_s`). Overlapping waits count once. A job counts as waiting while it is pending, not
only once it is over, so a phase's time limit (`phase_timeout_min`) moves on during the wait: a
phase whose job is still in the queue at its limit is not stopped. GPU hours count only the time
jobs run.

| Tier | `wall_hours` | `gpu_hours` | `usd` | `tokens` | `val_max_calls` | Intended for |
|---|---:|---:|---:|---:|---:|---|
| XS | 0.5 | 0 | 5 | 5M | 30 | smoke tests, toy tasks |
| S | 4 | 8 | 60 | 30M | 40 | cheap tasks, minutes per experiment |
| M | 12 | 64 | 150 | 80M | 30 | vision training / fine-tuning |
| L | 24 | 256 | 300 | 150M | 20 | robotics simulation, generation, multi-GPU |

These are pilot values; calibrate them on your tasks before publishing results.

## Harness hyperparameters (starting values of E0 / I0)

The agent may change these during a run (E by the improver; I by the improver in mode `full`).
To start from different values, edit them on `main` and release a new tag.

### `harness/executor/config.toml` (scope E)

| Key | Default | Meaning |
|---|---|---|
| `max_steps` | 60 | model calls per executor round |
| `max_tokens` | 32000 | output tokens per call (capped by `max_tokens_per_call`) |
| `compact_at_input_tokens` | 180000 | restart the conversation from a summary above this context size |
| `max_nudges` | 2 | replies without a tool call before the round ends |
| `bash_timeout_s` | 900 | timeout of one `bash` tool call |
| `max_tool_output_chars` | 16000 | tool output is clipped to this length |
| `notes_tail_chars` | 12000 | how much of the notes file goes into the round prompt |
| `history_rows` | 30 | earlier rounds shown in the round prompt |

### `harness/improver/config.toml` (scope I)

| Key | Default | Meaning |
|---|---|---|
| `max_steps` | 30 | model calls per improver or meta phase |
| `max_tokens` | 16000 | output tokens per call |
| `compact_at_input_tokens` | 150000 | as above |
| `max_nudges` | 1 | as above |
| `bash_timeout_s` | 300 | as above |
| `max_tool_output_chars` | 16000 | as above |
| `history_rows` | 40 | earlier rounds shown to the improver |
| `meta_every_rounds` | 4 | periodic self-review in mode `full` (0 disables) |
| `accept.sigma_k` | 2.0 | acceptance rule: required gain in task-noise sigmas |
| `accept.min_rel_improvement` | 0.0 | acceptance rule: required relative gain |

## Task and site settings

Task settings (`[rsi]` in `task.toml`: tier, metric, anchors, noise sigma, splits, GPUs, container,
model access) are described in [task-authoring.md](task-authoring.md). Site settings (SLURM
account, partitions, CPU/memory per GPU, container images and flags, the local backend's GPU pool
and `max_parallel`, the gateway's address) and environment variables (`RSIB_RUNS_ROOT`,
`RSIB_DATA_ROOT`, `RSIB_ARTIFACT_ROOT`, ...) are described in [operations.md](operations.md).

## Collecting results across runs

1. **One run**: `rsib report <worktree>` (markdown) or `--format json`; it reads only `records/`.
   Run `rsib posthoc <worktree>` first to get test AUC and the false-acceptance rate.
2. **Many runs**: `rsib results append --ref <branch>` summarizes a run branch into `runs.jsonl` on
   the `results` branch of your clone, one line per indexed commit
   ([the summary row](results.md#the-summary-row)).
3. **The comparison table**:

```bash
rsib results table                         # from the results index (local or origin/results)
rsib results table --branches 'run/*'      # straight from run branches, no index needed
rsib results table --task vsearch --format csv --output vsearch.csv     # also --model, --base, json
```

   Runs are grouped by task, model, effort, condition, release tag and `protocol_id` (a fingerprint
   of the protocol constants, the budgets and the task versions). The condition is the mode
   (`base`, `fixed-i`, `full`), or `probe:<ref>` for runs that imported an improver. Each cell is the
   mean ± standard error over runs (seeds) of one column ([Table columns](#table-columns)). For each
   run, its latest finished row is used, else its latest row. A second table lists the gains in
   normalized test score:

   - `fixed-i` − `base` (same task, model, effort, release and `protocol_id`);
   - `probe:<ref>` − `fixed-i` (an imported improver versus the original one);
   - `full` − `base`.

   Runs that differ in budgets (e.g. `--budget` overrides), protocol settings or task versions
   have different `protocol_id`s and are never averaged together; see
   [results.md](results.md#which-results-may-be-compared) for which results may be compared.

## Metrics

The normalized score is `(score - baseline) / (reference - baseline)` with the task's anchors: 0 is
the starter, 1 the reference method, and a solution better than the reference scores above 1.
"Average improvement per round" is not reported on purpose: it rewards splitting changes into many
small rounds.

A metric is a function of one run's records that returns a dict of values per task. `rsib report`,
`rsib results append` and `rsib results table --branches` run every registered metric and merge
the dicts of each task next to `task_id`, `tier`, `direction` and `anchors`; the summary lists the
metrics it used under `metrics`. Each key comes from one metric: two metrics returning the same key
is an error that names both.

| Built-in metric | Keys per task |
|---|---|
| `outcome` | `rounds`, `accepted`, `rejected`, `no_change`, `errors` (round decisions); `baseline_val`, `best_val`; `test`, `normalized {val, test}`, `ended_reason` (from `final.jsonl`) |
| `gains` | `first_round_gain`: normalized val gain of round 1 over the starter; `later_gain`: what the later rounds added |
| `auc` | `val_auc`: area under the normalized best-so-far val curve against the share of budget used; `test_auc`: the same for the test score of the current accepted version (needs `rsib posthoc`); `posthoc_points` |
| `acceptance` | `false_acceptance {count, judged}`: accepted on val, but worse on test than the previous accepted version |
| `harness` | `harness_commits {E, I}`, `harness_lines {E, I}` |
| `cost` | `violations`; `cost {wall_s, gpu_hours, usd, input_tokens, output_tokens}`, cumulative; `eval_model {calls, input_tokens, output_tokens, usd}`: the model calls of the val and final test evaluations (usage phases `eval-val`, `eval-test`), which no budget is charged for; input tokens include cache reads and writes, as in `cost`; zeros for tasks without model access |
| `progress` | `late_gain`: normalized best val at the end minus at the last round that ended within the first half of the budget; near 0 when improvement saturated, `null` when the task ended within the first half or has no anchors |

The built-ins always run, so every summary has the keys that
`src/rsibench/schemas/result_row.schema.json` requires.

### Writing a plugin

A plugin is a Python module or `.py` file whose import registers metrics:

```python
# lab_metrics.py
from rsibench.metrics import metric


@metric("best_round")
def best_round(run, task):
    """The round whose candidate set the best val score (0: the starter)."""
    accepted = [r["round"] for r in run.rows("rounds", task["id"]) if r["decision"] == "accept"]
    return {"best_round": accepted[-1] if accepted else 0}
```

`run` holds the run's records, read once: `run.run` is `run.json`, `run.files["<name>"]` the rows
of `records/<name>.jsonl`, and `run.rows("<name>", task_id)` the rows of one task. `task` is the
task's entry in `run.json` (`id`, `tier`, `direction`, `anchors`, `version_hash`, ...).
`rsibench.metrics` also has `normalize(score, anchors)` and `step_auc(points)`. Values must be
JSON; numbers, also inside nested dicts, can be table columns, and a few key names belong to the
table (see [Table columns](#table-columns)). A metric reads the records only, so it applies to
every run branch, including runs made before it was written.

### Loading plugins

`--metrics a,b` (on `rsib report`, `rsib results append` and `rsib results table`) lists the
plugins: module paths (`lab.metrics`, importable from `PYTHONPATH`) or `.py` files. Without the
flag, `RSIB_METRICS` is read from the environment or the dotenv file, in the same format. A plugin
that fails to import is an error naming it.

Rows of the results index keep the metrics computed when they were appended (set `RSIB_METRICS`
when appending to index plugin keys), so `rsib results table` loads plugins only with
`--branches`, where it computes the rows itself.

### Table columns

`rsib results table --column NAME` (repeatable) picks the columns. A column is a numeric key of
the task summaries, nested keys joined with dots: `normalized.test`, `cost.usd`,
`false_acceptance.count`, `late_gain`, or a plugin key; a name without numeric values shows `-`.
The default columns are `normalized_test`, `test`, `best_val`, `first_round_gain`, `later_gain`,
`val_auc`, `test_auc`, `rounds`, `accepted`, `usd` and `hours`, where `normalized_test` =
`normalized.test`, `usd` = `cost.usd` and `hours` = `cost.wall_s` / 3600. The gains table always
compares `normalized_test`.

The table's own fields (`task`, `model`, `effort`, `condition`, `base`, `protocol`, `run_id`,
`seed`, `ended`, `runs`, `seeds`) are not columns. Metric keys cannot take these names or those of
the standard columns `normalized_test`, `usd` and `hours`: `rsib results append` and
`rsib results table --branches` refuse a summary that has one.

```bash
rsib results table --branches 'run/*' --metrics lab_metrics.py \
    --column normalized_test --column late_gain --column best_round --format csv
```
