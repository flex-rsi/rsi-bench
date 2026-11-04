# Run records

Everything needed to analyze a run is in its branch: the git history (lineage) and the
`records/` directory (measurements). Large files live outside git and are referenced by pointers.

## Branch, worktree and private store

```
branch     [<namespace>/]run/<task | stream-<n>tasks-<hash>>/<model>/<mode>/s<seed>-<YYYYMMDD-HHMMSS>
worktree   $RSIB_RUNS_ROOT/<branch with "/" replaced by "--">
private    $RSIB_RUNS_ROOT/_private/<same>/tasks/<id>     evaluator copy of each task package
artifacts  $RSIB_ARTIFACT_ROOT/<same>/<task>/r<NNN>/...    dev evaluations, experiments, logs
           $RSIB_ARTIFACT_ROOT/<same>/<task>/transcripts/  harness phase transcripts
evals      $RSIB_RUNS_ROOT/_private/<same>/evals/<task>/   val and test evaluation outputs (never shown to the agent)
```

The optional namespace (`--namespace`, default `$RSIB_RUN_NAMESPACE`) groups the runs of a person or
team, e.g. `robo/run/...`. The working branch of that team then needs a name inside the namespace
too (`robo/main`), because git cannot hold a branch `robo` next to branches `robo/...`.

The context passes a task's artifact directory to the harness as `paths.artifacts`, and the
starting harness's file tools can read it. The private store is never given to the agent: no path
or content of it reaches the context, the records or the file tools. Phases still run shell
commands as your user; see the [security model](../README.md#security-model-and-limitations).

A branch is cut from a release tag (`harness-v*`) and is never merged.

## Commits

Only the orchestrator commits to a run branch, as `rsib-orchestrator`. Every commit after the
base carries `Rsib-Run` and `Rsib-Scope`, and the other trailers where noted:

| Trailer | Values |
|---|---|
| `Rsib-Run` | the branch name |
| `Rsib-Task` | task id (a round's commits) |
| `Rsib-Round` | round number (a round's commits) |
| `Rsib-Scope` | `init`, `solution`, `revert`, `E`, `I`, `notes`, `records`, `rollback`, `final`, `posthoc` |
| `Rsib-Decision` | `accept` or `reject` (solution and revert commits) |
| `Rsib-Val` | validation score of the candidate (solution commits with a score) |
| `Rsib-Decided-By` | `harness` (the improver's rule), `default` or `orchestrator` (solution commits) |
| `Rsib-Phase` | `executor`, `improver` or `meta` (harness and notes commits) |
| `Rsib-Reverts` | sha of the reverted candidate (revert commits) |
| `Rsib-Target` | commit `harness/` was rolled back to (rollback commits) |

A round produces, in order: notes from the executor, the candidate (`solution`), a `revert` if
rejected, `E`, `I` and notes commits from the improver and meta phases, a `rollback` after
repeated crashes, and finally `records`; a meta phase at a stream's task boundary commits before
the executor.
A round is complete once its `records` commit exists, and pushes happen after that commit. On an
interrupted run, `rsib run start` by default rewinds to the last complete round (logged as
`resume_discard`); with `--close-partial-round` it finishes the interrupted round in place instead.
Either way the records the partial round wrote are kept and its spend counts against the budget
(see [operations.md](operations.md#where-the-orchestrator-runs)).

Useful queries:

```bash
git log --format='%h %(trailers:key=Rsib-Round,valueonly,separator=) %(trailers:key=Rsib-Scope,valueonly,separator=) %s' <base>..<branch>
git log -p <base>..<branch> -- harness/          # every harness edit the agent made
git diff <branch>/<task>/r0 <branch>/<task>/final -- workspace/<task>
```

## Tags

| Tag | Points to |
|---|---|
| `<branch>/<task>/r0` | the starter state (baseline) |
| `<branch>/<task>/r<N>` | the accepted candidate of round N |
| `<branch>/<task>/final` | annotated; the task's end state, with `ended_reason`, `accepted_round`, `best_val`, `baseline_val` and `rounds` in the message |
| `<branch>/final` | the end of the run; use it as `--improver-from` for probe runs or `--executor-from` |

## Record files

All files except `run.json` and `notes/` are append-only JSON Lines. The schemas are in
`src/rsibench/schemas/`. The most important fields are listed below.

### run.json

`run_id`, `branch`, `created_at`, `base {ref, commit}`, `mode`, `seed`,
`tasks[] {id, tier, version_hash, domain, metric, direction, anchors, noise_sigma}`,
`model {adapter, name, spec, auth, effort, params, usd_metering}`, `protocol` (frozen run
constants), `budgets {<task>: {wall_hours, gpu_hours, usd, tokens, val_max_calls, tier}}`,
`improver_source` (probe runs), `harness_tree` (content hash of `harness/` at start),
`sandbox {enabled, read_only, hide, mask, git_view}`, `rsibench_version`.

### rounds.jsonl

| Field | Meaning |
|---|---|
| `task_id`, `round` | round 0 is the baseline |
| `phases.executor/improver/meta` | `status` (`finished`, `stopped`, `crash`, `timeout`, `budget`, `refusal`, `model_error`, `skipped`; `ok` for exit 0 without a status, or an unknown status), `steps`, `summary`, `error`, `warning` (an orchestrator problem the phase ran despite, e.g. no Git history view) |
| `candidate` | `changed`, `commit`, `files_changed`, `dev_score` (dev score of exactly this state, if one was measured; once accepted, the bar of the submit rule) |
| `val` | `status` (`ok`, `error`, `skipped`, `limit`), `score`, `metrics` (scalar; with `[rsi.metric] public`, only those and the primary metric) |
| `best_val_before`, `best_val_after` | best accepted val score before and after the round |
| `decision` | `baseline`, `accept`, `reject`, `no-change`, `error` |
| `decision_reason`, `decided_by` | why, and which rule decided |
| `accepted_commit` | sha of the accepted candidate (or the baseline) |
| `harness_changes.E/I` | `commit`, `lines` changed in the round |
| `violations` | number of violations logged in the round |
| `cost`, `cumulative` | `wall_s` (charged wall time), `gpu_hours`, `usd`, `input_tokens`, `output_tokens`, `queue_s` (time waiting on job queues or an unreachable model, not charged) |
| `budget_used_frac` | share of the binding budget resource used so far |

### experiments.jsonl

One line per `run_experiment` call and per dev, val and final test evaluation (`kind`:
`experiment`, `eval-dev`, `eval-val`, `eval-test`): `exp_id`, `round`, `phase`, `command`, `gpus`,
`backend`, `job_id`, timing, `gpu_hours`, `exit_code`, `status`, `score`, `metrics`, `log_path`,
`artifact_dir`. An experiment's `exp_id` is `<task>-r<NNN>-e<NNN>` (outputs in `r<NNN>/<exp_id>/`),
a val or test evaluation's `<task>-r<NNN>-eval-val` or `-eval-test`. `metrics` keeps the scalar
metrics of the result. For `eval-val` and `eval-test`, `metrics` keeps only the primary metric and
the task's [public ones](task-authoring.md#the-evaluator-contract) if `[rsi.metric] public` is
set, and `log_path` and `artifact_dir` are null: those outputs live in the private store.

`eval-dev` rows come from two sources. The executor's dev evaluations (phase `executor`, `exp_id`
`<task>-r<NNN>-dev<NNN>`, outputs in `r<NNN>/<exp_id>/`) also carry `snapshot_sha`, the content
hash of the scored snapshot (kept in `<artifact_dir>/snapshot/`), so a candidate identical to an
evaluated state keeps its dev score. A call with [candidates](harness.md#evaluating-candidates)
writes one such row per candidate, in request order, and its rows share the call's counter:
`exp_id` `<task>-r<NNN>-dev<NNN>-<id>`, `name` `evaluate dev <id>`. A candidate whose directory
could not be copied, or whose evaluation could not be set up, has `status` `error`, `gpu_hours` 0
and `snapshot_sha` null; any other candidate's row keeps its `snapshot_sha`, also when its
evaluator failed, timed out or never started. Under the
[submit rule](harness.md#gateway-api) the orchestrator scores the accepted solution on dev
whenever the bar is unknown, as in round 0 or after a candidate was accepted without a dev score:
phase `orchestrator`, `exp_id` `<task>-r<NNN>-devref`, outputs in `r<NNN>/eval-dev-reference/`,
no `snapshot_sha`. Its score is the bar.

### usage.jsonl

One line per model call: `task_id`, `round`, `phase`, `model`, `input_tokens`, `output_tokens`,
`cache_read_tokens`, `cache_write_tokens`, `usd`, `stop_reason`, `latency_s`, `lowerings` (the
deterministic changes the adapter made to the context, `[]` when it was delivered as given; see
[harness.md](harness.md#context-fidelity)), and `charged`.

OpenAI-compatible responses may also supply `provider` (the upstream provider, e.g. OpenRouter's
selected provider) and `response_id` (the completion's top-level `id`). Each field is optional
and omitted when absent; records without them keep the same shape, with no added nulls.

`phase` is the consumer of the [grant](harness.md#grants) that made the call:

| `phase` | Process | `round` |
|---|---|---|
| `executor`, `improver`, `meta` | that harness phase | the phase's round |
| `experiment` | a `run_experiment` job | the round of the phase that started it |
| `eval-dev` | a dev evaluation of a task with model access | the round of the evaluation |
| `eval-val`, `eval-test`, `posthoc` | a val, final test or `rsib posthoc` evaluation of such a task | the round whose candidate or accepted version is evaluated |

`charged: false` marks a call that was metered but not charged to the task budget: the calls of
val, test and post-hoc evaluations. Charged calls have no `charged` key. Rebuilding spend from
these rows, as a resume does for an interrupted round, counts charged calls only.

### artifacts.jsonl

Pointers to files outside git: `path`, `bytes`, `sha256` (null for transcripts, large files and
files above 512 MB). `kind`: `transcript` (a harness phase's transcript), `experiment` (each file
of an experiment's output directory), `large_file` (a file over `max_file_mb`, left uncommitted).

### violations.jsonl

`kind`: `path` (out-of-scope or record edits, reverted), `diff_size` (harness diff too large,
reverted), `large_file` (not committed), `crash_rollback` (harness restored after repeated
crashes), `accept_fallback` (the harness acceptance rule failed, so the default rule decided),
`resume_discard` (partial round discarded on resume).

### final.jsonl

One line per task, written after the agent finished all tasks: `ended_reason` (`budget_wall`,
`budget_usd`, `budget_tokens`, `budget_gpu`, `max_rounds`, `val_calls`, `idle`), `rounds`,
`final_commit`, `final_round`, `best_val`, `baseline_val`, `test {status, score, metrics}`,
`normalized {val, test}`, `totals`. `rsib run finalize` appends another line for a task whose
test evaluation failed; the latest line per task counts.

### posthoc.jsonl

The test score of each accepted version (`round`, `commit`, `label`, `status`, `score`,
`metrics`, `val_score`, `evaluated_at`), written at the end of the run (protocol `posthoc_at_end`)
or by `rsib posthoc`. The latest appended row for each `(task_id, round)` counts in reports,
summaries and end-of-run scoring, including when its status is an error.

`rsib posthoc <worktree> --redo 1,6,10` re-scores only the listed accepted rounds, even if they
already have rows. A round that was not accepted is an error; round 0 is the accepted baseline.
Each new row has `redo: true` and `supersedes`, the `evaluated_at` of the previous row for that
task and round (null if there was no previous row or timestamp). `--redo-reason TEXT` adds
`redo_reason`. Old rows are kept byte for byte. Without `--redo`, existing rows are skipped as
before.

Each redo uses a fresh directory in the hidden evaluation store:
`evals/<task>/posthoc-r010-redo1/`, then `posthoc-r010-redo2/`, and so on. Earlier evaluation
outputs and transcripts are preserved.

### environment.jsonl

Written once when a sandboxed run ends (protocol `archive_at_end`, or `rsib run archive`): the
inventory of the run's scratch directory (`inventory.bytes`, `top_level` sizes, `hf_models` with
revisions, `python_packages` with versions, `git_repos` with remote and commit), and where it went
(`archive`, `archived_at`, `delete_after`, `large_artifacts`, `large_artifact_bytes`). It is the
run's statement of what its solution depended on beyond the workspace.

### notes/

The agent's own files (`records/notes/<task>.md` by convention). They are kept when a candidate is
reverted and are shown to later rounds.

## Transcripts

Every model call of a harness phase is appended to
`$RSIB_ARTIFACT_ROOT/<run>/<task>/transcripts/r<NNN>-<phase>.jsonl`; an experiment or an evaluation
writes its own, `transcripts/<consumer>.jsonl` in its output directory (in the private store for
val, test and post-hoc evaluations, which are never shown to the agent). Each line holds the new
messages since the previous call of the same session, the call's `lowerings` and the response
(text, tool calls, raw provider content, usage); the first line of a session also holds the system
prompt and tool names. When the harness changed messages it had already sent in the session (or
put another message where the previous reply stood), the line holds the whole history and
`"history_edited": true`. Images are stored once under
`media/<sha16>.<ext>` next to the transcript and appear in the messages as
`{"type": "image", "mime", "file": "media/<sha16>.<ext>", "bytes"}`, never as base64. The improver
reads the executor's transcript of the round it reviews, and can look at its images with
`view_image`.

## Integrity checks

`rsib validate --ref <branch>` verifies that:

1. only `workspace/`, `harness/` and `records/` differ from the base commit;
2. every record file matches its schema;
3. every JSONL record file is append-only across all commits of the branch;
4. every commit carries `Rsib-Run` (matching the branch) and `Rsib-Scope`.
