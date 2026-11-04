# Result format and acceptance

This page defines what counts as a result, how results are summarized and which results may be
compared. The channel for submitting results and having them verified will open as soon as possible.

## What a result is

A result is a **run branch** (`run/...`, or `<namespace>/run/...` for a namespaced run) that

1. was created by `rsib run new` from a release tag (`harness-v*`), recorded in
   `records/run.json` (`base.ref`, `base.commit`);
2. passes `rsib validate --ref <branch>` ([integrity checks](records.md#integrity-checks));
3. is finished: `records/final.jsonl` has a line for every task, and its latest line per task has
   `test.status == "ok"` (use `rsib run finalize` to retry a failed test evaluation);
4. was sandboxed (the default).

Runs that fail any of these are not results. `rsib results append --ref <branch>` adds a run's
summary row to a results index (`runs.jsonl` on a `results` branch of your clone);
`rsib results table` prefers the latest finished row of each run.

## The summary row

`runs.jsonl` holds one JSON object per indexed commit of a run branch, validated against
`src/rsibench/schemas/result_row.schema.json` (`schema_version` "1.0"): `rsib results append`
refuses a row that does not match, and reading the index skips one with a warning. Fields:

| Field | Meaning |
|---|---|
| `run_id`, `ref`, `commit`, `indexed_at` | which run and which commit of its branch |
| `mode`, `model`, `effort`, `seed` | the condition |
| `base` | release tag the run started from |
| `improver_source` | for probe runs, the ref the improver was imported from |
| `protocol_id` | fingerprint of the frozen protocol constants, the budgets and the task versions |
| `model_calls` | lines of `usage.jsonl`: every model call of the run, charged or not |
| `metrics` | the metrics that computed `tasks[]`: the built-ins and any plugins loaded when the row was written ([configuration.md](configuration.md#metrics)) |
| `tasks[]` | per task: `task_id`, `tier`, `direction`, `anchors` and the keys of every metric in `metrics` ([built-in keys](configuration.md#metrics)) |

## A report of every run

```bash
rsib results report --output RESULTS.md                # every run branch, local and remote
```

The report reads each run branch's records (the newest commit of each run) and renders: an
overview with one row per run; per task, a score table with the primary test metric and the
task's secondary test metrics, a process table (rounds, gains, false acceptances, harness commits,
violations, cost, end reason) and the post-hoc test curve of every run's accepted versions; the
cross-run table of `rsib results table`; and the sandbox waivers of the results index.
Posthoc curves and metrics use the latest appended row per `(task_id, round)`, including scores
replaced with [`rsib posthoc --redo`](records.md#posthocjsonl).
Unsandboxed runs are left out unless `--include-unsandboxed`.

## Which results may be compared

Two runs are comparable only if they share the **release tag** (`base`) and the **protocol_id**:
same task versions, same protocol constants, same budgets. `rsib results table` groups by task,
model, effort, condition, release and protocol_id, so runs with different budgets or releases are
never averaged together. Comparisons across models are fine within one protocol_id.

Only sandboxed runs count by default: rows have `sandboxed` (from `run.json`), and
`rsib results table` leaves out rows without it (older rows included) and says how many it left
out; `--include-unsandboxed` brings them back. A run made without the sandbox that has been
reviewed (for example from its transcripts) can be counted as a standard run with
`rsib results append --ref <run> --accept-unsandboxed "<reason>"`: the new row says
`sandboxed: true` and keeps the reason in `sandbox_waiver`, and tables use it instead of the
earlier row. The run's own records stay as they are.

Comparisons between conditions (for example `full` vs `base`) need several seeds per condition; the table reports
mean ± standard error and the number of runs.

## Task outputs

Each task's evaluator writes `reward.json` with the primary metric named in `task.toml`
(`[rsi.metric] primary`) as a finite number. Its other top-level keys are secondary metrics;
`metrics`, `artifacts` and `info` are reserved for richer results (the Result contract,
[task-authoring.md](task-authoring.md#the-evaluator-contract)). A candidate whose evaluation fails,
for any cause the evaluator contract lists, is rejected. Records keep the score and the scalar
metrics; for val and test, only the public ones when the task lists them (`[rsi.metric] public`).
