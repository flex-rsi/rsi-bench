# Contributing

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all,tasks]'
pytest -q                           # offline: unit and end-to-end tests with a scripted mock model
RSIB_TEST_SLURM=1 pytest -m slurm   # one live SLURM job: see tests/test_slurm_live.py
```

The end-to-end tests copy the repository into a temporary directory with its own git history,
data, runs and artifact roots, so they never touch your checkout.

## Adding a task

1. Copy `tasks/_template/` to `tasks/<id>/` and follow [docs/task-authoring.md](docs/task-authoring.md).
2. `rsib task prepare <id>`, then `rsib task calibrate <id>` for the baseline and noise sigma, and
   `rsib task calibrate <id> --workspace <strong solution outside the repo>` for the reference.
3. Do not commit reference solutions, hidden labels or test data: the agent can read every file
   in its worktree, including `tasks/`.
4. Commit the task: a run reads task packages from the release it starts from (`--base`), not from
   the working tree. Then run a short `mock:finish` run with `--base` at that commit to check that
   the package loads and the starter scores on all splits.
5. Describe what the task tests, its data access, hardware and standard run in
   [docs/tasks.md](docs/tasks.md), and open a pull request.

## Adding a model adapter

Implement `ChatModel.complete()` in `src/rsibench/models/` (see `base.py` for the neutral message
format), register it in `models/__init__.py`, and add prices to `models/pricing.toml`, or report the
cost in `ModelResponse.extra["cost_usd"]` and add the adapter to `SELF_PRICED` in `runs.py`;
without either, the usd budget is not enforced. Assistant turns must be replayable unchanged
(`raw`). Build the provider's content shapes with `models/context.py`, keep the order of parts, and
return from `ChatModel.lowerings()` the name of every change your transport has to make to the
canonical context, including those a session it keeps already holds from earlier calls (see the
fidelity table in [docs/harness.md](docs/harness.md#context-fidelity)).

## Adding a metric

Write a plugin ([docs/configuration.md](docs/configuration.md#metrics)): a function of the records
registered with `@metric`, loaded with `--metrics` or `$RSIB_METRICS`. It reads only `records/`,
so it applies to every existing run branch. Make it a built-in in `src/rsibench/metrics.py` only if
every summary row should carry it, and document its keys in
`src/rsibench/schemas/result_row.schema.json`.

## Changing the starting harness or the protocol

Edits to `harness/`, `protocol/` or `tasks/` change what a run measures. Ship them as a new
release: update `CHANGELOG.md`, bump the version in `pyproject.toml` and `src/rsibench/__init__.py`,
and tag the commit `harness-vX.Y.Z`. Compare only runs with the same release and `protocol_id`
([docs/results.md](docs/results.md#which-results-may-be-compared)).

## Comparing an improvement method

Put the method in `harness/improver/` on a branch or tag (`--improver-from` imports only that
directory), then run probe runs with `--improver-from <ref> --mode fixed-i` on held-out tasks. See
[docs/harness.md](docs/harness.md#writing-your-own-improver-method-track).
