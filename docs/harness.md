# The harness: phases, context, gateway

`harness/` is ordinary Python and Markdown. During a run it is the agent's to rewrite, within the
scopes below. This page documents the interfaces it must keep, so that both the improver and
people writing their own harness or improver know what the orchestrator expects.

## Architecture

```
┌─ run branch: one git worktree per run ─────────────────────────────────────────────────┐
│ harness/executor/, core/, run_phase.py (E)   harness/improver/ (I)                     │
│ workspace/<task>/  the solution              records/  run.json, *.jsonl, notes/       │
│ src/, protocol/, tasks/, ...  frozen         .rsib/  git-ignored: ctx, out, scratch/   │
└──────▲────────────────────────────────────────────────────────────────┬────────────────┘
       │ each phase runs as a subprocess in its own                     │ after every phase:
       │ process group, with its context file and a                     │ out-of-scope edits,
       │ scrubbed environment, inside the filesystem                    │ edits to records and
       │ sandbox if the run has one; all but accept                     │ oversized harness diffs
       │ get a gateway token of their own                               │ are undone; the rest
       │                                                                │ is committed
┌──────┴────────────────────────────────────────────────────────────────▼────────────────┐
│ orchestrator: src/rsibench, frozen during a run                                        │
│                                                                                        │
│ engine     round 0 scores the starter on val; round k: [meta], executor, enforce, val, │
│            accept, candidate commit then tag (accepted) or revert (rejected),          │
│            improver, [meta], records commit                                            │
│ gateway    one grant per process, fixing its endpoints: executor /v1/complete,         │
│            /v1/experiment, /v1/evaluate (dev only), /v1/budget; improver, meta and     │
│            experiment jobs complete, budget; model-access evaluators complete.         │
│            Meters the budget; records usage, experiments, transcripts                  │
│ evaluator  runs tests/test.sh from a private copy of the task, reads reward.json       │
│            (the Result contract)                                                       │
└────────┬─────────────────────────────┬─────────────────────────────┬───────────────────┘
         ▼                             ▼                             ▼
  model adapters                compute backends              stores outside git
  one canonical context:        local: subprocesses,          $RSIB_DATA_ROOT/<task>: dev, val,
  claude-cli, codex-cli,        GPU leases, max_parallel      test (the harness sees dev only)
  anthropic, openai, mock       slurm: GPU jobs only,         $RSIB_ARTIFACT_ROOT/<run>, read by
  (--auth cli | api)            sbatch, sacct, containers;    the harness: transcripts, logs,
                                CPU jobs and phases           experiments, dev evaluations
                                stay local                    $RSIB_RUNS_ROOT/_private/<run>,
                                                              never shown: task copies, val and
                                                              test outputs

end of rsib run start: each task's final commit is scored on test (records/final.jsonl), every
       accepted version too (records/posthoc.jsonl), and the run's scratch is archived
then:  rsib validate, rsib results append → a results index (runs.jsonl)
                                                any time: rsib report <worktree>
```

The harness reaches the model, experiments and dev evaluation only through the orchestrator's
gateway. Each process that may use it (a phase, an experiment job, the evaluator of a task with
model access) holds a grant of its own: a token that fixes its endpoints, what its calls are
charged to and its call cap, revoked when its work ends. Details, and one executor phase step
by step: [below](#one-executor-phase).

### Rounds

```
round 0   score the starter workspace on the val split (baseline)
round k   [meta]    stream runs: self-review at the first round of every later task (mode full)
          executor  edits workspace/<task>/ using dev feedback and tracked experiments;
                    the round ends as soon as a dev evaluation beats the accepted solution
                    by a clear margin (submit_min_gain), or when the executor finishes
          enforce   edits outside the allowed scope are reverted and logged as violations
          val       the candidate is scored on the hidden validation split
          accept    the improver's acceptance rule (harness/improver/accept.py) decides;
                    in mode base, the protocol's default rule
          commit    candidate commit; a rejected candidate is reverted; accepted rounds are tagged
          improver  edits harness/executor, harness/core (E) and harness/improver (I)
          [meta]    every `meta_every_rounds` rounds (mode full)
          records   one line in records/rounds.jsonl, committed (pushed with --push)
end       budget, max_rounds, val calls or idle rounds exhausted -> tag <branch>/<task>/final
after     the final commit of each task is scored once on the hidden test split
```

The agent sees dev scores freely and val scores through the records. Test scores are produced
only after the agent is done. `rsib posthoc` scores every accepted version on test, which gives
the true improvement curve and the false-acceptance rate. To replace invalid scores,
`rsib posthoc <worktree> --redo 1,6,10` appends new scores for those accepted rounds; the latest
row per task and round counts. See [posthoc records](records.md#posthocjsonl).

### Repository layout

```
agent-rsi-benchmarking/
├── harness/                 the starting harness (E0 + I0), copied into every run branch
│   ├── run_phase.py         entry point the orchestrator calls for each phase
│   ├── core/                agent loop, tools, gateway client, prompt helpers      (E)
│   ├── executor/            executor prompts, config, phase                          (E)
│   └── improver/            improver prompts, config, accept.py, memory/lessons.md  (I)
├── src/rsibench/            the frozen orchestrator and the `rsib` CLI
│   ├── orchestrator/        round engine, model gateway, evaluator runner, scope policy
│   ├── models/              adapters: claude-cli, codex-cli, anthropic, openai(-compatible), mock; pricing
│   ├── compute/             backends: local subprocess, SLURM (+ pyxis/enroot containers)
│   ├── schemas/             JSON Schemas of every record file
│   └── runs.py, posthoc.py, report.py, validate.py, results.py, cli.py
├── protocol/default.toml    frozen protocol constants and budget tiers
├── tasks/                   task packages (Harbor layout + [rsi] section)
│   ├── <task>/              one package per benchmark task (docs/tasks.md)
│   └── _template/           template with notes for vision, 3D, robotics and generation tasks
├── examples/                site profile template; setup of the RoboDojo simulator stack
├── docs/                    task list, configuration, result format, task authoring, records, harness, operations
├── tests/                   unit and end-to-end tests (offline, mock model)
├── .github/workflows/       CI (tests)
├── CHANGELOG.md             release notes (tags harness-vX.Y.Z)
└── CONTRIBUTING.md          adding tasks, adapters, metrics and methods; releasing the harness
```

## Scopes

| Path | Scope | Executor | Improver (`fixed-i`) | Improver / meta (`full`) |
|---|---|---|---|---|
| `workspace/<task>/` | solution | write | – | – |
| `harness/executor/`, `harness/core/`, other `harness/*` | E | – | write | write |
| `harness/improver/` | I | – | reverted | write |
| `records/notes/` | notes | write | write | write |
| `.rsib/scratch/` (`paths.scratch`) | scratch | write | – | – |
| everything else | frozen | reverted | reverted | reverted |

The tools refuse writes outside these paths. The orchestrator also checks `git status` after
every phase and reverts anything out of scope, including changes made through `bash`; the accept
phase may change nothing. If the orchestrator stops before it could revert a change to
`records/run.json`, the next `rsib run` command puts back the committed file, and the run's branch
(and every path named after it) always comes from the worktree's Git metadata, which agents cannot
reach. `.rsib/` is git-ignored and these checks skip it: nothing in it is committed or counts as
a change. The orchestrator empties the scratch directory before every executor phase, so it holds
the candidate variants of one round (see [Evaluating candidates](#evaluating-candidates)).

Sandboxed phases (of runs created with the sandbox's `git_view`, the default) have read-only
access to their run's Git history and tags, including earlier rejected candidates; the context's
`environment.git_history` says whether this phase has it. Use `git log` to find a commit and
`git show <commit>:workspace/<task>/<file>` to inspect it or copy it into the workspace. Only the
orchestrator commits. The view excludes other runs' refs and objects; see the
[filesystem sandbox](operations.md#filesystem-sandbox).

## Phases

The orchestrator runs, from the worktree root:

```bash
python harness/run_phase.py <executor|improver|meta|accept> --context .rsib/ctx-<phase>.json \
       --output .rsib/out-<phase>.json
```

with a scrubbed environment (no credentials) plus:

| Variable | Phases | Meaning |
|---|---|---|
| `RSIB_GATEWAY_URL`, `RSIB_GATEWAY_TOKEN` | executor, improver, meta | the gateway, and the token of the phase's [grant](#grants) |
| `RSIB_PHASE`, `RSIB_ROUND`, `RSIB_TASK`, `RSIB_WORKTREE` | all | where you are |
| `RSIB_DEV_DATA` | executor | dev split directory |
| `RSIB_TRANSCRIPT_DIR` | all | transcripts of the current task |

When the phase process exits or times out, the orchestrator kills its process group, revokes the
phase's token and stops every process that still carries the phase's marker `RSIB_SANDBOX_TOKEN`
in its environment (the starter's `bash` tool runs each command in a process group of its own).
So what the phase left running ends with it instead of changing the workspace after it, and a
process that escaped both (`setsid` and a cleaned environment) gets HTTP 401 from the gateway.
Long jobs belong in `run_experiment`.

### One executor phase

How the orchestrator, the gateway and the executor meet in one phase. Improver and meta phases
start, get a token and end the same way, with only `/v1/complete` and `/v1/budget` and without
scratch directory or dev data; after them the orchestrator enforces their scopes (with the
`max_harness_diff_lines` cap) and commits the harness changes. Accept gets no token.

```
orchestrator                                   executor phase (one subprocess)
────────────                                   ───────────────────────────────
empty .rsib/scratch/, write .rsib/ctx-executor.json, grant token T (consumer executor;
its submit bar is the accepted solution's dev score)
start harness/run_phase.py executor ─────────▶ RSIB_GATEWAY_TOKEN = T; reads its context,
in its own process group, with a timeout,      notes and dev data; writes workspace/<task>/,
inside the sandbox if the run has one          records/notes/ and .rsib/scratch/
gateway, serving T:
  /v1/complete   ◀──────────────────────────── model calls
  /v1/experiment ◀──────────────────────────── a job run in workspace/<task>/ itself
                 (no snapshot), on the run's backend if it asks for GPUs, else locally;
                 the call blocks until the job ends, and the job holds a grant of its own
                 (experiment: complete, budget) until then
  /v1/evaluate   ◀──────────────────────────── the workspace, or several candidates
                 each scored on dev on its own snapshot; the evaluator holds a grant of
                 its own (eval-dev) if the task has model access. Submit rule: the first
                 call that beats the bar submits that snapshot (of candidates, the best
                 improving one); experiments and evaluations then get 409, and T has
                 submit_grace_calls more model calls for notes and finish
  /v1/budget     ◀──────────────────────────── remaining budget
kill the process group, revoke T ◀──────────── exits (status in .rsib/out-executor.json)
and stop the processes that still carry the    or times out
phase's marker (RSIB_SANDBOX_TOKEN)
T's jobs still queued never start; running ones finish (an experiment keeps its own grant),
and an evaluation that ends after the phase submits nothing
then: copy a submitted snapshot over workspace/<task>/ → enforce scopes → commit notes
      → if the solution changed and val calls remain: val (with model access, an eval-val
      grant, metered only) → accept rule → commit the candidate, then tag it (accepted)
      or revert it (rejected) → improver → records
```

### Context (`ctx-<phase>.json`)

```json
{
  "phase": "executor", "round": 3, "mode": "full", "run_id": "run/...", "worktree": "/abs/path",
  "task": {"id": "...", "domain": "...", "instruction": "<markdown>", "metric": "miou",
           "direction": "maximize", "noise_sigma": 0.0, "anchors": {"baseline": 0.6, "reference": 0.94},
           "workspace": "workspace/<id>", "experiment_gpus_max": 0, "model_access": false,
           "dev_data": "/abs/dev"},
  "paths": {"notes": "records/notes/<id>.md", "artifacts": "/abs/.../<id>",
            "lessons": "harness/improver/memory/lessons.md", "records": "records",
            "transcripts": "/abs/.../<id>/transcripts", "scratch": ".rsib/scratch"},
  "state": {"best_val": 0.91, "baseline_val": 0.6, "accepted_round": 2, "val_calls_used": 2,
            "val_calls_max": 15, "accepted_dev": 0.9, "auto_submit_at": 0.9068},
  "environment": {"sandboxed": false, "cache": null, "git_history": false},
  "rules": {"submit_on_dev_improvement": true, "submit_grace_calls": 3,
            "submit_min_gain": 0.02, "max_model_calls_per_phase": 80, "max_rounds": 15},
  "history": [{"round": 1, "decision": "accept", "val": 0.88, "best_val_after": 0.88,
               "executor_status": "finished", "executor_summary": "...",
               "harness_changes": {"E": 12, "I": 0}, "budget_used_frac": 0.1}],
  "budget": {"remaining": {...}, "fractions": {...}, "limits": {...}},
  "allow": {"edit_improver": true},
  "limits": {"max_harness_diff_lines": 400},
  "stream": {"index": 0, "tasks": ["..."]}
}
```

`state.accepted_dev` is the dev score of the accepted solution (null until measured), the bar of
the submit rule, and `state.auto_submit_at` the dev score past which an evaluation submits on its
own (see the gateway API below); `state.val_calls_used` of `state.val_calls_max` validation calls
are spent, one per submission. Only the executor's context has `task.dev_data` and
`paths.scratch` (relative to the worktree, an empty directory when the phase starts).

`environment` says whether the run is sandboxed (`sandboxed`), where its download cache is
(`cache`) and whether the phase can read the run's Git history (`git_history`); the starter
executor prompt explains what that means for the solution.

`rules` carries the protocol limits a phase must plan around: `submit_on_dev_improvement`,
`submit_grace_calls`, `submit_min_gain`, `unfinished_submit`, `max_rounds` and
`max_model_calls_per_phase` (the gateway refuses calls of the phase's grant beyond it). The starter agent loop never plans for more calls than that cap and tells the
model when three calls are left, so a phase is not cut off in the middle of its work.

The improver additionally receives `last_round` (`executor`, `val`, `decision`,
`decision_reason`, `transcript`, `experiments`); meta receives `trigger` (`periodic` or
`task_boundary`); accept receives `accept_input` (`score`, `best`, `baseline`, `direction`,
`noise_sigma`, `history`).

### Output (`out-<phase>.json`)

```json
{"status": "finished", "steps": 17, "summary": "what was done", "tool_calls": {"bash": 4}}
```

`status` is one of `finished`, `stopped`, `budget`, `refusal`, `model_error`, `crash`. For
`accept`: `{"status": "ok", "accept": true, "reason": "..."}`. Without a `status` in the file, a
non-zero exit counts as `crash` and exit 0 as `ok`, and an unknown `status` counts as `ok`; a
phase killed at its timeout (at most `phase_timeout_min`; 2 minutes for accept) counts as
`timeout`. If the accept phase fails, the orchestrator uses its default rule and logs a violation.
Two consecutive rounds with a crashed phase roll `harness/` back to the last state that ran a
clean round.

## Gateway API

All requests are `POST` with `Authorization: Bearer $RSIB_GATEWAY_TOKEN` and a JSON body. The token
names the [grant](#grants) of the calling process, which decides what it may call.

| Endpoint | Body | Returns |
|---|---|---|
| `/v1/complete` | `system`, `messages` (below), `tools`, `max_tokens`, `session` | `text`, `tool_calls`, `stop_reason`, `raw`, `model`, `usage`, `budget` (null when the grant's calls are not charged) |
| `/v1/experiment` | `name`, `command`, `gpus`, `timeout_min` (executor only) | `exp_id`, `status`, `exit_code`, `wall_s`, `gpu_hours`, `metrics`, `artifacts`, `info`, `detail`, `log_tail`, `artifact_dir`, `budget` |
| `/v1/evaluate` | `split: "dev"`, optional `candidates` (executor only) | `status`, `score`, `metric`, `direction`, `metrics`, `artifacts`, `info`, `detail`, `log_tail`, `artifact_dir`; with `candidates`, `results` instead (those fields per candidate id, [Evaluating candidates](#evaluating-candidates)); `budget`; `submit_required` and `message` when a result is submitted, else `submit_rule` (`accepted_dev`, `auto_submit_at`) while the bar is known (below) |
| `/v1/budget` | `{}` | `remaining`, `fractions`, `used`, `limits` |

Experiments and evaluations return the [Result contract](task-authoring.md#the-evaluator-contract)
(`src/rsibench/contracts.py`): `metrics` may be nested, and `artifacts` lists files (`path`,
`mime`, `role`, `note`) under the task's artifact directory, which the context passes as
`paths.artifacts` and the starting harness adds to its readable roots.

With `rules.submit_on_dev_improvement` (protocol `submit_on_dev_improvement`), the first dev
evaluation whose score beats `state.accepted_dev` by more than the margin (`submit_min_gain` times
the task's anchor span |reference − baseline|, so past `state.auto_submit_at`) ends the executor's
round: its reply adds `submit_required: true` and a `message`, the orchestrator submits exactly
that evaluated snapshot (of a call with candidates, the best candidate past the line), further
experiments and evaluations are refused, and the phase has `submit_grace_calls` model calls left
to update its notes and call `finish`. An evaluation short of the line ends nothing; its reply
carries `submit_rule` (`accepted_dev`, `auto_submit_at`), and the executor may go on working or
call `finish`, which submits the workspace as it is. A release without `submit_min_gain` submits on
any gain. An executor that ends without calling `finish` (its status is not `finished`: the call
limit, a timeout, the budget, a crash) submits, with `rules.unfinished_submit` = `best_dev`, only
the best of its dev evaluations that beat `state.accepted_dev` (that snapshot replaces the
workspace); if none did, its changes to the workspace are discarded and no validation call is
spent. With `workspace` (a release without the key) the workspace is submitted as it was left.

Errors come back as `{"error": "<code>", "message": "..."}`. HTTP 401 is an unknown or revoked
token (`unauthorized`). 402 means a limit is used up: the budget (`budget_exhausted`), the grant's
model-call cap (`phase_call_limit`) or the grace calls after a submission (`submit_required`). 403
is an endpoint outside the grant, or a split other than dev (`forbidden`); 409 an experiment or
evaluation after a submission (`submit_required`); 502 a provider error (`model_error`). A model
call is refused with HTTP 400 when:

| `error` | Cause |
|---|---|
| `bad_request` | `system`, `messages` or `tools` is missing; `messages` breaks the message schema (`message` lists the first errors by path, e.g. `3/content/1/mime`); an image's `data` is not base64 of its `mime` format |
| `inexact_context` | the run sets `exact_context` and the adapter would lower this context (see below) |

### Grants

Every process that may use the gateway holds a grant of its own, named by its token. A grant fixes
the endpoints the process may call, the budget its model calls are charged to and its transcript,
and counts its own model calls against the protocol cap `max_model_calls_per_phase` (the grant of
an evaluation: its task's `resources.eval_model_calls_max`, if the task sets one). When the
process's work ends the orchestrator revokes the grant: its token gets HTTP 401 from then on, and
jobs it requested that are still waiting to start never start.

| Consumer (`phase` of its usage rows) | Process | Endpoints | Model calls charged to | Transcript | Revoked when |
|---|---|---|---|---|---|
| `executor` | the executor phase | complete, experiment, evaluate, budget | the task budget | `transcripts/r<NNN>-executor.jsonl` | its process group is killed |
| `improver`, `meta` | that phase | complete, budget | the task budget | `transcripts/r<NNN>-<phase>.jsonl` | its process group is killed |
| `experiment` | one `run_experiment` job | complete, budget | the task budget | `transcripts/experiment.jsonl` in the job's artifact directory | the job ends |
| `eval-dev` | one dev evaluation of a task with model access: a candidate, the workspace, the accepted solution | complete | the task budget | `transcripts/eval-dev.jsonl` in its output directory | the evaluation ends |
| `eval-val`, `eval-test`, `posthoc` | a val, final test or `rsib posthoc` evaluation of such a task | complete | nothing (metered, `charged: false`) | `transcripts/<consumer>.jsonl` in its output directory, in the private store | the evaluation ends |

The accept phase gets no token. Grants are independent: concurrent experiments and candidates
never share a call counter, and a CLI adapter keeps the sessions of each grant apart, also under the
same `session` label. The executor's grant also holds the submit rule's state (the bar, the
submission and its grace calls), so a call that outlived its phase submits nothing. Which tasks
give their evaluator a grant is part of the task
([task-authoring.md](task-authoring.md#solutions-that-call-the-model)); how the gateway is reached
from other machines is part of the site profile ([operations.md](operations.md#site-profile)).

### Messages: the canonical context

`messages` is the whole conversation, oldest first:

```json
{"role": "user", "content": <content>}
{"role": "assistant", "text": "...", "tool_calls": [{"id": "...", "name": "...", "arguments": {}}], "raw": <opaque>}
{"role": "tool_results", "results": [{"tool_call_id": "...", "content": <content>, "is_error": false}]}
```

A `<content>` is a string, or an ordered list of parts (a string is one text part):

```json
{"type": "text", "text": "..."}
{"type": "image", "mime": "image/png", "data": "<base64>"}
```

- **Order is the contract.** Every adapter delivers the messages, and the parts of each content,
  in list order, so one context means the same to every model. Adjacent text parts read as one
  text. A tool result with no content (an empty string or list, or only empty text) is
  `(no output)` in a provider's native tool result, and an empty `<tool_result>` body where
  results travel as text (see Context fidelity).
- **Images**: `mime` is `image/png`, `image/jpeg`, `image/gif` or `image/webp` and must match the
  bytes; `data` is standard base64 without line breaks, at most 7,000,000 characters, which holds
  an image of up to 5 MiB. A provider's own limits (size, pixels) can be lower; a call it refuses
  comes back as a provider error (HTTP 502). Images go in user messages and tool results, inline:
  the gateway never reads a file on the harness's behalf.
- The schema is `src/rsibench/schemas/messages.schema.json`; parts take no other keys.
- `harness/core/context.py` builds content: `text(s)`, `image(path)` (checks the extension, the
  format and the 5 MiB limit) and `extend(content, parts)`. `Agent.run` takes a string or a list
  of parts as the prompt. The `view_image` tool, which the executor and the improver both have,
  returns `[text "<path> (<bytes> B)", image]` as its result: plots, rendered frames, predicted
  masks, transcript media.

Send `raw` back unchanged and never edit earlier messages. To shrink the context, start a new
conversation (a new `session` label) from a summary, as `harness/core/agent.py` does.
`stop_reason` is one of `end_turn`, `tool_use`, `max_tokens`, `refusal`, `pause_turn`, `other`.
Tools are neutral specs: `{"name", "description", "input_schema"}`.

The starting harness asks for a text summary. If the model instead puts state in tool calls,
compaction preserves those calls as text without executing them, together with recent tool
results. An empty summary also falls back to recent tool results. A plain text summary follows
the same restart path as before. Both are bounded: each string is cut to its first and last 2000
characters, anything nested deeper than 20 levels is replaced by a note, the preserved calls are
cut to 16000 characters in all, and the digest keeps the last six results (ids and names cut to
200 characters, 32000 characters in all), labelled as data rather than instructions, with images
and other non-text parts left out.

### Context fidelity

An adapter that cannot deliver a context as given applies a named, deterministic *lowering* and
reports it; no adapter changes content or order silently. The CLI adapters keep the conversation in
a CLI session and send each call's delta: the user-side messages after the last assistant reply,
items separated by a blank line, each tool result wrapped as
`<tool_result id="...">` ... `</tool_result>` (`<tool_result id="..." (error)>` for an error).
A call continues the session only when the session holds exactly the messages before its last
reply; otherwise (no session yet, earlier messages changed, or assistant turns the session did
not produce) it starts a new session and sends the whole history.
An `openai` tool message holding an error result starts with `ERROR: `.

| Adapter | Lowering | When | What the model receives |
|---|---|---|---|
| `anthropic` | – | – | images in user content and in tool results, in place |
| `claude-cli` | – | – | the delta as content blocks on stdin (stream-json), images in place |
| `openai` | `tool_results_as_user_message` | a tool result holds an image | one stub tool message per call (`(output in the next user message)`), then one user message with all results of that turn in order, wrapped as in the delta, images in place |
| `codex-cli` | `images_first` | the delta holds an image | every image first (`--image` files, in order), then the text with `[Image #k]` where image k stood |
| `claude-cli`, `codex-cli` | `history_replayed` | a call starts a session with a history that holds assistant turns: a conversation sent under a new session (a few-shot demonstration, a task's solution that starts a session on every call), one rebuilt after a compaction that keeps earlier turns, or one whose earlier messages changed | the whole history in order in one message: the note `[The conversation so far, in order. ...]`, user content and tool results as in the delta, each assistant turn as `<assistant_turn>` text with its tool calls as `<tool_call id="..." name="...">` arguments as JSON `</tool_call>` |
| `claude-cli`, `codex-cli` | `continue_prompt` | nothing follows the last assistant turn | the text `Continue.` |
| `mock` | – | – | – |

A CLI session keeps what it received, so a lowering applies to the call that introduces it and
to every later call of that session. The lowerings of every call are recorded in
`records/usage.jsonl` (`lowerings`; `[]` means the context was delivered as given) and in its
transcript line. A run created with `--exact-context` (protocol `exact_context = true`) refuses
such a call with HTTP 400 `inexact_context` before anything reaches the model, so every model in a
comparison sees exactly the context the harness built.

Lowerings concern the messages. How tools travel is the transport's own: native tool calls for
the API adapters; for the CLI adapters, tool descriptions in the system prompt and replies forced
into a JSON schema.

Transcripts keep each image once as a file under `media/`, never as base64
([records.md](records.md#transcripts)).

### Experiments

`run_experiment` runs `bash -c <command>` in `workspace/<task>/`, on the run's backend when it asks
for GPUs (`gpus`, at most `task.experiment_gpus_max`, else HTTP 400) and locally otherwise, with:

- `RSIB_DEV_DATA`: the dev split;
- `RSIB_ARTIFACT_DIR`: a fresh directory outside git for large outputs;
- `RSIB_METRICS_FILE`: write a JSON dict here: metric keys plus the optional `metrics`,
  `artifacts` (paths relative to `RSIB_ARTIFACT_DIR`) and `info` of the Result contract. It is
  returned, and its scalar metrics are recorded. A reserved key of another type is kept as a
  plain metric and named in `detail`;
- `RSIB_WORKSPACE`: the workspace path;
- `RSIB_PYTHON`: the Python interpreter (the orchestrator's, or `python` inside a container image);
- `RSIB_GATEWAY_URL`, `RSIB_GATEWAY_TOKEN`: the job's own [grant](#grants): until the job ends it
  may call `/v1/complete` (the request above) and `/v1/budget`, charged to the task budget like the
  phase's own calls;
- `CUDA_VISIBLE_DEVICES`: set by the backend to the GPUs it gave the job.

The call blocks until the job ends, and processes the job leaves running are stopped with it. GPU
hours are charged after the job, for its running time. The timeout (`timeout_min`, default 60) is
capped by the protocol's `experiment_timeout_min` and by the remaining wall-clock and GPU budget,
and counts from the job's start: waiting for a free slot or GPU of the
[local backend](operations.md#local-backend), or in the SLURM queue, is not part of it. A job
still waiting when its phase ends is not started: an experiment is recorded with `status`
`cancelled`, a dev evaluation as an `error` that says so, and neither is charged.

### Evaluating candidates

`/v1/evaluate` with `candidates` scores several complete solutions in one call, for example
variants the executor wrote to its scratch directory:

```json
{"split": "dev", "candidates": [{"id": "lr-small", "path": ".rsib/scratch/lr-small"},
                                {"id": "lr-large", "path": ".rsib/scratch/lr-large"}]}
```

- A call takes at most `max_eval_candidates` candidates (protocol, default 8). An `id` matches
  `[A-Za-z0-9][A-Za-z0-9._-]{0,31}` and is unique in the call, ignoring case. A `path` is
  absolute or relative to the worktree, and must resolve (symlinks resolved) to a directory
  inside `workspace/<task>/` or the scratch directory. Anything else is HTTP 400 `bad_request`.
- Each candidate is scored on a snapshot of its own directory, without caches and files over
  `max_file_mb`, as a call without `candidates` scores the workspace's. The evaluations run
  concurrently, within the limits of the backend: the local backend runs at most
  `[local] max_parallel` jobs at once and gives each job its own GPUs.
- The budget is checked once, before the call. Each evaluation is charged its own GPU hours and
  recorded as its own `eval-dev` row ([records.md](records.md#experimentsjsonl)).
- The reply is `{"results": {"<id>": {...}, ...}, "budget": {...}}`: per id, in request order, the
  fields of a single evaluation. A candidate whose directory cannot be copied (for example because
  of a dangling symlink), or whose evaluation cannot be started, gets `status` `error` with the
  reason in `detail` and costs nothing. An evaluation that ran is charged like any other, also when
  its evaluator failed or timed out. One candidate's error leaves the others' results intact.
- Under the submit rule, the best candidate past `state.auto_submit_at` (by the task's
  direction; the first in request order among equal scores) is submitted as a single improving
  evaluation would be: the reply adds `submit_required` and `message`, and after the phase that
  candidate's evaluated snapshot replaces the committable files of `workspace/<task>/`.

The starting harness's `evaluate_dev` tool takes the same `candidates` and returns the reply
without `budget`. To keep every candidate's result in its output, as valid JSON, it shortens log
tails and, if the reply is still too long, reduces the largest results to `status`, `score`,
`detail` and `artifact_dir`, where the evaluator's files stay. The executor's file tools may write
the scratch directory.

## Writing your own improver (method track)

To compare an improvement method under this protocol, put it in `harness/improver/`, commit it on
a branch or tag, and run probe runs with `--improver-from <ref>` in mode `fixed-i`. The option
imports only `harness/improver/`; the rest of `harness/` comes from the run's base release, whose
`run_phase.py` calls `run(ctx, gateway)` (improver) and `run_meta(ctx, gateway)` (meta) of
`harness/improver/phase.py` and `decide(accept_input)` of `harness/improver/accept.py`. Keep these
entry points and the context/output formats above, use only the gateway for model calls, and keep
the standard-library-only rule, or document extra dependencies and install them in the environment
that runs the orchestrator.
