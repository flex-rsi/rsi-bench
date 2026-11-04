# Harness template (E0 + I0)

This directory is the starting harness that every run branch inherits. During a run the agent
rewrites it; the git history of `harness/` on a run branch is the record of that self-modification.

| Path | Scope | Role |
|---|---|---|
| `run_phase.py` | E | entry point the orchestrator calls for each phase |
| `core/agent.py` | E | generic tool-using loop (append-only history, summary-based compaction) |
| `core/context.py` | E | message content as ordered text and image parts (`text`, `image`, `extend`) |
| `core/tools.py` | E | bash, file tools, view_image, run_experiment, evaluate_dev, check_budget, finish |
| `core/gateway_client.py` | E | HTTP client for the orchestrator gateway (model, experiments, dev evaluation, budget) |
| `core/prompts.py` | E | templating and formatting helpers |
| `executor/` | E | executor prompts (`system.md`, `round_prompt.md`), `config.toml`, `phase.py` |
| `improver/` | I | improver and meta prompts, `config.toml`, `phase.py`, `accept.py` (acceptance rule), `memory/lessons.md` |

Rules the orchestrator enforces from outside (they cannot be changed from here):

- The executor may write only `workspace/<task>/` and `records/notes/`, plus its scratch
  directory `.rsib/scratch/`, which is emptied before every executor phase and never committed.
- The improver may write `harness/` (E always; I only in mode `full`) and `records/notes/`.
- At most `max_harness_diff_lines` changed lines per improver/meta phase.
- Model calls, experiments and dev evaluation go through the gateway, which meters the budget.
- Repeated phase crashes roll `harness/` back to the last state that ran a clean round.

The harness uses the Python standard library only, so it runs under any interpreter the
orchestrator uses. If you change it, keep `python harness/run_phase.py <phase> --context ... --output ...`
working; the context and output formats are documented in `run_phase.py` and
[docs/harness.md](../docs/harness.md).
