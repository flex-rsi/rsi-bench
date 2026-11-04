"""Executor phase: work on the task for one round."""

from __future__ import annotations

from pathlib import Path

from harness.core.agent import Agent
from harness.core.prompts import (budget_summary, direction_text, fmt, history_table, load_config,
                                  read_text, render)
from harness.core.tools import ToolBox

HERE = Path(__file__).resolve().parent


def _model_access_note(task: dict) -> str:
    if task.get("model_access"):
        return ("This task gives the solution model access: during evaluation it may call the "
                "run's model through the gateway the evaluator hands it (see the task instruction), "
                "so letting the model look at an input and answer is a valid solution; the calls "
                "are metered and capped.")
    return ("This task gives the solution no model access: during evaluation it runs as "
            "self-contained code (it may load open models from the workspace or the run cache).")


def _environment_note(env: dict) -> str:
    note = ""
    if env.get("sandboxed"):
        note += ("- Sandbox: this phase, your experiments and the evaluator's run of your solution "
                 "see only the worktree, the dev data, the system software and the run cache "
                 f"`{env.get('cache')}` ($RSIB_CACHE); everything else on the machine is hidden. "
                 "Put downloaded weights and any Python environment you build in $RSIB_CACHE "
                 "(HF_HOME, TORCH_HOME and PIP_CACHE_DIR already point there). Evaluation sees the "
                 "workspace and $RSIB_CACHE read-only, so the solution must not depend on anything "
                 "else.\n")
    if env.get("git_history"):
        note += ("- Git history: this phase has read-only access to this run's commits and tags. "
                 "Use `git log` and `git show <commit>:workspace/<task>/<file>` to inspect or "
                 "recover earlier versions. Only the orchestrator commits.\n")
    if env.get("phase_cpus") or env.get("cpu_jobs") not in (None, "local"):
        cores = f"{env['phase_cpus']} CPU cores" if env.get("phase_cpus") else "a few CPU cores"
        where = ("as a job on a CPU node" if env.get("cpu_jobs") not in (None, "local")
                 else "as a separate job")
        note += (f"- Compute: this phase shares {cores} of a login machine with other users. Run "
                 "anything CPU- or GPU-heavy (inference over many clips, feature extraction, "
                 "sweeps, builds) with `run_experiment`, not from the shell: with `gpus` 0 it runs "
                 f"{where}, with `gpus` >= 1 on a GPU node. Keep shell commands light.\n")
    return note


def _submissions_note(ctx: dict) -> str:
    """What a submission costs and when a dev evaluation submits on its own."""
    rules, state, task = ctx.get("rules") or {}, ctx.get("state") or {}, ctx["task"]
    limits = (ctx.get("budget") or {}).get("limits") or {}
    cap = state.get("val_calls_max") or limits.get("val_max_calls")
    lines = []
    if cap:
        lines.append(f"- Validation submissions used: {state.get('val_calls_used', 0)} of {cap}. A "
                     "round that ends with a changed solution spends one; a round that leaves it "
                     "unchanged spends none. The task ends when they run out.")
    if rules.get("max_rounds"):
        lines.append(f"- This is round {ctx['round']} of at most {rules['max_rounds']}.")
    if not rules.get("submit_on_dev_improvement"):
        return "\n".join(lines)
    if rules.get("unfinished_submit") == "best_dev":
        lines.append("- If this phase ends before you call `finish` (call limit, time limit), only "
                     "the best state a dev evaluation scored above the accepted solution is "
                     "submitted; changes no dev evaluation measured are discarded.")
    at, bar = state.get("auto_submit_at"), state.get("accepted_dev")
    side = "above" if task["direction"] == "maximize" else "below"
    if at is None:
        lines.append("- Automatic submission: a dev evaluation that beats the accepted solution's "
                     "dev score ends the round and submits that state.")
    elif rules.get("submit_min_gain"):
        lines.append(f"- Automatic submission: a dev evaluation scoring {side} {fmt(at)} ends this "
                     f"round and submits exactly that state. The accepted solution scores "
                     f"{fmt(bar)} on dev; the line asks for a clear gain "
                     f"({rules['submit_min_gain']:.0%} of the gap between the starter and the "
                     "reference). A smaller gain does not end the round: keep improving toward the "
                     "line, or, if the gain is worth one of your submissions, leave the workspace "
                     "in that state and call `finish`.")
    else:
        lines.append(f"- Automatic submission: a dev evaluation scoring {side} {fmt(at)} (the "
                     "accepted solution's dev score) ends this round and submits exactly that state.")
    return "\n".join(lines)


def run(ctx: dict, gateway) -> dict:
    cfg = load_config(HERE / "config.toml")
    root = Path(ctx["worktree"])
    task, paths, state = ctx["task"], ctx["paths"], ctx["state"]
    notes_path = root / paths["notes"]
    if not notes_path.exists():
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(f"# Notes for {task['id']}\n\n")
    mapping = {
        "round": ctx["round"],
        "task_id": task["id"],
        "instruction": task["instruction"].strip(),
        "workspace": task["workspace"],
        "dev_data": task.get("dev_data", "n/a"),
        "notes_file": paths["notes"],
        "worktree": str(root),
        "environment_note": _environment_note(ctx.get("environment") or {}),
        "model_access_note": _model_access_note(task),
        "scratch": paths.get("scratch", "n/a"),
        "notes": read_text(notes_path, "(empty)", int(cfg.get("notes_tail_chars", 12000))),
        "lessons": read_text(root / paths["lessons"], "(none)"),
        "metric": task["metric"],
        "direction_text": direction_text(task["direction"]),
        "baseline_val": fmt(state.get("baseline_val")),
        "best_val": fmt(state.get("best_val")),
        "accepted_round": fmt(state.get("accepted_round")),
        "gpus_max": task.get("experiment_gpus_max", 0),
        "history_table": history_table(ctx.get("history", []), int(cfg.get("history_rows", 30))),
        "budget_summary": budget_summary(ctx.get("budget", {})),
        "submissions": _submissions_note(ctx),
    }
    system = render((HERE / "system.md").read_text(), mapping)
    prompt = render((HERE / "round_prompt.md").read_text(), mapping)
    # dev data, plus this task's outputs: dev evaluations, experiments, transcripts
    read_roots = [Path(p) for p in (task.get("dev_data"), paths.get("artifacts")) if p]
    writable = [p for p in (task["workspace"], "records/notes", paths.get("scratch")) if p]
    toolbox = ToolBox(root=root, writable=writable, gateway=gateway,
                      cwd=root / task["workspace"], read_roots=read_roots, experiments=True,
                      bash_timeout_s=int(cfg.get("bash_timeout_s", 900)),
                      max_output_chars=int(cfg.get("max_tool_output_chars", 16000)))
    agent = Agent(gateway, system, toolbox, label=f"executor-r{ctx['round']}",
                  max_steps=cfg.get("max_steps", 60), max_tokens=cfg.get("max_tokens", 32000),
                  compact_at_input_tokens=cfg.get("compact_at_input_tokens", 180000),
                  max_nudges=cfg.get("max_nudges", 2),
                  call_cap=(ctx.get("rules") or {}).get("max_model_calls_per_phase"))
    return agent.run(prompt)
