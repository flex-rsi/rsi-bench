"""Improver phases: per-round review (`run`) and the periodic self-review (`run_meta`)."""

from __future__ import annotations

from pathlib import Path

from harness.core.agent import Agent
from harness.core.prompts import (budget_summary, direction_text, fmt, history_table, load_config,
                                  read_text, render)
from harness.core.tools import ToolBox

HERE = Path(__file__).resolve().parent
E_PATHS = ["harness/executor", "harness/core", "harness/run_phase.py", "harness/README.md"]


def _policy(ctx: dict) -> str:
    if ctx["allow"]["edit_improver"]:
        return "In this run you may edit scope I as well."
    return "In this run scope I is frozen: edits to harness/improver/ are reverted."


def _harness_files(root: Path) -> str:
    lines = []
    for p in sorted((root / "harness").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            lines.append(f"- {p.relative_to(root).as_posix()} ({p.stat().st_size} B)")
    return "\n".join(lines)


def _experiments_table(experiments: list[dict]) -> str:
    if not experiments:
        return "(none)"
    lines = ["| id | kind | status | score / metrics | wall s |", "|---|---|---|---|---:|"]
    for e in experiments[-40:]:
        metrics = e.get("score") if e.get("score") is not None else e.get("metrics")
        text = str(metrics).replace("|", "/")[:120]
        lines.append(f"| {e['exp_id']} | {e['kind']} | {e['status']} | {text} | "
                     f"{e.get('wall_s', 0):.0f} |")
    return "\n".join(lines)


def _harness_history(history: list[dict]) -> str:
    rows = ["| round | E lines | I lines | next round decision | next val |", "|---:|---:|---:|---|---:|"]
    for cur, nxt in zip(history, history[1:] + [None]):
        hc = cur.get("harness_changes") or {}
        if not (hc.get("E") or hc.get("I")):
            continue
        rows.append(f"| {cur['round']} | {hc.get('E', 0)} | {hc.get('I', 0)} | "
                    f"{nxt['decision'] if nxt else '(pending)'} | {fmt(nxt.get('val')) if nxt else '-'} |")
    return "\n".join(rows) if len(rows) > 2 else "(no harness changes yet)"


def _evidence_dirs(ctx: dict) -> list[Path]:
    """Transcripts, plus this task's dev evaluations and experiments (when the context has
    paths.artifacts)."""
    paths = ctx["paths"]
    return [Path(p) for p in (paths["transcripts"], paths.get("artifacts")) if p]


def _toolbox(ctx: dict, gateway, cfg: dict, extra_read: list[Path]) -> ToolBox:
    root = Path(ctx["worktree"])
    writable = E_PATHS + ["records/notes"]
    if ctx["allow"]["edit_improver"]:
        writable.append("harness/improver")
    return ToolBox(root=root, writable=writable, gateway=gateway, cwd=root, read_roots=extra_read,
                   experiments=False, bash_timeout_s=int(cfg.get("bash_timeout_s", 300)),
                   max_output_chars=int(cfg.get("max_tool_output_chars", 16000)))


def _agent(gateway, system: str, toolbox: ToolBox, cfg: dict, label: str, ctx: dict) -> Agent:
    return Agent(gateway, system, toolbox, label=label, max_steps=cfg.get("max_steps", 30),
                 max_tokens=cfg.get("max_tokens", 16000),
                 compact_at_input_tokens=cfg.get("compact_at_input_tokens", 150000),
                 max_nudges=cfg.get("max_nudges", 1),
                 call_cap=(ctx.get("rules") or {}).get("max_model_calls_per_phase"))


def _base_mapping(ctx: dict, cfg: dict) -> dict:
    root = Path(ctx["worktree"])
    task, state = ctx["task"], ctx["state"]
    return {
        "round": ctx["round"], "task_id": task["id"], "metric": task["metric"],
        "direction_text": direction_text(task["direction"]),
        "best_val": fmt(state.get("best_val")), "baseline_val": fmt(state.get("baseline_val")),
        "history_table": history_table(ctx.get("history", []), int(cfg.get("history_rows", 40))),
        "harness_files": _harness_files(root),
        "lessons": read_text(root / ctx["paths"]["lessons"], "(empty)"),
        "budget_summary": budget_summary(ctx.get("budget", {})),
        "improver_policy": _policy(ctx),
        "max_diff_lines": ctx["limits"]["max_harness_diff_lines"],
    }


def run(ctx: dict, gateway) -> dict:
    cfg = load_config(HERE / "config.toml")
    last = ctx.get("last_round", {})
    ex = last.get("executor", {})
    mapping = _base_mapping(ctx, cfg)
    mapping.update({
        "ex_status": ex.get("status", "n/a"), "ex_steps": ex.get("steps", 0),
        "ex_summary": (ex.get("summary") or "(none)").strip()[:3000],
        "val_status": last.get("val", {}).get("status"), "val_score": fmt(last.get("val", {}).get("score")),
        "decision": last.get("decision"), "decision_reason": last.get("decision_reason", ""),
        "experiments_table": _experiments_table(last.get("experiments", [])),
        "transcript": last.get("transcript", "(none)"),
    })
    system = render((HERE / "system.md").read_text(), mapping)
    prompt = render((HERE / "reflect_prompt.md").read_text(), mapping)
    toolbox = _toolbox(ctx, gateway, cfg, _evidence_dirs(ctx))
    return _agent(gateway, system, toolbox, cfg, f"improver-r{ctx['round']}", ctx).run(prompt)


def run_meta(ctx: dict, gateway) -> dict:
    cfg = load_config(HERE / "config.toml")
    mapping = _base_mapping(ctx, cfg)
    mapping.update({"trigger": ctx.get("trigger", "periodic"),
                    "harness_history": _harness_history(ctx.get("history", []))})
    system = render((HERE / "system.md").read_text(), mapping)
    prompt = render((HERE / "meta_prompt.md").read_text(), mapping)
    toolbox = _toolbox(ctx, gateway, cfg, _evidence_dirs(ctx))
    return _agent(gateway, system, toolbox, cfg, f"meta-r{ctx['round']}", ctx).run(prompt)
