"""Who may change what, and the default acceptance rule.

Scopes (by path in a run worktree):
    workspace/<task>/     solution   the task solution
    harness/improver/     I          improver prompts, config, acceptance rule, memory
    harness/              E          everything else in the harness (executor, core loop, tools)
    records/notes/        notes      free-form agent notes
    anything else         frozen     records, tasks, framework code, CI ... (always reverted)

Scope enforcement skips git-ignored paths (.rsib/, build/, caches): they are never committed and
never reverted.
"""

from __future__ import annotations

PHASE_SCOPES = {
    "executor": {"solution", "notes"},
    "improver": {"E", "I", "notes"},
    "meta": {"E", "I", "notes"},
    "accept": set(),
}


def classify(path: str, task_id: str | None = None) -> str:
    if path.startswith("workspace/"):
        if task_id is None or path.startswith(f"workspace/{task_id}/"):
            return "solution"
        return "frozen"
    if path.startswith("harness/improver/"):
        return "I"
    if path.startswith("harness/"):
        return "E"
    if path.startswith("records/notes/"):
        return "notes"
    return "frozen"


def allowed_scopes(phase: str, mode: str) -> set[str]:
    scopes = set(PHASE_SCOPES[phase])
    if mode == "fixed-i":
        scopes.discard("I")
    if mode == "base" and phase in ("improver", "meta"):
        scopes = {"notes"}
    return scopes


def default_accept(score: float, best: float | None, direction: str, sigma: float,
                   sigma_k: float, min_rel: float) -> tuple[bool, str]:
    """Accept when the candidate beats the best accepted score by more than the noise margin."""
    if best is None:
        return True, "no accepted score yet"
    margin = max(sigma_k * sigma, min_rel * abs(best))
    gain = (score - best) if direction == "maximize" else (best - score)
    if gain > margin:
        return True, f"improves best by {gain:.6g} > margin {margin:.6g}"
    return False, f"gain {gain:.6g} does not exceed margin {margin:.6g}"
