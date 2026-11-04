"""Acceptance rule for candidate solutions (scope I).

Input (from the orchestrator): score, best, baseline, direction, noise_sigma, history.
Output: {"accept": bool, "reason": str}. If this raises or returns something else, the
orchestrator falls back to its default rule and logs a violation.
"""

from __future__ import annotations

from pathlib import Path

from harness.core.prompts import load_config

HERE = Path(__file__).resolve().parent


def decide(inp: dict) -> dict:
    cfg = load_config(HERE / "config.toml").get("accept", {})
    k = float(cfg.get("sigma_k", 2.0))
    min_rel = float(cfg.get("min_rel_improvement", 0.0))
    score, best = inp["score"], inp.get("best")
    if best is None:
        return {"accept": True, "reason": "first valid candidate"}
    margin = max(k * float(inp.get("noise_sigma") or 0.0), min_rel * abs(best))
    gain = score - best if inp["direction"] == "maximize" else best - score
    if gain > margin:
        return {"accept": True, "reason": f"gain {gain:.6g} > margin {margin:.6g}"}
    return {"accept": False, "reason": f"gain {gain:.6g} <= margin {margin:.6g}"}
