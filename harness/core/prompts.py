"""Prompt templating and formatting helpers shared by the executor and the improver."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def render(template: str, mapping: dict) -> str:
    """Replace {{name}} placeholders; unknown names become an empty string."""
    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}",
                  lambda m: str(mapping.get(m.group(1), "")), template)


def load_config(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def read_text(path: Path, default: str = "", tail_chars: int | None = None) -> str:
    try:
        text = path.read_text()
    except OSError:
        return default
    if tail_chars and len(text) > tail_chars:
        return "...[earlier content omitted]...\n" + text[-tail_chars:]
    return text


def fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def direction_text(direction: str) -> str:
    return "higher is better" if direction == "maximize" else "lower is better"


def budget_summary(budget: dict) -> str:
    rem, lim = budget.get("remaining", {}), budget.get("limits", {})
    parts = []
    if lim.get("wall_hours") is not None:
        used_min = lim["wall_hours"] * 60 - rem.get("wall_s", 0) / 60
        parts.append(f"wall {used_min:.1f}/{lim['wall_hours'] * 60:.0f} min")
    if lim.get("usd") is not None and rem.get("usd") is not None:
        parts.append(f"usd {lim['usd'] - rem['usd']:.2f}/{lim['usd']:.2f}")
    if lim.get("tokens"):
        parts.append(f"tokens {(lim['tokens'] - rem.get('tokens', 0)) / 1e6:.2f}M/{lim['tokens'] / 1e6:.1f}M")
    if lim.get("gpu_hours"):
        parts.append(f"gpu {lim['gpu_hours'] - rem.get('gpu_hours', 0):.2f}/{lim['gpu_hours']:.1f} h")
    if lim.get("val_max_calls") is not None:
        parts.append(f"val calls {lim['val_max_calls'] - rem.get('val_calls', 0)}/{lim['val_max_calls']}")
    return " · ".join(parts)


def history_table(history: list[dict], rows: int = 30) -> str:
    if not history:
        return "(no rounds yet)"
    lines = ["| round | decision | val | best after | executor | E/I lines | executor report |",
             "|---:|---|---:|---:|---|---|---|"]
    for h in history[-rows:]:
        hc = h.get("harness_changes") or {}
        report = (h.get("executor_summary") or "").replace("\n", " ").replace("|", "/")[:160]
        lines.append(f"| {h['round']} | {h['decision']} | {fmt(h.get('val'))} | "
                     f"{fmt(h.get('best_val_after'))} | {h.get('executor_status') or '-'} | "
                     f"{hc.get('E', 0)}/{hc.get('I', 0)} | {report} |")
    return "\n".join(lines)
