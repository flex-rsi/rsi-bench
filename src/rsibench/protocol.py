"""Frozen protocol constants and budget resolution."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

from .paths import repo_root

BUDGET_KEYS = ("wall_hours", "gpu_hours", "usd", "tokens", "val_max_calls")
# [run] keys a run may set although its release's protocol file lacks them; a missing key means
# the default. Defaults are not written into run.json, so a release keeps its protocol_id.
RUN_DEFAULTS = {"exact_context": False, "max_eval_candidates": 8, "submit_min_gain": 0.0,
                "unfinished_submit": "workspace"}


def run_constant(protocol: dict, key: str) -> Any:
    """A RUN_DEFAULTS key of a run's [run] constants; its default if the run's release lacks it."""
    return protocol.get(key, RUN_DEFAULTS[key])


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    path = path or repo_root() / "protocol" / "default.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if "run" not in data or "tiers" not in data:
        raise ValueError(f"{path}: protocol needs [run] and [tiers.*] sections")
    return data


def resolve_budget(protocol: dict, tier: str, overrides: dict | None = None) -> dict:
    tiers = protocol["tiers"]
    if tier not in tiers:
        raise ValueError(f"unknown budget tier {tier!r}; known: {sorted(tiers)}")
    budget = {k: tiers[tier][k] for k in BUDGET_KEYS}
    for key, value in (overrides or {}).items():
        if key not in BUDGET_KEYS:
            raise ValueError(f"unknown budget key {key!r}; known: {BUDGET_KEYS}")
        budget[key] = value
    budget["tier"] = tier
    return budget


def parse_budget_overrides(items: list[str] | None) -> dict:
    """Parse CLI overrides such as ["wall_hours=0.1", "usd=1"]."""
    out: dict[str, float | int] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"budget override {item!r} must look like key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        out[key] = int(value) if key in ("tokens", "val_max_calls") else float(value)
    return out


def frozen_run_protocol(protocol: dict) -> dict:
    """The part of the protocol copied into run.json (run-level constants)."""
    return copy.deepcopy(protocol["run"])
