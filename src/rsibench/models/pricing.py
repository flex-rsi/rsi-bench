"""Token pricing lookup for the usd budget."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from importlib import resources
from pathlib import Path


@lru_cache(maxsize=None)
def _table() -> dict:
    override = os.environ.get("RSIB_PRICING")
    if override:
        with open(Path(override), "rb") as f:
            return tomllib.load(f)
    text = resources.files("rsibench.models").joinpath("pricing.toml").read_text()
    return tomllib.loads(text)


def price(adapter: str, model: str) -> dict | None:
    table = _table()
    family = "openai" if adapter.startswith("openai") or adapter == "codex-cli" else adapter
    return table.get(family, {}).get(model)


def cost_usd(adapter: str, model: str, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    p = price(adapter, model)
    if p is None:
        return None
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_read_tokens * p.get("cache_read", p["input"])
        + cache_write_tokens * p.get("cache_write", p["input"])
    ) / 1e6
