"""Machine-local settings: API keys and a few RSIB_* defaults, from the environment or a dotenv file.

The orchestrator looks a value up by variable name; a non-empty environment variable wins. Keys
are handed straight to the provider SDK. Values read from the file are never copied into
os.environ, so no process the orchestrator starts (harness phases, experiments, evaluators, the
model CLIs) can inherit them. Variables exported in the shell are kept away from phases,
experiments and evaluators by envs.scrubbed_env; the model CLIs inherit them, minus their own
provider's key variables.

Dotenv files: $RSIB_ENV_FILE alone when set; otherwise ~/.config/rsib/.env, then <repo>/.env
(gitignored), the first file that has the variable wins. Agents with shell access run as your
user and can read files on disk, so prefer the location outside the repository for real runs.

Settings read this way besides API keys: RSIB_RUN_NAMESPACE (run branch prefix),
RSIB_CLAUDE_CONFIG_DIR (claude-cli login profile), RSIB_CODEX_HOME (codex-cli login profile),
RSIB_METRICS (metric plugins of reports and results).
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import repo_root


def env_files() -> list[Path]:
    env = os.environ.get("RSIB_ENV_FILE")
    if env:
        candidates = [Path(env).expanduser()]
    else:
        candidates = [Path("~/.config/rsib/.env").expanduser()]
        try:
            candidates.append(repo_root() / ".env")
        except RuntimeError:
            pass
    return [p for p in candidates if p.is_file()]


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines; `export`, quotes and trailing ` # comments` are allowed, no expansion."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            continue
        if value[:1] in ("'", '"') and value.find(value[0], 1) > 0:
            value = value[1:value.find(value[0], 1)]
        else:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def _from_files(name: str) -> tuple[str, Path] | None:
    for path in env_files():
        try:
            value = parse_env(path.read_text()).get(name)
        except OSError:
            continue
        if value:
            return value, path
    return None


def lookup(name: str) -> str | None:
    """The value of `name`: a non-empty environment variable wins over the dotenv files."""
    if os.environ.get(name):
        return os.environ[name]
    found = _from_files(name)
    return found[0] if found else None


def where(name: str) -> str | None:
    """Where `name` would be read from (for messages; never the value)."""
    if os.environ.get(name):
        return "environment"
    found = _from_files(name)
    return str(found[1]) if found else None
