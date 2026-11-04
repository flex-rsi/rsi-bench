"""The Result contract: what evaluations and experiments return.

An evaluator writes $RSIB_OUTPUT_DIR/reward.json and an experiment writes $RSIB_METRICS_FILE, both
in one form (schemas/result.schema.json):

    {"<metric>": <number>, ...,               top-level keys: metrics (a Harbor reward.json)
     "metrics":   {...any JSON...},           more metrics; nested values are allowed
     "artifacts": [{"path": "frames/0.png", "mime": "image/png", "role": "failure"}, ...],
     "info":      {...any JSON...}}           free-form diagnostics

Artifact paths are relative to the output directory and must stay inside it. `parse_result`
turns such a file into the Result fields {"metrics", "artifacts", "info"}; the evaluator runner
and the gateway add "status", "score" and "detail".

Visibility: the agent receives full Results for the dev split. For the hidden splits (val, test)
the records keep only `public_view`: status, score and scalar metrics (a failure's detail says
where the file breaks the contract, quoting none of it); a task's `[rsi.metric] public` list
narrows the scalar metrics to those keys and the primary metric. Nested metrics, artifacts and
info stay in the private store, because per-example outputs of a hidden split would leak it.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from .records import validate

RESERVED = ("metrics", "artifacts", "info")
HIDDEN_SPLITS = ("val", "test")


def parse_result(raw: object, base_dir: Path, strict: bool = True,
                 hidden: bool = False) -> tuple[dict, list[str]]:
    """A result file's JSON -> ({"metrics", "artifacts", "info"}, problems).

    A schema problem empties the fields if `strict`, else a malformed reserved key stays a plain
    metric (experiment files may be any JSON dict). Missing artifacts are dropped and reported.
    `hidden` problems (val, test) name the path and the broken rule, never a value.
    """
    empty = {"metrics": {}, "artifacts": [], "info": {}}
    if not isinstance(raw, dict):
        return empty, ["the result is not a JSON object"]
    problems = validate("result", raw, values=not hidden)
    reserved = {k for k in RESERVED if k in raw}
    if problems:
        if strict:
            return empty, problems
        reserved = {k for k in reserved if not validate("result", {k: raw[k]})}
    metrics = {k: v for k, v in raw.items() if k not in reserved}
    metrics.update(raw["metrics"] if "metrics" in reserved else {})
    base = Path(base_dir).resolve()
    artifacts = []
    for i, item in enumerate(raw["artifacts"] if "artifacts" in reserved else []):
        path = (base / item["path"]).resolve()
        if base not in path.parents or not path.is_file():
            name = "" if hidden else f"{item['path']!r} "
            problems.append(f"artifacts/{i}: {name}is not a file inside the output directory")
            continue
        mime = item.get("mime") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        artifacts.append({**item, "path": str(path), "mime": mime})
    info = raw["info"] if "info" in reserved else {}
    return {"metrics": metrics, "artifacts": artifacts, "info": info}, problems


def scalars(metrics: dict, keys: list[str] | None = None) -> dict:
    """The metrics that are plain values (what records keep); with `keys`, only those."""
    return {k: v for k, v in (metrics or {}).items()
            if (isinstance(v, (int, float, str, bool)) or v is None)
            and (keys is None or k in keys)}


def public_view(result: dict, public: list[str] | None = None) -> dict:
    """What may leave the private store for a hidden split. `public` (Task.public_metrics: the
    task's public keys and its primary metric) limits the scalar metrics to those keys."""
    return {"status": result["status"], "score": result.get("score"),
            "metrics": scalars(result.get("metrics"), public)}
