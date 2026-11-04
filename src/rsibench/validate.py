"""Integrity checks for a run branch (used by CI on every push to run/** and */run/**).

1. Only workspace/, harness/ and records/ differ from the base commit.
2. records/ files match their JSON schemas.
3. Every records/*.jsonl file is append-only across the branch history.
4. Every commit after the base carries Rsib-Run and Rsib-Scope trailers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import gitops
from .records import JSONL_FILES, validate_records_dir

ALLOWED_PREFIXES = ("workspace/", "harness/", "records/")


def validate_ref(repo: Path, ref: str) -> list[str]:
    errors: list[str] = []
    run_text = gitops.show(repo, ref, "records/run.json")
    if run_text is None:
        return [f"{ref}: records/run.json not found (not a run branch?)"]
    run = json.loads(run_text)
    base = run["base"]["commit"]
    if not gitops.ref_exists(repo, base):
        return [f"base commit {base} is not in this repository (fetch tags/history)"]

    for path in gitops.changed_files(repo, base, ref):
        if not path.startswith(ALLOWED_PREFIXES):
            errors.append(f"frozen path changed: {path}")

    with tempfile.TemporaryDirectory(prefix="rsib-validate-") as tmp:
        gitops.archive_extract(repo, ref, "records", Path(tmp))
        errors += validate_records_dir(Path(tmp) / "records")

    commits = gitops.log_commits(repo, f"{base}..{ref}")
    for c in commits:
        if "Rsib-Run" not in c["trailers"] or "Rsib-Scope" not in c["trailers"]:
            errors.append(f"commit {c['sha'][:10]} ({c['subject'][:60]}) lacks Rsib-* trailers")
        elif c["trailers"]["Rsib-Run"] != run["branch"]:
            errors.append(f"commit {c['sha'][:10]} belongs to run {c['trailers']['Rsib-Run']}")

    for name in JSONL_FILES:
        path = f"records/{name}.jsonl"
        previous = ""
        for c in commits:
            text = gitops.show(repo, c["sha"], path)
            if text is None:
                continue
            if not text.startswith(previous):
                errors.append(f"{path} rewritten in commit {c['sha'][:10]} (records are append-only)")
            previous = text
    return errors
