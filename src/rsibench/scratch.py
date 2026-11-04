"""What agent code leaves behind, and how it is retired when a run ends.

During a sandboxed run, everything agent code creates outside its workspace lives in the run's
scratch directory (`$RSIB_SCRATCH_ROOT/<run>/`: HOME, the download cache `$RSIB_CACHE`, tmp).
When the run is over (final test and posthoc done) the orchestrator

1. inventories the scratch (downloaded Hugging Face models, Python packages, git checkouts, sizes)
   into `records/environment.jsonl`, so the run states what its solution depended on;
2. moves the scratch, and every artifact larger than the protocol's `max_file_mb` (transcripts
   excepted), into `$RSIB_ARCHIVE_ROOT/<run>__<time>/`, which no agent can see;
3. leaves the archive for review until its `delete_after` date; `rsib archive purge` (also run by
   `rsib run new` / `rsib run start`) deletes expired archives.

So a later run, by any agent, starts without the weights, environments and code of earlier ones.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_RETENTION_DAYS = 7
MANIFEST = "ARCHIVE.json"


def _size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _git_head(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
    except OSError:
        return None
    if text.startswith("ref: "):
        ref = repo / ".git" / text[5:]
        try:
            return ref.read_text().strip()
        except OSError:
            packed = repo / ".git" / "packed-refs"
            for line in packed.read_text().splitlines() if packed.exists() else []:
                if line.endswith(" " + text[5:]):
                    return line.split()[0]
            return None
    return text


def _git_remote(repo: Path) -> str | None:
    try:
        lines = (repo / ".git" / "config").read_text().splitlines()
    except OSError:
        return None
    in_origin = False
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            in_origin = line.startswith('[remote "')
        elif in_origin and line.startswith("url"):
            return line.split("=", 1)[1].strip()
    return None


def inventory(scratch: Path, max_depth: int = 8) -> dict:
    """Downloaded models, packages and repositories found in a run's scratch directory."""
    scratch = Path(scratch)
    out = {"bytes": 0, "top_level": [], "hf_models": [], "python_packages": [], "git_repos": []}
    if not scratch.is_dir():
        return out
    for child in sorted(scratch.iterdir()):
        size = _size(child) if child.is_dir() else child.stat().st_size
        out["top_level"].append({"name": child.name, "bytes": size})
        out["bytes"] += size
    seen_pkgs = set()
    for root, dirs, _files in os.walk(scratch):
        here = Path(root)
        if len(here.relative_to(scratch).parts) > max_depth:
            dirs[:] = []
            continue
        for d in list(dirs):
            path = here / d
            if d.startswith(("models--", "datasets--")) and (path / "snapshots").is_dir():
                kind, _, rest = d.partition("--")
                revs = sorted(p.name for p in (path / "snapshots").iterdir())
                out["hf_models"].append({"repo": rest.replace("--", "/"), "type": kind[:-1],
                                         "revisions": revs, "bytes": _size(path)})
                dirs.remove(d)
            elif d.endswith(".dist-info"):
                name, _, version = d[: -len(".dist-info")].partition("-")
                key = (name.lower(), version)
                if key not in seen_pkgs:
                    seen_pkgs.add(key)
                    out["python_packages"].append({"name": name, "version": version})
                dirs.remove(d)
            elif (path / ".git").is_dir():
                out["git_repos"].append({"path": str(path.relative_to(scratch)),
                                         "remote": _git_remote(path), "commit": _git_head(path)})
    out["python_packages"].sort(key=lambda p: (p["name"].lower(), p["version"]))
    return out


def archive(scratch: Path, art_root: Path, run_id: str, slug: str, max_file_mb: float,
            retention_days: float = DEFAULT_RETENTION_DAYS, root: Path | None = None) -> dict:
    """Move a finished run's scratch and its large artifacts into the archive root."""
    from .paths import archive_root

    now = datetime.now(timezone.utc)
    dest = Path(root or archive_root()) / f"{slug}__{now.strftime('%Y%m%dT%H%M%SZ')}"
    dest.mkdir(parents=True, exist_ok=False)
    moved_scratch = False
    if Path(scratch).is_dir():
        shutil.move(str(scratch), str(dest / "scratch"))
        moved_scratch = True
    limit = max_file_mb * 1024 * 1024
    large, large_bytes = 0, 0
    art_root = Path(art_root)
    if art_root.is_dir():
        for path in sorted(art_root.rglob("*")):
            rel = path.relative_to(art_root)
            if "transcripts" in rel.parts or not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            if size > limit:
                target = dest / "artifacts" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
                large += 1
                large_bytes += size
    record = {"run_id": run_id, "archive": dest.name, "archived_at": now.isoformat(),
              "delete_after": (now + timedelta(days=retention_days)).isoformat(),
              "scratch_moved": moved_scratch, "large_artifacts": large,
              "large_artifact_bytes": large_bytes}
    (dest / MANIFEST).write_text(json.dumps(record, indent=2))
    return record


def list_archives(root: Path | None = None) -> list[dict]:
    from .paths import archive_root

    root = Path(root or archive_root())
    rows = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        manifest = d / MANIFEST
        if not manifest.exists():
            continue
        try:
            info = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        info["path"] = str(d)
        rows.append(info)
    return rows


def purge(root: Path | None = None, now: datetime | None = None, run: str | None = None,
          force: bool = False) -> list[dict]:
    """Delete archives whose retention has run out (all matching ones with `force`)."""
    now = now or datetime.now(timezone.utc)
    gone = []
    for info in list_archives(root):
        if run and run not in (info.get("run_id"), info.get("archive")):
            continue
        expired = datetime.fromisoformat(info["delete_after"]) <= now
        if expired or force:
            shutil.rmtree(info["path"], ignore_errors=True)
            gone.append(info)
    return gone
