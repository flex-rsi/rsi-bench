"""Create a run: one branch + one worktree, cut from a released harness tag.

Branch name:  [<namespace>/]run/<task-or-stream>/<model>/<mode>/s<seed>-<UTC timestamp>
              (namespace: --namespace or RSIB_RUN_NAMESPACE, e.g. robo/run/...)
Worktree:     $RSIB_RUNS_ROOT/<branch with '/' -> '--'>
Private copy: $RSIB_RUNS_ROOT/_private/<same>/tasks/<id>   (what the evaluator runs)
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from . import __version__, gitops, localenv
from .models import parse_spec, resolve_model
from .models.pricing import price
from .paths import private_root, repo_root, runs_root
from .protocol import RUN_DEFAULTS, frozen_run_protocol, load_protocol, resolve_budget
from .records import Records
from .tasks import load_task_dir
from .util import now_iso, sha256_tree, slugify, stamp

SELF_PRICED = {"claude-cli"}  # adapters that report the cost of every call themselves
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}  # default key variables
BASE_PATHS = ["harness", "tasks", "protocol"]  # what a run takes from its base commit


def branch_name(task_ids: list[str], model_name: str, mode: str, seed: int,
                namespace: str | None = None) -> str:
    if len(task_ids) == 1:
        head = slugify(task_ids[0])
    else:
        digest = hashlib.sha1("+".join(task_ids).encode()).hexdigest()[:6]
        head = f"stream-{len(task_ids)}tasks-{digest}"
    prefix = f"{namespace}/" if namespace else ""
    return f"{prefix}run/{head}/{slugify(model_name)}/{mode}/s{seed}-{stamp()}"


def _import_executor(repo: Path, commit: str, harness: Path) -> None:
    """Replace E (everything under harness/ except harness/improver/) with the one at commit."""
    with tempfile.TemporaryDirectory() as tmp:
        gitops.archive_extract(repo, commit, "harness", Path(tmp))
        for entry in harness.iterdir():
            if entry.name != "improver":
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        for entry in (Path(tmp) / "harness").iterdir():
            if entry.name != "improver":
                (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, harness / entry.name)


def new_run(task_ids: list[str], model_spec: str, mode: str = "full", seed: int = 0,
            effort: str | None = None, model_params: dict | None = None,
            auth: str | None = None, base_ref: str | None = None, improver_from: str | None = None,
            executor_from: str | None = None, budget_overrides: dict | None = None, tier: str | None = None,
            protocol_overrides: dict | None = None, sandbox: dict | None = None,
            namespace: str | None = None,
            repo: Path | None = None, notes: str = "", log=print) -> tuple[str, Path]:
    if mode not in ("base", "fixed-i", "full"):
        raise ValueError("mode must be base, fixed-i or full")
    if not task_ids:
        raise ValueError("at least one task is required")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("a stream may not repeat a task")
    repo = Path(repo or repo_root())
    model_spec, auth, preset = resolve_model(model_spec, auth)
    model_params = {**preset, **(model_params or {})}
    base_ref = _base_ref(repo, base_ref, log)
    base_commit = gitops.rev_parse(repo, base_ref)

    adapter, model_name, _ = parse_spec(model_spec)
    display_name = {"mock": f"mock-{model_name}"}.get(adapter, model_name)
    if namespace is None:
        namespace = localenv.lookup("RSIB_RUN_NAMESPACE")
    namespace = slugify(namespace) if namespace else None
    branch = branch_name(task_ids, display_name, mode, seed, namespace)
    if namespace and gitops.ref_exists(repo, f"refs/heads/{namespace}"):
        raise ValueError(f"a branch named {namespace!r} exists, so {branch} cannot be created; "
                         f"rename it (for example to {namespace}/main)")
    slug = slugify(branch.replace("/", "--"))
    worktree = runs_root() / slug
    private = private_root() / slug
    if worktree.exists() or private.exists():
        raise FileExistsError(f"{worktree} already exists")

    # Task packages come from the base commit, never from a working tree someone may have edited.
    tasks = []
    for tid in task_ids:
        gitops.archive_extract(repo, base_commit, f"tasks/{tid}", private)
        tasks.append(load_task_dir(private / "tasks" / tid))

    protocol = load_protocol(_protocol_at(repo, base_commit, private))
    known = set(protocol["run"]) | set(RUN_DEFAULTS)
    unknown = set(protocol_overrides or {}) - known
    if unknown:
        raise ValueError(f"unknown protocol keys {sorted(unknown)}; known: {sorted(known)}")
    gitops.add_worktree(repo, worktree, branch, base_commit)

    for task in tasks:
        shutil.copytree(task.workspace_src, worktree / "workspace" / task.id,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    executor_source = None
    if executor_from:
        src_commit = gitops.rev_parse(repo, executor_from)
        _import_executor(repo, src_commit, worktree / "harness")
        executor_source = {"ref": executor_from, "commit": src_commit}

    improver_source = None
    if improver_from:
        src_commit = gitops.rev_parse(repo, improver_from)
        shutil.rmtree(worktree / "harness" / "improver")
        gitops.archive_extract(repo, src_commit, "harness/improver", worktree)
        improver_source = {"ref": improver_from, "commit": src_commit}

    records = Records(worktree)
    records.ensure_layout()
    run = {
        "schema_version": "1.0",
        "run_id": branch,
        "branch": branch,
        "created_at": now_iso(),
        "base": {"ref": base_ref, "commit": base_commit},
        "mode": mode,
        "tasks": [{"id": t.id, "tier": tier or t.tier, "version_hash": t.version_hash(),
                   "domain": t.domain, "metric": t.primary_metric, "direction": t.direction,
                   "anchors": {"baseline": t.baseline, "reference": t.reference},
                   "noise_sigma": t.noise_sigma} for t in tasks],
        "model": {"adapter": adapter, "name": display_name, "spec": model_spec, "auth": auth,
                  "effort": effort, "params": dict(model_params),
                  "usd_metering": adapter in SELF_PRICED or price(adapter, display_name) is not None},
        "seed": int(seed),
        "protocol": {**frozen_run_protocol(protocol), **(protocol_overrides or {})},
        "budgets": {t.id: resolve_budget(protocol, tier or t.tier, budget_overrides) for t in tasks},
        "improver_source": improver_source,
        "executor_source": executor_source,
        "sandbox": sandbox or {"enabled": False},
        "harness_tree": sha256_tree(worktree / "harness"),
        "rsibench_version": __version__,
        "notes": notes,
    }
    records.write_run(run)
    gitops.commit(worktree, f"init {branch}", {"Rsib-Run": branch, "Rsib-Scope": "init"},
                  ["workspace", "harness", "records"])
    log(f"created {branch}\n  worktree: {worktree}")
    if not run["sandbox"].get("enabled"):
        log("  note: no filesystem sandbox; agents can read anything this user can read")
    if not run["model"]["usd_metering"]:
        log(f"  note: no price for {adapter}:{display_name}; usd budget is not enforced")
    if adapter in KEY_ENV:
        log(_key_note(adapter, model_params))
    return branch, worktree


def _base_ref(repo: Path, base_ref: str | None, log) -> str:
    """The release to branch from: --base, else the newest harness-v* tag in the checked-out
    branch, so a branch with its own releases (harness-v0.3.0-mylab.1) gets its own runs."""
    if base_ref is not None:
        if gitops.ref_exists(repo, f"refs/heads/{base_ref}"):
            log(f"note: --base {base_ref} is a branch; reports group runs by base name, so runs "
                f"from different commits of it are pooled. Tag a release to keep them apart.")
        return base_ref
    tag = gitops.latest_tag(repo, merged="HEAD")
    if tag is None:
        log("warning: no harness-v* release tag found; branching from HEAD")
        return "HEAD"
    ahead = gitops.count_commits(repo, f"{tag}..HEAD", BASE_PATHS)
    if ahead:
        log(f"warning: {ahead} commit(s) after {tag} on the checked-out branch change "
            f"{', '.join(p + '/' for p in BASE_PATHS)} and are not in this run; tag a release "
            f"(git tag harness-vX.Y.Z-<name>) or pass --base")
    return tag


def _key_note(adapter: str, params: dict) -> str:
    """Where the API key will come from at `rsib run start` (variable names only)."""
    name = params.get("api_key_env", KEY_ENV[adapter])
    found = localenv.where(name)
    if found:
        return f"  api key: {name} from {found}"
    if adapter == "anthropic" and localenv.where("ANTHROPIC_AUTH_TOKEN"):
        return f"  api key: ANTHROPIC_AUTH_TOKEN from {localenv.where('ANTHROPIC_AUTH_TOKEN')}"
    return (f"  warning: {name} is not in the environment or a .env file "
            f"($RSIB_ENV_FILE, ~/.config/rsib/.env, <repo>/.env); set it before `rsib run start` "
            f"(a local server without auth needs none)")


def _protocol_at(repo: Path, commit: str, scratch: Path) -> Path:
    """Protocol file as of the base commit (so a run reproduces its release)."""
    gitops.archive_extract(repo, commit, "protocol/default.toml", scratch)
    return scratch / "protocol" / "default.toml"
