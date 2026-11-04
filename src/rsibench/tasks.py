"""Task packages: Harbor-style layout plus an [rsi] section.

    tasks/<id>/
      task.toml          Harbor fields ([task], [agent], [environment], [verifier], [scoring])
                         + [rsi] (tier, metric, anchors, noise, splits, entrypoints, resources,
                         including model_access: the solution may call the model when evaluated)
      instruction.md     what the agent is asked to do
      environment/workspace/   starter solution copied into the run's workspace/<id>/
      prepare.py         (optional) builds $RSIB_DATA_ROOT/<id>/{dev,val,test}
      tests/test.sh      evaluator; writes $RSIB_OUTPUT_DIR/reward.json

Only dev data is exposed to the agent. Val and test live under the data root and are read only
by the evaluator, which the orchestrator runs from a private copy of the task package.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import data_root, repo_root
from .util import sha256_tree

DIRECTIONS = {"maximize": "maximize", "max": "maximize", "higher": "maximize",
              "minimize": "minimize", "min": "minimize", "lower": "minimize"}


class TaskError(ValueError):
    pass


@dataclass
class Task:
    id: str
    root: Path
    spec: dict

    @property
    def rsi(self) -> dict:
        return self.spec.get("rsi", {})

    @property
    def tier(self) -> str:
        return self.rsi.get("tier", "S")

    @property
    def domain(self) -> str:
        return self.rsi.get("domain", "generic")

    @property
    def primary_metric(self) -> str:
        return self.rsi.get("metric", {}).get("primary", "reward")

    @property
    def public_metrics(self) -> list[str] | None:
        """Scalar metrics that val and test records keep: [rsi.metric] public plus the primary
        metric; None (no list) keeps every scalar metric."""
        public = self.rsi.get("metric", {}).get("public")
        return None if public is None else list(dict.fromkeys([self.primary_metric, *public]))

    @property
    def direction(self) -> str:
        raw = self.rsi.get("metric", {}).get("direction") or \
            self.spec.get("scoring", {}).get("direction", "maximize")
        return DIRECTIONS[str(raw).lower()]

    @property
    def baseline(self) -> float | None:
        return self.rsi.get("anchors", {}).get("baseline")

    @property
    def reference(self) -> float | None:
        return self.rsi.get("anchors", {}).get("reference")

    @property
    def noise_sigma(self) -> float:
        return float(self.rsi.get("noise", {}).get("sigma", 0.0))

    @property
    def instruction(self) -> str:
        return (self.root / "instruction.md").read_text()

    def entry(self, key: str, default: str | None = None) -> Path | None:
        rel = self.rsi.get("entrypoints", {}).get(key, default)
        return (self.root / rel) if rel else None

    @property
    def workspace_src(self) -> Path:
        return self.entry("workspace", "environment/workspace")

    @property
    def evaluator(self) -> Path:
        return self.entry("evaluate", "tests/test.sh")

    def split_dir(self, split: str, root: Path | None = None) -> Path:
        name = self.rsi.get("splits", {}).get(split, split)
        return (root or data_root()) / self.id / name

    @property
    def eval_timeout_s(self) -> float:
        return float(self.spec.get("verifier", {}).get("timeout_sec", 600))

    @property
    def eval_gpus(self) -> int:
        return int(self.rsi.get("resources", {}).get("eval_gpus", 0))

    @property
    def container(self) -> str | None:
        """Logical container name; the site profile maps it to an image ([slurm.containers])."""
        return self.rsi.get("resources", {}).get("container")

    @property
    def experiment_gpus_max(self) -> int:
        return int(self.rsi.get("resources", {}).get("experiment_gpus_max", 0))

    @property
    def model_access(self) -> bool:
        """Whether the solution may call the run's model during evaluation (through a grant)."""
        return self.rsi.get("resources", {}).get("model_access", False) is True

    @property
    def eval_model_calls(self) -> int | None:
        """The most model calls one evaluation may make (None: the protocol's per-grant cap)."""
        return self.rsi.get("resources", {}).get("eval_model_calls_max")

    def version_hash(self) -> str:
        return sha256_tree(self.root)

    def normalize(self, score: float | None) -> float | None:
        """(score - baseline) / (reference - baseline); 0 = baseline, 1 = reference."""
        if score is None or self.baseline is None or self.reference is None:
            return None
        span = self.reference - self.baseline
        if span == 0:
            return None
        return (score - self.baseline) / span

    def better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "maximize" else a < b


def load_task_dir(root: Path) -> Task:
    root = Path(root)
    toml_path = root / "task.toml"
    if not toml_path.exists():
        raise TaskError(f"{root}: missing task.toml")
    with open(toml_path, "rb") as f:
        spec = tomllib.load(f)
    rsi = spec.get("rsi")
    if not rsi:
        raise TaskError(f"{toml_path}: missing [rsi] section")
    task_id = rsi.get("id") or root.name
    task = Task(task_id, root, spec)
    problems = check_task(task)
    if problems:
        raise TaskError(f"{toml_path}: " + "; ".join(problems))
    return task


def check_task(task: Task) -> list[str]:
    problems = []
    if not (task.root / "instruction.md").exists():
        problems.append("missing instruction.md")
    if not task.workspace_src or not task.workspace_src.is_dir():
        problems.append(f"missing starter workspace {task.workspace_src}")
    if not task.evaluator.exists():
        problems.append(f"missing evaluator {task.evaluator}")
    try:
        task.direction
    except KeyError:
        problems.append("metric.direction must be maximize/minimize (or higher/lower)")
    if task.baseline is not None and task.reference is not None and task.baseline == task.reference:
        problems.append("anchors.baseline equals anchors.reference")
    public = task.rsi.get("metric", {}).get("public")
    if public is not None and not (isinstance(public, list)
                                   and all(isinstance(k, str) for k in public)):
        problems.append("metric.public must be a list of metric names")
    if not isinstance(task.rsi.get("resources", {}).get("model_access", False), bool):
        problems.append("resources.model_access must be true or false")
    calls = task.rsi.get("resources", {}).get("eval_model_calls_max")
    if calls is not None:
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            problems.append("resources.eval_model_calls_max must be a positive integer")
        elif not task.model_access:
            problems.append("resources.eval_model_calls_max needs resources.model_access = true")
    return problems


def tasks_dir() -> Path:
    return repo_root() / "tasks"


def load_task(task_id: str, base: Path | None = None) -> Task:
    root = (base or tasks_dir()) / task_id
    return load_task_dir(root)


def list_tasks(base: Path | None = None) -> list[Task]:
    out = []
    for path in sorted((base or tasks_dir()).iterdir()):
        if path.is_dir() and not path.name.startswith(("_", ".")) and (path / "task.toml").exists():
            out.append(load_task_dir(path))
    return out
