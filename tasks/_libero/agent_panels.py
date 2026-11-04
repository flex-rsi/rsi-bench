"""Freeze stratified LIBERO-PRO/MAX panels for the agent-as-policy campaign.

Selection reads official metadata only. Earlier result files on this host are scanned for
case/pair IDs (never for outcomes): any physical state that appears in one is "exposed" and is
kept out of the hidden val and test panels. Dev is shown to the agent anyway and may reuse
exposed states. No physical state appears in two splits. Every stratum (PRO category, MAX event) gets the same count in each
split, spread over the goals of its role, so what dev shows also appears in val and test.

Run with the robot runtime's interpreter:
    python tasks/_libero/agent_panels.py --runtime <runtime.json> --out panels
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flex/tasks/_libero"))
from prepare import pro_cases, read_json, task_map  # noqa: E402
from subset import CATEGORIES, EVENTS, goal  # noqa: E402

PROFILE = "libero-agent-20261017"
SEED = 20261017
ROLES = {
    "target": [("libero_spatial", 4), ("libero_object", 3), ("libero_goal", 5)],
    "source": [("libero_object", 5), ("libero_spatial", 2)],
}
# Cases per stratum in each split: PRO has 10 categories, MAX has 8 events.
PER_STRATUM = {"target": {"test": 3, "val": 2, "dev": 1},
               "source": {"test": 1, "val": 1, "dev": 1}}
TASK_IDS = {("target", "pro"): "libero-pro-agent", ("target", "max"): "libero-max-agent",
            ("source", "pro"): "libero-pro-agent-source", ("source", "max"): "libero-max-agent-source"}
# directories with result files of earlier campaigns, whose cases count as exposed (os.pathsep-separated)
SCAN_ROOTS = tuple(p for p in os.environ.get("LIBERO_EXPOSED_ROOTS", "").split(os.pathsep) if p)
RESULT_NAMES = ("end_to_end_results.jsonl", "pro_results.json", "paired_results.json",
                "baseline-result.json", "feedback.json")


def physical_identity(case, mapping):
    """Camera/light/language variants share the original state; Plus layouts are their own."""
    suite, index = goal(case, mapping)
    name = case["task_name"]
    if not case.get("substrate_variant") and any(x in name for x in ("_add_", "_level")):
        return ("layout", suite, name, case["init_state_index"])
    return ("original", suite, mapping[suite + "_lan"][index], case["init_state_index"])


def stratum(case, benchmark):
    return case["substrate_variant"]["category"] if benchmark == "pro" else case["scenario"]["change_type"]


def scan_ids(path, found):
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("case_id", "pair_id") and isinstance(item, str):
                    found.add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    try:
        text = path.read_text(errors="replace")
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    walk(json.loads(line))
        else:
            walk(json.loads(text))
    except (ValueError, OSError):
        pass


def exposed_ids(campaign_root):
    found, files = set(), 0
    for root in SCAN_ROOTS:
        for name in RESULT_NAMES:
            for path in Path(root).rglob(name):
                if campaign_root in path.parents or "pytest" in str(path) or "/tests/" in str(path):
                    continue
                files += 1
                scan_ids(path, found)
    return found, files


def select(role, benchmark, split, pool, mapping, excluded, assignment, rng_key):
    goals = ROLES[role]
    strata = CATEGORIES if benchmark == "pro" else EVENTS
    per = PER_STRATUM[role][split]
    by_slot = defaultdict(list)
    for case in pool:
        g = goal(case, mapping)
        if g in goals:
            by_slot[(stratum(case, benchmark), g)].append(case)
    chosen, used, fallbacks, capacity = [], set(), [], {}
    for s_index, s in enumerate(strata):
        for k in range(per):
            preferred = goals[(s_index + k) % len(goals)]
            order = [preferred] + [g for g in goals if g != preferred]
            picked = None
            for g in order:
                candidates = sorted(by_slot[(s, g)], key=lambda c: c["case_id"])
                random.Random(f"{SEED}/{rng_key}/{s}/{g}").shuffle(candidates)
                free = [c for c in candidates
                        if ((i := physical_identity(c, mapping)) not in excluded or split == "dev")
                        and i not in used
                        and assignment.get(i, split) == split]
                capacity[f"{s}|{g[0]}:{g[1]}"] = len(free)
                if free:
                    picked = free[0]
                    if g != preferred:
                        fallbacks.append({"stratum": s, "wanted": list(preferred), "used": list(g)})
                    break
            if picked is None:
                raise ValueError(f"{role} {benchmark} {split}: no fresh state for {s}")
            identity = physical_identity(picked, mapping)
            used.add(identity)
            assignment[identity] = split
            chosen.append(picked)
    return sorted(chosen, key=lambda c: c["case_id"]), fallbacks, capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runtime = read_json(args.runtime)
    mapping = task_map(Path(runtime["pro_root"]))
    bench = Path(runtime["max_root"]) / "benchmark"
    max_manifest = read_json(bench / "max8000/libero_max_8000.json")
    revision = read_json(bench / "max8000/pro_source_lock.json")["dataset_revision"]
    pools = {"pro": pro_cases(runtime, revision, list(CATEGORIES)), "max": max_manifest["cases"]}

    ids, files = exposed_ids(ROOT)
    official = {c["case_id"]: c for b in pools for c in pools[b]}
    excluded = {physical_identity(official[i], mapping) for i in ids if i in official}
    unknown = sorted(i for i in ids if i not in official)

    assignment, panels, report = {}, {}, {"profile": PROFILE, "seed": SEED,
        "exposure": {"result_files": files, "ids_found": len(ids), "ids_unmatched": len(unknown),
                     "excluded_physical_states": len(excluded)}, "panels": {}}
    for split in ("test", "val", "dev"):  # the hidden split takes the freshest states first
        for role in ("target", "source"):
            for benchmark in ("max", "pro"):
                cases, fallbacks, capacity = select(role, benchmark, split, pools[benchmark], mapping,
                                                    excluded, assignment, f"{role}/{benchmark}/{split}")
                task = TASK_IDS[(role, benchmark)]
                panels.setdefault(task, {})[split] = cases
                report["panels"].setdefault(task, {})[split] = {
                    "count": len(cases),
                    "strata": dict(Counter(stratum(c, benchmark) for c in cases)),
                    "goals": {f"{g[0]}:{g[1]}": n for g, n in Counter(goal(c, mapping) for c in cases).items()},
                    "fallbacks": fallbacks,
                    "min_free_per_slot": min(capacity.values()) if capacity else None}

    seen = defaultdict(set)
    for task, splits in panels.items():
        for split, cases in splits.items():
            for c in cases:
                seen[physical_identity(c, mapping)].add(split)
                assert split == "dev" or physical_identity(c, mapping) not in excluded
    assert all(len(s) == 1 for s in seen.values()), "a physical state crosses splits"

    for task, splits in panels.items():
        benchmark = "pro" if "-pro-" in task else "max"
        for split, cases in splits.items():
            out = args.out / task / split
            out.mkdir(parents=True, exist_ok=True)
            panel = {"task_id": task, "benchmark": benchmark, "profile": PROFILE, "cases": cases}
            if benchmark == "max":
                panel.update({k: max_manifest[k] for k in ("benchmark_id", "benchmark_version", "protocol")})
            (out / "cases.json").write_text(json.dumps(panel, indent=2) + "\n")
            shutil.copyfile(args.runtime, out / "runtime.json")
    (args.out / "selection-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
