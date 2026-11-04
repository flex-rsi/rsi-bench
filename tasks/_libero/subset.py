"""Joint PRO/MAX subset selection from official metadata, without rollout outcomes.

The operator selects both tracks together. Task preparers only install this frozen
selection, so independent preparation cannot assign a physical state twice.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil

from prepare import SUITES, identity, read_json

GOALS = {"libero_spatial": (1, 3, 4), "libero_object": (1, 3, 7),
         "libero_goal": (3, 5, 9), "libero_10": (1, 2, 9)}
CATEGORIES = ("semantic", "object", "position", "task", "visual_noise_glare",
              "camera_view_angle", "object_texture", "view_occlusion", "object_shape",
              "initial_pose_position_angle")
EVENTS = ("camera_shift", "distractor_burst", "illumination_switch", "obstacle_insertion",
          "receptacle_relocation", "sensor_noise_onset", "target_relocation", "visual_theme_switch")
COUNTS = {"pro": {"dev": 20, "val": 40, "test": 80},
          "max": {"dev": 24, "val": 32, "test": 80}}
PROFILE = "pro-max-joint-20261016"
SEED = 195
# One metadata-only correction after complete exposure reconciliation: Lite has
# six fresh selected-goal Plus receptacle cases, not the seven in the draft.
FINAL_PLUS = {event: 8 if event == "camera_shift" else
              6 if event == "receptacle_relocation" else 7 for event in EVENTS}


def goal(case: dict, mapping: dict) -> tuple[str, int]:
    suite = case["task_suite_name"]
    names = mapping[suite + "_lan"]
    matches = [(len(name), index) for index, name in enumerate(names)
               if case["task_name"].startswith(name)]
    if not matches:
        raise ValueError(f"unknown original goal: {case['task_name']}")
    return suite, max(matches)[1]


def selected_goal(case: dict, mapping: dict) -> bool:
    suite, index = goal(case, mapping)
    return index in GOALS[suite]


def source(case: dict) -> str:
    return "pro" if "substrate_variant" in case else "plus"


def _select_panel(pool: list[dict], benchmark: str, split: str, mapping: dict) -> list[dict]:
    # scipy is only needed by the operator's metadata selection, not rollout.
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    count = COUNTS[benchmark][split]
    pool = sorted(pool, key=lambda case: case["case_id"])
    rng = np.random.default_rng(SEED + (0 if benchmark == "pro" else 10)
                                + ("dev", "val", "test").index(split))
    pool = [pool[int(i)] for i in rng.permutation(len(pool))]
    groups = [identity(case) for case in pool]
    goals = [goal(case, mapping) for case in pool]
    rows, lower, upper = [], [], []

    def add(indices, low, high, label):
        indices = list(indices)
        capacity = len({groups[i] for i in indices})
        if capacity < low:
            raise ValueError(f"{benchmark} {split} {label}: required {low}, available {capacity}")
        rows.append(indices)
        lower.append(low)
        upper.append(high)

    add(range(len(pool)), count, count, "total")
    for suite in SUITES:
        add((i for i, g in enumerate(goals) if g[0] == suite),
            count // 4, count // 4, suite)
    for suite, indices in GOALS.items():
        for index in indices:
            minimum, maximum = (4, 10) if benchmark == "max" and split == "test" else (1, count)
            add((i for i, g in enumerate(goals) if g == (suite, index)),
                minimum, maximum, f"{suite}:{index}")

    if benchmark == "pro":
        per_category = {"dev": 2, "val": 4, "test": 8}[split]
        categories = [case["substrate_variant"]["category"] for case in pool]
        for category in CATEGORIES:
            add((i for i, c in enumerate(categories) if c == category),
                per_category, per_category, category)
            for suite in SUITES:
                n = {"dev": 0, "val": 1, "test": 2}[split]
                add((i for i, c in enumerate(categories)
                     if c == category and goals[i][0] == suite),
                    n, max(1, n), f"{category}/{suite}")
            if split == "test":
                for g in {(suite, index) for suite, tasks in GOALS.items() for index in tasks}:
                    add((i for i, c in enumerate(categories) if c == category and goals[i] == g),
                        0, 1, f"{category}/{g}")
    else:
        per_event = {"dev": 3, "val": 4, "test": 10}[split]
        sources = [source(case) for case in pool]
        events = [case["scenario"]["change_type"] for case in pool]
        for event in EVENTS:
            add((i for i, e in enumerate(events) if e == event),
                per_event, per_event, event)
            if split == "test":
                for substrate, n in (("plus", FINAL_PLUS[event]), ("pro", 10 - FINAL_PLUS[event])):
                    add((i for i, e in enumerate(events)
                         if e == event and sources[i] == substrate), n, n, f"{event}/{substrate}")
        pro_count = {"dev": 7, "val": 10, "test": 24}[split]
        add((i for i, substrate in enumerate(sources) if substrate == "pro"),
            pro_count, pro_count, "PRO source")
        # Rotate the underlying source categories within the available event
        # combinations; no stronger per-category quota was approved for MAX.
        for substrate in ("plus", "pro"):
            categories = sorted({case["substrate_category"] for case in pool
                                 if source(case) == substrate})
            if len(categories) <= (pro_count if substrate == "pro" else count - pro_count):
                for category in categories:
                    add((i for i, case in enumerate(pool)
                         if case["substrate_category"] == category), 1, count,
                        f"source category {category}")

    by_identity = defaultdict(list)
    for i, group in enumerate(groups):
        by_identity[group].append(i)
    for indices in by_identity.values():
        add(indices, 0, 1, "distinct identity")
    row_indices, column_indices = [], []
    for row, indices in enumerate(rows):
        row_indices.extend([row] * len(indices))
        column_indices.extend(indices)
    matrix = coo_matrix((np.ones(len(row_indices)), (row_indices, column_indices)),
                        shape=(len(rows), len(pool))).tocsc()
    # A fixed shuffled order breaks ties. The objective contains no task scores.
    result = milp(np.zeros(len(pool)), integrality=np.ones(len(pool)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": 60})
    if result.x is None:
        raise ValueError(f"{benchmark} {split}: joint quota infeasible: {result.message}")
    chosen = [case for case, value in zip(pool, result.x) if value > 0.5]
    return sorted(chosen, key=lambda case: case["case_id"])


def select_joint(pro, maximum, lite, mapping, excluded):
    selected = {"pro": {}, "max": {}}
    assignment = {}
    lite_groups = {identity(case) for case in lite}
    # Final MAX has the narrowest official pool; reserve its groups first.
    for split in ("test", "val", "dev"):
        for benchmark in ("max", "pro"):
            pool = pro if benchmark == "pro" else lite if split == "test" else maximum
            available = [case for case in pool if selected_goal(case, mapping)
                         and (split == "dev" or identity(case) not in excluded)
                         and (split == "test" or identity(case) not in lite_groups)
                         and assignment.get(identity(case), split) == split]
            chosen = _select_panel(available, benchmark, split, mapping)
            selected[benchmark][split] = chosen
            assignment.update((identity(case), split) for case in chosen)
    validate_selection(selected, pro, maximum, lite, mapping, excluded)
    return selected


def validate_selection(selected, pro, maximum, lite, mapping, excluded):
    """Check the real deliverable, independently of the selection constraints."""
    official = {"pro": {case["case_id"]: case for case in pro},
                "max": {case["case_id"]: case for case in maximum}}
    lite_ids = {case["case_id"] for case in lite}
    lite_groups = {identity(case) for case in lite}
    assignment, summaries = {}, {}
    for benchmark, splits in selected.items():
        summaries[benchmark] = {}
        for split, count in COUNTS[benchmark].items():
            cases = splits[split]
            if len(cases) != count or len({identity(c) for c in cases}) != count:
                raise ValueError(f"{benchmark} {split}: incomplete or repeated physical identity")
            if any(c != official[benchmark].get(c["case_id"]) for c in cases):
                raise ValueError(f"{benchmark} {split}: modified official case")
            for case in cases:
                group = identity(case)
                if split != "dev" and group in excluded:
                    raise ValueError(f"{benchmark} {split}: historical identity in hidden split")
                if split != "test" and group in lite_groups:
                    raise ValueError(f"{benchmark} {split}: reserved Lite identity")
                if group in assignment and assignment[group] != split:
                    raise ValueError("physical identity crosses benchmark splits")
                assignment[group] = split
            goals = Counter(goal(c, mapping) for c in cases)
            if set(goals) != {(s, i) for s, ids in GOALS.items() for i in ids}:
                raise ValueError(f"{benchmark} {split}: selected goal coverage")
            suites = Counter(c["task_suite_name"] for c in cases)
            if suites != {suite: count // 4 for suite in SUITES}:
                raise ValueError(f"{benchmark} {split}: suite quota")
            summary = {"count": count, "distinct_identities": count,
                       "suites": dict(suites), "goals": {f"{s}:{i}": n for (s, i), n in sorted(goals.items())}}
            if benchmark == "pro":
                categories = Counter(c["substrate_variant"]["category"] for c in cases)
                if categories != {category: count // 10 for category in CATEGORIES}:
                    raise ValueError(f"pro {split}: category quota")
                for category in CATEGORIES:
                    cat_cases = [c for c in cases if c["substrate_variant"]["category"] == category]
                    cat_suites = Counter(c["task_suite_name"] for c in cat_cases)
                    if split == "dev" and sorted(cat_suites.values()) != [1, 1]:
                        raise ValueError("pro dev: category must span two suites")
                    if split != "dev" and cat_suites != {s: count // 40 for s in SUITES}:
                        raise ValueError(f"pro {split}: category/suite quota")
                    if split == "test" and len({goal(c, mapping) for c in cat_cases}) != 8:
                        raise ValueError("pro test: repeat goal in category/suite")
                summary["categories"] = dict(categories)
            else:
                events = Counter(c["scenario"]["change_type"] for c in cases)
                sources = Counter(source(c) for c in cases)
                expected_pro = {"dev": 7, "val": 10, "test": 24}[split]
                if events != {event: count // 8 for event in EVENTS}:
                    raise ValueError(f"max {split}: event quota")
                if sources != {"pro": expected_pro, "plus": count - expected_pro}:
                    raise ValueError(f"max {split}: source quota")
                if split == "test":
                    if any(c["case_id"] not in lite_ids for c in cases):
                        raise ValueError("max test: non-Lite case")
                    if any(not 4 <= n <= 10 for n in goals.values()):
                        raise ValueError("max test: per-goal count outside 4..10")
                    for event in EVENTS:
                        n = sum(source(c) == "plus" for c in cases
                                if c["scenario"]["change_type"] == event)
                        if n != FINAL_PLUS[event]:
                            raise ValueError(f"max test: {event} source quota")
                summary.update(events=dict(events), sources=dict(sources),
                               source_categories=dict(Counter(c["substrate_category"] for c in cases)),
                               event_sources={e: dict(Counter(source(c) for c in cases
                                   if c["scenario"]["change_type"] == e)) for e in EVENTS})
            summaries[benchmark][split] = summary
    return {"counts": COUNTS, "panels": summaries, "historical_identity_count": len(excluded),
            "cross_split_identity_overlap": 0, "hidden_historical_overlap": 0,
            "dev_val_lite_overlap": 0}


def install_frozen(benchmark: str) -> int:
    """Native task preparation installs, rather than resamples, joint panels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    default = os.environ.get("LIBERO_RSI_FROZEN_PANELS")
    parser.add_argument("--frozen-root", type=Path, default=default, required=default is None)
    args = parser.parse_args()
    task = f"libero-{benchmark}-adapter-open"
    panels = {}
    for split, count in COUNTS[benchmark].items():
        origin = args.frozen_root / task / split
        panel = read_json(origin / "cases.json")
        if (panel.get("task_id") != task or panel.get("profile") != PROFILE
                or len(panel["cases"]) != count):
            raise ValueError(f"{task} {split}: wrong frozen panel")
        panels[split] = origin
    for split, origin in panels.items():
        destination = args.out / split
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("cases.json", "runtime.json"):
            if (origin / name).resolve() != (destination / name).resolve():
                shutil.copyfile(origin / name, destination / name)
    return 0
