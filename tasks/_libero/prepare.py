"""Select fixed PRO episodes and whole MAX pairs from the official frozen data.

LIBERO_RSI_RUNTIME points to the runtime JSON made by setup_libero.sh. Both task
preparers use one partition of the underlying task/initial-state identities;
all MAX Lite identities are reserved for test, including their PRO counterparts.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from itertools import cycle
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
AXES = {"lan": "semantic", "object": "object", "swap": "position", "task": "task"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def task_map(pro_root: Path) -> dict:
    source = pro_root / "libero/libero/benchmark/libero_suite_task_map.py"
    for statement in ast.parse(source.read_text()).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise ValueError(f"libero_task_map is missing from {source}")


def pro_cases(runtime: dict, revision: str, included_categories: list[str]) -> list[dict]:
    config = read_json(Path(runtime["pro_config"]) / "config.yaml")
    bddl_root = Path(config["bddl_files"])
    init_root = Path(config["init_states"])
    mapping = task_map(Path(runtime["pro_root"]))
    variants = []
    for suite in SUITES:
        for suffix, axis in AXES.items():
            if axis not in included_categories:
                continue
            full_suite = suite + "_" + suffix
            for task_index, task in enumerate(mapping.get(full_suite, [])):
                relative = f"{full_suite}/{task}.bddl"
                variants.append((suite, task_index, task, axis, axis, relative,
                                 f"{full_suite}/{task}.pruned_init", relative))
    metadata = read_json(Path(config["datasets"]) / "metadata/dataset_index.json")
    for entry in metadata["entries"]:
        category = entry["category"].split("_", 1)[1]
        if category not in included_categories:
            continue
        suite, task = entry["suite"], entry["task"]
        task_index = mapping[suite + "_lan"].index(task)
        relative = str(PurePosixPath(entry["path"]).relative_to("bddl_files"))
        variants.append((suite, task_index, task, "environment", category, relative,
                         f"{suite}/{task}.pruned_init", f"{suite}_lan/{task}.bddl"))
    paths = sorted({str(init_root / variant[6]) for variant in variants})
    lengths_script = (
        "import json,sys,torch; "
        "print(json.dumps([len(torch.load(p,weights_only=False)) for p in json.loads(sys.argv[1])]))"
    )
    result = subprocess.run([runtime["python"], "-c", lengths_script, json.dumps(paths)],
                            check=True, capture_output=True, text=True)
    lengths = dict(zip(paths, json.loads(result.stdout)))
    cases = []
    for suite, task_index, task, axis, category, bddl, initial, reference in variants:
        text = (bddl_root / bddl).read_text()
        match = re.search(r"\(:language\s+([^)]*)\)", text)
        language = " ".join(match.group(1).split())
        for index in range(lengths[str(init_root / initial)]):
            cases.append({
                "case_id": f"pro-{category}-{suite}-t{task_index:02d}-i{index:02d}-p195",
                "task_suite_name": suite, "task_index": task_index, "task_name": task,
                "init_state_index": index, "policy_seed": 195, "scenario": {"seed": 195},
                "perturbation": axis, "substrate_category": "LIBERO-PRO/" + category,
                "substrate_variant": {"benchmark": "LIBERO-PRO", "category": category,
                    "bddl_file": bddl, "init_states_file": initial,
                    "init_reference_bddl_file": reference, "language": language,
                    "source_revision": revision},
            })
    return cases


def identity(case: dict) -> tuple:
    variant = case.get("substrate_variant")
    if variant is not None:
        reference = PurePosixPath(variant["init_reference_bddl_file"])
        suite = str(reference.parent)
        for suffix in AXES:
            if suite.endswith("_" + suffix):
                suite = suite[:-(len(suffix) + 1)]
                break
        return "pro", suite, reference.stem, case["init_state_index"]
    return "plus", case["task_suite_name"], case["task_name"], case["init_state_index"]


def partition(pro: list[dict], maximum: list[dict], lite: list[dict]) -> dict[tuple, str]:
    held_out = {identity(case) for case in lite}
    every = {identity(case) for case in pro + maximum}
    assignment = {group: "test" for group in held_out}
    for split, group in zip(cycle(("dev", "val", "test")), sorted(every - held_out)):
        assignment[group] = split
    return assignment


def balanced(cases: list[dict], fields: list) -> list[dict]:
    """Round-robin at each level, so a small panel covers axes or events first."""
    if not fields:
        return sorted(cases, key=lambda case: case["case_id"])
    buckets = defaultdict(list)
    for case in cases:
        buckets[fields[0](case)].append(case)
    iterators = [iter(balanced(buckets[key], fields[1:])) for key in sorted(buckets)]
    result = []
    while iterators:
        remaining = []
        for iterator in iterators:
            try:
                result.append(next(iterator))
                remaining.append(iterator)
            except StopIteration:
                pass
        iterators = remaining
    return result


def formal_pro(cases: list[dict], count: int, split_index: int) -> list[dict]:
    """Spread fixed episodes across categories, suites and base goals."""
    categories = sorted({case["substrate_variant"]["category"] for case in cases})
    category_rank = {category: index for index, category in enumerate(categories)}

    def category(case):
        return category_rank[case["substrate_variant"]["category"]]

    fields = [category,
              lambda case: (SUITES.index(case["task_suite_name"]) - 2 * category(case)) % 4,
              lambda case: (case["task_index"] - category(case) - 3 * split_index) % 10]
    return balanced(cases, fields)[:count]


def formal_max(cases: list[dict], count: int, split_index: int, mapping: dict,
               prefix: list[dict] | None = None) -> list[dict]:
    """Balance official events and sources, optionally extending a frozen prefix."""
    events = sorted({case["scenario"]["change_type"] for case in cases})
    rotation = 4 * split_index % len(events)
    events = events[rotation:] + events[:rotation]
    quotas = [count // len(events) + (rank < count % len(events))
              for rank in range(len(events))]
    pro_quotas = [3 * quota // 10 for quota in quotas]
    # Largest remainders; ties follow the split's rotated event order.
    remainder_order = sorted(range(len(events)), key=lambda rank: (-(3 * quotas[rank] % 10), rank))
    remaining = (3 * count + 5) // 10 - sum(pro_quotas)
    for rank in remainder_order[:remaining]:
        pro_quotas[rank] += 1
    frozen = prefix or []
    prefix_pro = [sum(case["scenario"]["change_type"] == event and "substrate_variant" in case
                      for case in frozen) for event in events]
    prefix_plus = [sum(case["scenario"]["change_type"] == event and "substrate_variant" not in case
                       for case in frozen) for event in events]
    if frozen:
        # Preserve the four old validation pairs even when a count override moves
        # a rounded PRO quota to an event whose frozen pair came from Plus.
        pro_quotas = [min(max(quota, prefix_pro[rank]), quotas[rank] - prefix_plus[rank])
                      for rank, quota in enumerate(pro_quotas)]
        remaining = (3 * count + 5) // 10 - sum(pro_quotas)
        order = remainder_order if remaining > 0 else list(reversed(remainder_order))
        for rank in order:
            if remaining > 0:
                change = min(remaining, quotas[rank] - prefix_plus[rank] - pro_quotas[rank])
            else:
                change = -min(-remaining, pro_quotas[rank] - prefix_pro[rank])
            pro_quotas[rank] += change
            remaining -= change
    canonical = {}
    for case in cases:
        names = mapping[case["task_suite_name"] + "_lan"]
        canonical[case["case_id"]] = max(
            (len(name), index) for index, name in enumerate(names)
            if case["task_name"].startswith(name))[1]
    selected = list(frozen)
    prefix_ids = {case["case_id"] for case in selected}
    for event_rank, event in enumerate(events):
        for source, quota in (("plus", quotas[event_rank] - pro_quotas[event_rank]),
                              ("pro", pro_quotas[event_rank])):
            quota -= prefix_pro[event_rank] if source == "pro" else prefix_plus[event_rank]
            pool = [case for case in cases if case["scenario"]["change_type"] == event
                    and ("pro" if "substrate_variant" in case else "plus") == source
                    and case["case_id"] not in prefix_ids]
            categories = sorted({case["substrate_category"] for case in pool})
            category_rank = {category: index for index, category in enumerate(categories)}

            def category(case):
                return category_rank[case["substrate_category"]]

            fields = [lambda case: (category(case) - 2 * event_rank - split_index) % len(categories),
                      lambda case: (SUITES.index(case["task_suite_name"]) - category(case)
                                    - event_rank - split_index) % 4,
                      lambda case: (canonical[case["case_id"]] - category(case)
                                    - event_rank - 3 * split_index) % 10,
                      lambda case: case["task_name"]]
            selected.extend(balanced(pool, fields)[:quota])
    return selected


def main(benchmark: str) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=("pilot", "smoke", "formal", "small20", "rsi-v2"),
                        default=os.environ.get("LIBERO_RSI_PROFILE", "pilot"))
    parser.add_argument("--history", default=os.environ.get("LIBERO_RSI_HISTORY"),
                        help="prior hidden identities; rsi-v2 defaults to the bundled exposure ledger")
    for split in ("dev", "val", "test"):
        parser.add_argument(f"--{split}-count", type=int,
                            default=os.environ.get("LIBERO_RSI_" + split.upper() + "_COUNT"))
    args = parser.parse_args()
    excluded = set()
    if args.profile == "rsi-v2":
        history = Path(args.history) if args.history else Path(__file__).with_name("prior-hidden-v2.json")
        excluded = {tuple(group) for group in read_json(history)["hidden_identities"]}
    runtime = read_json(Path(os.environ["LIBERO_RSI_RUNTIME"]))
    maximum_root = Path(runtime["max_root"]) / "benchmark"
    maximum_manifest = read_json(maximum_root / "max8000/libero_max_8000.json")
    lite_manifest = read_json(maximum_root / "lite/libero_max_lite.json")
    maximum = maximum_manifest["cases"]
    lite = lite_manifest["cases"]
    source_lock = read_json(maximum_root / "max8000/pro_source_lock.json")
    pro = pro_cases(runtime, source_lock["dataset_revision"], source_lock["included_categories"])
    assignment = partition(pro, maximum, lite)
    if benchmark == "pro":
        fields = [lambda case: case["perturbation"],
                  lambda case: case["substrate_variant"]["category"],
                  lambda case: case["task_suite_name"]]
    else:
        fields = [lambda case: case["scenario"]["change_type"],
                  lambda case: "pro" if "substrate_variant" in case else "plus",
                  lambda case: case["task_suite_name"]]
    formal_selection = args.profile in ("formal", "small20", "rsi-v2")
    mapping = task_map(Path(runtime["pro_root"])) if formal_selection and benchmark == "max" else None
    for split_index, (split, pilot_count) in enumerate((("dev", 16), ("val", 16), ("test", 32))):
        count = getattr(args, split + "_count")
        default_count = {"smoke": 1, "pilot": pilot_count,
                         "formal": (60, 20, 20)[split_index],
                         "small20": (12, 4, 4)[split_index],
                         "rsi-v2": (12, 16, 10)[split_index]}[args.profile]
        count = count if count is not None else default_count
        if count < 1:
            raise ValueError("each split needs at least one case")
        pool = pro if benchmark == "pro" else (lite if split == "test" else maximum)
        available = [case for case in pool if assignment[identity(case)] == split]
        if args.profile == "rsi-v2" and split == "test":
            available = [case for case in available if identity(case) not in excluded]
        if formal_selection:
            prefix = None
            if args.profile == "rsi-v2" and benchmark == "max" and split == "val":
                prefix = formal_max(available, min(4, count), split_index, mapping)
            cases = formal_pro(available, count, split_index) if benchmark == "pro" else formal_max(
                available, count, split_index, mapping, prefix)
        else:
            cases = balanced(available, fields)[:count]
        if len(cases) != count:
            raise ValueError(f"{benchmark} {split}: requested {count}, only {len(cases)} cases available")
        out = Path(args.out) / split
        out.mkdir(parents=True, exist_ok=True)
        panel = {"benchmark": benchmark, "profile": args.profile, "cases": cases}
        if benchmark == "max":
            manifest = lite_manifest if split == "test" else maximum_manifest
            panel.update({key: manifest[key] for key in ("benchmark_id", "benchmark_version", "protocol")})
        (out / "cases.json").write_text(json.dumps(panel, indent=2) + "\n")
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        unit = "pairs" if benchmark == "max" else "episodes"
        print(f"{split}: {len(cases)} {unit} ({args.profile})")
    return 0
