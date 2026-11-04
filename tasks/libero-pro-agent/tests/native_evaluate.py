#!/usr/bin/env python3
"""Coordinate frozen LIBERO rollouts and map official outcomes to FLEX Results.

The simulator/policy interpreter is separate from the harness interpreter.
MAX Plus and PRO substrates run in separate processes, each with one source
checkout. Held-out feedback remains under the existing FLEX private store.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def partition_manifests(payload: dict) -> dict[str, dict]:
    header = {key: payload[key] for key in ("benchmark_id", "benchmark_version", "protocol")}
    return {source: {**header, "cases": selected} for source in ("pro", "plus")
            if (selected := [case for case in payload["cases"]
                             if bool(case.get("substrate_variant")) == (source == "pro")])}


def runtime_env(runtime: dict, source: str) -> dict:
    maximum = Path(runtime["max_root"])
    env = dict(os.environ)
    env.update(
        PATH=str(Path(runtime["python"]).parent) + ":" + env.get("PATH", "/usr/bin:/bin"),
        PYTHONPATH=os.pathsep.join(map(str, (maximum / "scripts/libero_source_overlay",
                                           maximum / "src", maximum / "scripts"))),
        LIBERO_SOURCE_PACKAGE_ROOT=str(Path(runtime[source + "_root"]) / "libero"),
        LIBERO_CONFIG_PATH=runtime[source + "_config"],
        HF_HOME=str(Path(runtime["python"]).parents[2] / "cache/huggingface"),
        PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    # Workspace code is launched separately with images only and its own env.
    env.pop("RSIB_GATEWAY_URL", None)
    env.pop("RSIB_GATEWAY_TOKEN", None)
    return env


def pro_reward(episodes: list[dict], cases: list[dict]) -> dict:
    planned = len(cases)
    if (not planned or len(episodes) != planned
            or {row["case_id"] for row in episodes} != {case["case_id"] for case in cases}
            or any(not isinstance(row.get("success"), bool) for row in episodes)):
        raise RuntimeError("PRO episode coverage is incomplete")
    outcomes = {row["case_id"]: row for row in episodes}
    groups = {}
    for case in cases:
        group = groups.setdefault(case["perturbation"], {"planned": 0, "successes": 0})
        group["planned"] += 1
        group["successes"] += int(outcomes[case["case_id"]]["success"])
    for group in groups.values():
        group["success_rate"] = group["successes"] / group["planned"]
    successes = sum(row["success"] for row in episodes)
    return {"success_rate": successes / planned, "successes": successes,
            "planned_episodes": planned, "completed_episodes": len(episodes), "rollouts": planned,
            "metrics": {"by_perturbation": groups}}


def max_reward(report: dict, cases: list[dict]) -> dict:
    coverage, metrics = report["coverage"], report["end_to_end_metrics"]
    planned = len(cases)
    if (not planned or coverage["planned"] != planned or not coverage["execution_complete"]
            or not metrics["complete"]
            or any(metrics[arm]["measured"] != planned for arm in ("control", "intervention"))):
        raise RuntimeError("MAX paired coverage is incomplete or invalid")
    return {"dynamic_success_rate": metrics["intervention"]["accuracy_on_planned"],
            "base_success_rate": metrics["control"]["accuracy_on_planned"],
            "paired_delta": metrics["paired_robustness_delta_on_planned"],
            "planned_pairs": planned, "completed_pairs": planned, "coverage": 1.0,
            "trigger_unreached": coverage["trigger_unreached"],
            "response_query_unreached": coverage["response_query_unreached"],
            "rollouts": 2 * planned,
            "metrics": {"official": report}}


def execute(command: list[str], runtime: dict, source: str, log: Path) -> None:
    with log.open("w") as stream:
        result = subprocess.run(command, env=runtime_env(runtime, source), stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        if os.environ.get("RSIB_SPLIT", "dev") == "dev":
            print(log.read_text(errors="replace")[-3000:], file=sys.stderr)
        raise RuntimeError(f"frozen {source} runtime exited {result.returncode}; see {log.name}")


def public_dev_evidence(case_roots: list[Path], output: Path) -> dict:
    """Export public action/eef/query traces; retain native trace files unchanged."""
    evidence = {}
    for root in case_roots:
        for trace in sorted(root.glob("cases/*/*/trace.jsonl")):
            case_id, arm = trace.parent.parent.name, trace.parent.name
            rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            if not rows:
                continue
            native = rows[-1]
            queries = native.get("policy_queries", [])
            query_steps = [query["policy_step"] for query in queries if query.get("source", "model") == "model"]
            events = native.get("intervention_events", [])
            # Historical cosmos_query_boundary_step is the actual event step.
            event_step = events[0]["cosmos_query_boundary_step"] if events else None
            next_query = next((step for step in query_steps
                               if event_step is not None and step >= event_step), None)
            timing = {"event_step": event_step, "next_query_step": next_query,
                      "query_delay_steps": None if next_query is None else next_query - event_step}
            public = {"arm": arm, "timing": timing, "query_steps": query_steps,
                      "queries": [{key: query[key] for key in ("policy_step", "source") if key in query}
                                  for query in queries],
                      "steps": [{key: row[key] for key in ("policy_step", "action", "eef_position_m")
                                 if key in row} for row in native.get("executed_actions", [])]}
            target = trace.with_name("dev_trajectory.json")
            target.write_text(json.dumps(public, indent=2))
            item = evidence.setdefault(case_id, {"trajectory_paths": {}})
            item["trajectory_paths"][arm] = str(target.relative_to(output))
            if arm == "intervention":
                item["timing"] = timing
    return evidence


def case_feedback(benchmark: str, cases: list[dict], episodes: list[dict],
                  image_diagnostics: list[dict], dev_evidence: dict | None = None) -> list[dict]:
    """Join native outcomes with current-policy image summaries on the dev panel."""
    outcomes = {row["case_id"] if benchmark == "pro" else row["pair_id"]: row for row in episodes}
    diagnostics = {}
    for row in image_diagnostics:
        diagnostics.setdefault(row["case_id"], {})[row["arm"]] = {
            key: value for key, value in row.items() if key not in ("case_id", "arm")}
    feedback = []
    for case in cases:
        row = outcomes[case["case_id"]]
        common = {"case_id": case["case_id"], "task_suite_name": case["task_suite_name"],
                  "init_state_index": case["init_state_index"],
                  "image_diagnostics": diagnostics.get(case["case_id"], {})}
        if benchmark == "pro":
            common.update(perturbation=case["perturbation"], success=row["success"],
                          policy_steps=row["policy_steps"], failure_category=row["failure_category"])
        else:
            common.update(base_success=row["control_correct"], dynamic_success=row["intervention_correct"],
                          paired_delta=int(row["intervention_correct"]) - int(row["control_correct"]),
                          trigger_reached=row["trigger_reached"],
                          response_query_reached=row["response_query_reached"],
                          terminal_status=row["terminal_status"],
                          event={key: row[key] for key in ("change_type", "change_family", "severity")},
                          trigger=case["scenario"]["trigger"])
        evidence = (dev_evidence or {}).get(case["case_id"], {})
        if evidence:
            common.update(evidence)
        sequences = {arm: row["frame_sequence_path"]
                     for arm, row in diagnostics.get(case["case_id"], {}).items()
                     if "frame_sequence_path" in row}
        if sequences:
            common["frame_sequence_paths"] = sequences
        feedback.append(common)
    return feedback


def evaluate(data: Path, workspace: Path, output: Path, split: str, benchmark: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads((data / "cases.json").read_text())
    runtime_path = data / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    started = time.monotonic()
    partitions = partition_manifests(payload) if benchmark == "max" else {"pro": payload}
    case_roots = []
    for source, manifest in partitions.items():
        root = output / source
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        command = [runtime["python"], "-u", str(HERE / "runner.py"),
                   "--benchmark", benchmark, "--manifest", str(manifest_path),
                   "--runtime", str(runtime_path), "--workspace", str(workspace),
                   "--output-root", str(root)]
        if split == "dev":
            command.append("--capture-frames")
        execute(command, runtime, source, root / "run.log")
        case_roots.append(root)
    if benchmark == "pro":
        episodes = json.loads((case_roots[0] / "pro_results.json").read_text())["episodes"]
        reward = pro_reward(episodes, payload["cases"])
    else:
        manifest_path = output / "manifest.json"
        combined = {key: payload[key] for key in ("benchmark_id", "benchmark_version", "protocol", "cases")}
        manifest_path.write_text(json.dumps(combined, indent=2))
        command = [runtime["python"], str(Path(runtime["max_root"]) / "scripts/aggregate_cosmos_benchmark.py"),
                   str(case_roots[0]), "--manifest", str(manifest_path), "--output-dir", str(output),
                   "--query-interval", "30", "--require-render-qa"]
        for root in case_roots[1:]:
            command.extend(("--case-root", str(root)))
        execute(command, runtime, "pro" if "pro" in partitions else "plus", output / "aggregate.log")
        report = json.loads((output / "benchmark_summary.json").read_text())
        reward = max_reward(report, payload["cases"])
        episodes = [json.loads(line) for line in
                    (output / "end_to_end_results.jsonl").read_text().splitlines() if line]
    reward["wall_s"] = time.monotonic() - started
    reward["info"] = {"benchmark": payload["benchmark"], "profile": payload["profile"],
                      "method": "fixed-xvla-native-30-image-transform", "model_weights_frozen": True}
    feedback = {"episodes": episodes}
    if split == "dev":
        image_diagnostics = []
        for root in case_roots:
            for row in json.loads((root / "frames/image_diagnostics.json").read_text()):
                if "frame_sequence_path" in row:
                    row["frame_sequence_path"] = str((root / row["frame_sequence_path"]).relative_to(output))
                image_diagnostics.append(row)
        dev_evidence = public_dev_evidence(case_roots, output) if benchmark == "max" else {}
        feedback = {"cases": case_feedback(benchmark, payload["cases"], episodes, image_diagnostics, dev_evidence),
                    "timing_basis": "event_step is the actual intervention step; next_query_step is the "
                                    "first actual model query at or after it, excluding control replay. "
                                    "Historical cosmos_query_boundary_step retains its event-step meaning.",
                    "image_statistics_basis": "Means over actual model queries, excluding MAX action replay; "
                                              "robust_sigma uses median absolute mixed spatial differences."}
    (output / "feedback.json").write_text(json.dumps({**reward["info"], **feedback}, indent=2))
    reward["artifacts"] = [{"path": "feedback.json", "role": "diagnostics", "mime": "application/json"}]
    if split == "dev":
        for pattern in ("*/cases/*/*/dev_trajectory.json", "*/frames/*/*_sequence.json"):
            for path in sorted(output.glob(pattern)):
                reward["artifacts"].append({"path": str(path.relative_to(output)),
                                            "role": "diagnostics", "mime": "application/json"})
        for frame in sorted(output.glob("*/frames/*/*.png")):
            reward["artifacts"].append({"path": str(frame.relative_to(output)), "role": "observation",
                                        "mime": "image/png", "note": frame.parent.name + "/" + frame.name})
    if benchmark == "max":
        reward["artifacts"].append({"path": "benchmark_summary.json", "role": "diagnostics",
                                    "mime": "application/json"})
    (output / "reward.json").write_text(json.dumps(reward, indent=2))
    return reward


def main() -> int:
    benchmark = "pro" if HERE.parent.name == "libero-pro-open" else "max"
    reward = evaluate(Path(os.environ["RSIB_DATA_DIR"]), Path(os.environ["RSIB_WORKSPACE"]),
                      Path(os.environ["RSIB_OUTPUT_DIR"]), os.environ.get("RSIB_SPLIT", "dev"), benchmark)
    primary = "success_rate" if benchmark == "pro" else "dynamic_success_rate"
    print(json.dumps({primary: reward[primary], "rollouts": reward["rollouts"], "wall_s": reward["wall_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
