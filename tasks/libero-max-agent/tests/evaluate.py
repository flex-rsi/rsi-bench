#!/usr/bin/env python3
"""Evaluate agent-as-policy controllers on native PRO or paired MAX cases.

Cases run in parallel shards (one resident X-VLA per shard process). The evaluation's gateway
grant is handed to the trusted runners only; each controller gets a per-episode relay URL."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPISODE_MODEL_CALLS = 16        # model calls one episode (one MAX arm) may make
EPISODE_TIME_LIMIT_S = 1200.0   # an episode still running after this is scored as it stands
PROCESSES = int(os.environ.get("RSIB_LIBERO_EVAL_PROCESSES", "3"))


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Native LIBERO helpers, vendored from libero-max-open/tests so the package is self-contained.
old=load_module(HERE/"native_evaluate.py","libero_adapter_eval_helpers")
pairing=load_module(HERE/"paired_summary.py","libero_adapter_eval_pairing")
partition_manifests=old.partition_manifests


def runtime_env(runtime, source):
    """The native runtime environment plus the evaluation's grant, for the trusted runner only."""
    env = old.runtime_env(runtime, source)
    for name in ("RSIB_GATEWAY_URL", "RSIB_GATEWAY_TOKEN"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def shard_jobs(partitions, output, runtime, runtime_path, workspace, benchmark, split):
    """Split each runtime partition into shards, PROCESSES in total across partitions."""
    total = sum(len(m["cases"]) for m in partitions.values())
    jobs = []
    for source, manifest in partitions.items():
        cases = manifest["cases"]
        count = min(len(cases), max(1, round(PROCESSES * len(cases) / total)))
        for shard in range(count):
            root = output / source / f"shard{shard}"
            root.mkdir(parents=True, exist_ok=True)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({**manifest, "cases": cases[shard::count]}, indent=2))
            command = [runtime["python"], "-u", str(HERE / "runner.py"), "--benchmark", benchmark,
                       "--manifest", str(manifest_path), "--runtime", str(runtime_path),
                       "--workspace", str(workspace), "--output-root", str(root),
                       "--episode-model-calls", str(EPISODE_MODEL_CALLS),
                       "--episode-time-limit-s", str(EPISODE_TIME_LIMIT_S)]
            if split == "dev":
                command.append("--capture-frames")
            jobs.append((source, root, command))
    return jobs


def run_jobs(jobs, runtime):
    procs = []
    for source, root, command in jobs:
        log = (root / "run.log").open("w")
        procs.append((root, log, subprocess.Popen(command, env=runtime_env(runtime, source),
                                                  stdout=log, stderr=subprocess.STDOUT)))
    failed = []
    for root, log, proc in procs:
        proc.wait()
        log.close()
        if proc.returncode:
            failed.append(root)
    if failed:
        if os.environ.get("RSIB_SPLIT", "dev") == "dev":
            for root in failed:
                print((root / "run.log").read_text(errors="replace")[-3000:], file=sys.stderr)
        raise RuntimeError("runner shard(s) failed: " + ", ".join(str(r.relative_to(r.parents[1])) for r in failed))


def public_dev_evidence(case_roots, output):
    """Publish already recorded dev evidence without replacing richer trajectories."""
    evidence = {}
    for root in case_roots:
        for trajectory in sorted(root.glob("cases/*/*/dev_trajectory.json")):
            case_id, arm = trajectory.parent.parent.name, trajectory.parent.name
            public = json.loads(trajectory.read_text())
            item = evidence.setdefault(case_id, {"trajectory_paths": {},
                                                 "episode_frame_paths": {},
                                                 "stage_diagnostic_paths": {},
                                                 "stage_summary": {},
                                                 "command_tracking_summary": {}})
            item["trajectory_paths"][arm] = str(trajectory.relative_to(output))
            if "tracking_summary" in public:
                item["command_tracking_summary"][arm] = public["tracking_summary"]
            sequence = trajectory.with_name("dev_frames.json")
            item["episode_frame_paths"][arm] = str(sequence.relative_to(output))
            stage = trajectory.with_name("dev_stage_diagnostics.json")
            if stage.exists():
                item["stage_diagnostic_paths"][arm] = str(stage.relative_to(output))
                item["stage_summary"][arm] = json.loads(stage.read_text())["first_reached_steps"]
            if arm == "intervention":
                item["timing"] = public["timing"]
    return evidence


def pro_reward(episodes, cases):
    reward = old.pro_reward(episodes, cases)
    outcomes = {row["case_id"]: row for row in episodes}
    for field, name in (("task_suite_name", "by_suite"), ("substrate_category", "by_source_category")):
        groups = {}
        for case in cases:
            label = case.get(field, case["perturbation"])
            group = groups.setdefault(label, {"planned": 0, "successes": 0})
            group["planned"] += 1
            group["successes"] += int(outcomes[case["case_id"]]["success"])
        for group in groups.values():
            group["success_rate"] = group["successes"] / group["planned"]
        reward["metrics"][name] = groups
    reward["model_queries"] = sum(row["model_query_count"] for row in episodes)
    reward["model_query_elapsed_s"] = sum(row["model_query_elapsed_s"] for row in episodes)
    return reward


def evaluate(data,workspace,output,split,benchmark="max"):
    output.mkdir(parents=True,exist_ok=True)
    payload=json.loads((data/"cases.json").read_text())
    runtime_path=data/"runtime.json"
    runtime=json.loads(runtime_path.read_text())
    started=time.monotonic()
    partitions = partition_manifests(payload) if benchmark == "max" else {"pro": payload}
    jobs = shard_jobs(partitions, output, runtime, runtime_path, workspace, benchmark, split)
    run_jobs(jobs, runtime)
    roots = [root for _, root, _ in jobs]
    by_id = {}
    for root in roots:
        rows = (json.loads((root/"pro_results.json").read_text())["episodes"] if benchmark == "pro"
                else json.loads((root/"paired_results.json").read_text()))
        for row in rows:
            by_id[row["case_id"] if benchmark == "pro" else row["pair_id"]] = row
    episodes = [by_id[case["case_id"]] for case in payload["cases"]]  # planned order, all present
    if benchmark == "pro":
        reward = pro_reward(episodes, payload["cases"])
        report = {"planned": reward["planned_episodes"], "completed": reward["completed_episodes"],
                  "successes": reward["successes"], "success_rate": reward["success_rate"],
                  **reward["metrics"]}
    else:
        report = pairing.aggregate_pairs(payload["cases"], episodes)
        reward = old.max_reward(report, payload["cases"])
        reward.update(report["diagnostics"])
    stats = ([row.get("policy_model") for row in episodes] if benchmark == "pro" else
             [row.get(f"{arm}_policy_model") for row in episodes for arm in ("base", "dynamic")])
    stats = [s for s in stats if s]
    reward["policy_model_calls"] = sum(s["calls"] for s in stats)
    reward["policy_model_live_calls"] = sum(s["live_calls"] for s in stats)
    reward["policy_model_refused"] = sum(s["refused"] for s in stats)
    reward["policy_model_failures"] = sum(s["model_failures"] for s in stats)
    reward["episode_time_limits"] = (sum(bool(row.get("episode_time_limit_reached")) for row in episodes)
        if benchmark == "pro" else
        sum(bool(row.get(f"{arm}_time_limit_reached")) for row in episodes for arm in ("base", "dynamic")))
    report["policy_model"] = {k: reward[k] for k in ("policy_model_calls", "policy_model_live_calls",
                              "policy_model_refused", "policy_model_failures", "episode_time_limits")}
    (output/"benchmark_summary.json").write_text(json.dumps(report,indent=2))
    (output/"end_to_end_results.jsonl").write_text("".join(json.dumps(row)+"\n" for row in episodes))
    reward["wall_s"]=time.monotonic()-started
    reward["info"]={"benchmark":payload["benchmark"],"profile":payload["profile"],
        "method":"agent-as-policy: frozen X-VLA plus the run's model through a per-episode relay",
        "episode_model_calls_max":EPISODE_MODEL_CALLS,"episode_time_limit_s":EPISODE_TIME_LIMIT_S,
        "model_weights_frozen":True,
        "primary_metric": ("native success on all planned episodes" if benchmark == "pro" else
                          "absolute Dynamic success on all planned pairs"),
        "policy_chunk_steps":30, "query_schedule":"controller-selected; starter uses steps 0,30,60,...",
        "query_elapsed_basis":"synchronous resident model call wall seconds; replay excluded"}
    feedback={**reward["info"],"summary":report}
    artifacts=[{"path":"feedback.json","role":"diagnostics","mime":"application/json"},
               {"path":"benchmark_summary.json","role":"diagnostics","mime":"application/json"}]
    if split=="dev":
        diagnostics=[]
        for root in roots:
            for row in json.loads((root/"frames/image_diagnostics.json").read_text()):
                if "frame_sequence_path" in row:
                    row["frame_sequence_path"]=str((root/row["frame_sequence_path"]).relative_to(output))
                diagnostics.append(row)
        evidence=public_dev_evidence(roots,output)
        feedback["cases"]=old.case_feedback(benchmark,payload["cases"],episodes,diagnostics,evidence)
        by_id = {row["case_id"] if benchmark == "pro" else row["pair_id"]: row for row in episodes}
        for case in feedback["cases"]:
            row = by_id[case["case_id"]]
            if benchmark == "pro":
                case.update(evidence.get(case["case_id"], {}))
                keys = ("model_query_count", "model_query_elapsed_s", "policy_model",
                        "episode_time_limit_reached")
            else:
                keys = ("base_model_queries", "dynamic_model_queries", "dynamic_replay_queries",
                        "base_model_query_elapsed_s", "dynamic_model_query_elapsed_s", "prefix_replay_steps",
                        "base_policy_model", "dynamic_policy_model",
                        "base_time_limit_reached", "dynamic_time_limit_reached")
            case.update({key: row.get(key) for key in keys})
        feedback["timing_basis"] = ("policy steps follow the native ten-action warmup" if benchmark == "pro" else
            "event_step is the actual intervention step; next_query_step is the first actual Dynamic model query at or after it, excluding replay")
        for pattern,mime,role in (("*/*/cases/*/*/dev_trajectory.json","application/json","diagnostics"),
                                  ("*/*/cases/*/*/dev_stage_diagnostics.json","application/json","diagnostics"),
                                  ("*/*/cases/*/*/dev_frames.json","application/json","diagnostics"),
                                  ("*/*/cases/*/*/dev_frames/*.png","image/png","observation"),
                                  ("*/*/cases/*/*/adapter-io/adapter.log","text/plain","diagnostics"),
                                  ("*/*/frames/*/*_sequence.json","application/json","diagnostics"),
                                  ("*/*/frames/*/*.png","image/png","observation")):
            for path in sorted(output.glob(pattern)):
                artifacts.append({"path":str(path.relative_to(output)),"role":role,"mime":mime})
    (output/"feedback.json").write_text(json.dumps(feedback,indent=2))
    reward["artifacts"]=artifacts
    (output/"reward.json").write_text(json.dumps(reward,indent=2))
    return reward


def main(benchmark="max"):
    reward=evaluate(Path(os.environ["RSIB_DATA_DIR"]),Path(os.environ["RSIB_WORKSPACE"]),
                    Path(os.environ["RSIB_OUTPUT_DIR"]),os.environ.get("RSIB_SPLIT","dev"),benchmark=benchmark)
    metric = "success_rate" if benchmark == "pro" else "dynamic_success_rate"
    print(json.dumps({metric:reward[metric],
                      "rollouts":reward["rollouts"],"wall_s":reward["wall_s"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main("pro" if "-pro-" in HERE.parent.name else "max"))
