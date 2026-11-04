#!/usr/bin/env python3
"""Shared PRO/MAX per-step X-VLA control with deterministic MAX paired prefixes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return value


def run_arm(*, env, worker, initial_observation, instruction, horizon, query_policy,
            public_observation, replay=None, capture=None, policy_seed=195, diagnostics=None, paired=True,
            set_replaying=None, time_limit_s=None):
    """Control an already-settled PRO episode or MAX arm.

    MAX replay reconstructs the fresh CPU controller.

    Replay rows are private in-memory Base public inputs, candidate requests and
    trusted replies. Only one pair's full observations are retained at a time.
    The MAX env applies changes after an action; its next observation switches
    this loop from prefix reconstruction to actual model calls. PRO uses the
    same query/action path without storing camera history or applying replay.
    """
    raw, last_action, history = initial_observation, None, []
    prefix_steps, success, steps = 0, False, 0
    queries = env.policy_queries if paired else []
    model_elapsed = 0.0
    started, timed_out = time.monotonic(), False
    if diagnostics is not None:
        diagnostics.observe_state(raw, env, 0)
    for step in range(horizon):
        reconstruct = replay is not None and not env.runtime.applied
        if reconstruct and step >= len(replay):
            raise RuntimeError("Base prefix history ended before Dynamic intervention")
        if set_replaying is not None:
            set_replaying(reconstruct)
        if time_limit_s is not None and not reconstruct and time.monotonic() - started > time_limit_s:
            timed_out = True  # scored as the scene stands, like any other unfinished episode
            break
        current_instruction = getattr(env.runtime, "current_instruction", instruction) if paired else instruction
        public = (snapshot(replay[step]["observation"]) if reconstruct else
                  public_observation(raw, env, current_instruction, step, last_action))
        if diagnostics is not None:
            diagnostics.record_pre_action(public)
        recorded = {"observation": snapshot(public), "queries": []} if paired and replay is None else None
        calls = 0

        def query(images):
            nonlocal calls, model_elapsed
            calls += 1
            if calls > 1:
                raise RuntimeError("only one policy query is allowed per step")
            requested = snapshot(images)
            elapsed = 0.0
            if reconstruct:
                expected = replay[step]["queries"]
                if not expected:
                    raise RuntimeError("prefix query decision differs from Base")
                if (set(requested) != set(expected[0]["images"]) or
                        any(not np.array_equal(requested[key], expected[0]["images"][key])
                            for key in requested)):
                    raise RuntimeError("prefix query images differ from Base")
                chunk, source = expected[0]["reply"].copy(), "replay"
            else:
                start = time.monotonic()
                chunk = query_policy(raw, env, current_instruction, step, images)
                elapsed = time.monotonic() - start
                model_elapsed += elapsed
                source = "model"
            chunk = np.asarray(chunk)
            if chunk.dtype != np.float32 or chunk.shape != (30,7) or not np.isfinite(chunk).all():
                raise ValueError("fixed policy must return finite float32 [30,7]")
            if diagnostics is not None:
                diagnostics.record_query(step, source, chunk)
            if recorded is not None:
                recorded["queries"].append({"images": requested, "reply": chunk.copy()})
            queries.append({"policy_step": step, "source": source,
                "query_seed": policy_seed + step, "actions": chunk.tolist(),
                "action_chunk_shape": [30,7], "instruction": current_instruction,
                "query_elapsed_s": elapsed})
            if capture is not None and source == "model":
                capture.record({key: public["images"][key][0] for key in IMAGE_KEYS},
                    {key: requested[key][0] for key in IMAGE_KEYS}, policy_step=step,
                    action=None if last_action is None else last_action.tolist(),
                    eef_position_m=public["proprio"]["eef_pos"].tolist())
            return chunk.copy()

        action = np.asarray(worker.act(public, query))
        if calls > 1:
            raise RuntimeError("only one policy query is allowed per step")
        if action.dtype != np.float32 or action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("controller must return a finite float32 seven-element action")
        if reconstruct:
            if calls != len(replay[step]["queries"]):
                raise RuntimeError("prefix query decision differs from Base")
            if not np.array_equal(action, replay[step]["action"]):
                raise RuntimeError("prefix action decision differs from Base")
            action = replay[step]["action"].copy()
            prefix_steps += 1
        if recorded is not None:
            recorded["action"] = action.copy()
            history.append(recorded)
        if diagnostics is not None:
            diagnostics.record_executed(action)
        raw, _, done, info = env.step(action.tolist())
        steps = step + 1
        if diagnostics is not None:
            diagnostics.observe_state(raw, env, step + 1)
        if capture is not None and paired:
            capture.observe_step(raw, env, action, (info or {}).get("libero_max_event"))
        last_action = action.copy()
        if done:
            success = True
            break
    row = (env.record_outcome(success) if paired else
           {"success": success, "policy_steps": steps, "policy_queries": queries})
    if paired:
        row.update(prefix_replay_steps=prefix_steps, prefix_verified=True)
    row.update(model_query_count=sum(q["source"] == "model" for q in queries),
               replay_query_count=sum(q["source"] == "replay" for q in queries),
               model_query_elapsed_s=model_elapsed, episode_time_limit_reached=timed_out)
    if diagnostics is not None:
        if steps:
            diagnostics.record_post_action(public_observation(
                raw, env, current_instruction, steps, last_action))
        diagnostics.finish(row)
    return row, history


def native_public_observation(raw, env, instruction, step, last_action):
    """Expose the two native preprocessor inputs and the robot's own state."""
    images = {target: np.ascontiguousarray(np.asarray(raw[source]).transpose(2,0,1)[None],
                                         dtype=np.float32) / np.float32(255)
              for source,target in (("agentview_image",IMAGE_KEYS[0]),
                                    ("robot0_eye_in_hand_image",IMAGE_KEYS[1]))}
    matrix = np.asarray(env.robots[0].controller.ee_ori_mat, dtype=np.float32)
    proprio = {"eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy(),
               "eef_rot6d": np.concatenate((matrix[:,0],matrix[:,1])).astype(np.float32)}
    for source,target in (("robot0_gripper_qpos","gripper_qpos"),("robot0_joint_pos","joint_qpos")):
        if source in raw:
            proprio[target] = np.asarray(raw[source], dtype=np.float32).copy()
    return {"images":images,"instruction":instruction,"step":step,"proprio":proprio,
            "last_action":None if last_action is None else last_action.copy()}


def agent_public_observation():
    """native_public_observation plus calibration and depth, with one geometry per arm."""
    agent = load_module(HERE/"agent.py", "libero_agent_geometry")
    geometry = agent.EpisodeGeometry()

    def observe(raw, env, instruction, step, last_action):
        public = native_public_observation(raw, env, instruction, step, last_action)
        public["cameras"], public["depth"] = geometry.observe(env, raw)
        return public
    return observe


def native_query(native, policy_parts, action_key, policy_seed=195):
    """Reuse pinned native inference and processors with requested image tensors."""
    def query(raw, env, instruction, step, images):
        original = native._format_observation
        def format_observation(raw_obs, query_env):
            formatted = original(raw_obs, query_env)
            for key in IMAGE_KEYS:
                formatted[key] = formatted[key].new_tensor(images[key])
            return formatted
        native._format_observation = format_observation
        try:
            return native._predict_action_chunk(raw_obs=raw,env=env,instruction=instruction,
                query_interval=30,query_seed=policy_seed+step,action_key=action_key,**policy_parts)
        finally:
            native._format_observation = original
    return query


def require_render_qa(env):
    qa = env.render_initialization_qa
    if not isinstance(qa, dict) or qa.get("status") != "passed":
        raise RuntimeError("native render initialization QA did not pass")
    if int(qa.get("stable_camera_render", {}).get("fallbacks", 0)) != int(qa.get("fallbacks_before_policy", 0)):
        raise RuntimeError("native policy observation used a runtime render fallback")


def run_pair(native, case, policy_parts, dependencies, worker_factory, output_root, capture=None,
             diagnostics_factory=None, relay=None, time_limit_s=None):
    """Run Base and Dynamic with native setup, goals, horizons and event timing.

    The Dynamic arm reconstructs the Base prefix; the relay replays the Base arm's model replies
    for exactly that prefix, then serves live calls."""
    seed = int(case["policy_seed"])
    task, states = native.load_case_task(case,dependencies["benchmark"],dependencies["suite_cache"])
    case_root = output_root / "cases" / case["case_id"]
    case_root.mkdir(parents=True,exist_ok=True)
    (case_root/"scenario.json").write_text(json.dumps(case["scenario"],indent=2))
    rows, history, base_exchanges = {}, None, None
    horizon = native.MAX_STEPS[case["task_suite_name"]]
    query = native_query(native,policy_parts,dependencies["action_key"],policy_seed=seed)
    if capture is not None:
        capture.start_case(case["case_id"])
    for arm in ("control","intervention"):
        dependencies["reseed"](seed)
        env,instruction = native.create_libero_env_with_retry(
            lambda: native._create_libero_env(task,360,seed,
                bddl_root=dependencies["get_libero_path"]("bddl_files"),env_type=dependencies["env_type"]),
            policy_seed=seed,reseed=dependencies["reseed"])
        env = native.wrap_case_env(env,case)
        arm_root = case_root/arm
        arm_root.mkdir(parents=True,exist_ok=True)
        trace = arm_root/"trace.jsonl"
        trace.write_text("")
        worker, url = None, None
        try:
            env.seed(seed)
            wrapped = native.CosmosInterventionEnv(env=env,task_description=instruction,
                scenario=case["scenario"],arm=arm,trace_path=trace,
                original_task_index=case["task_index"],init_state_index=case["init_state_index"])
            wrapped.configure_episode(task_suite_name=case["task_suite_name"],policy_seed=seed,
                                      query_interval=30,max_policy_steps=horizon)
            wrapped.reset()
            raw = wrapped.set_init_state(states[case["init_state_index"]])
            require_render_qa(wrapped)
            for _ in range(wrapped.warmup_steps):
                raw,_,_,_ = wrapped.step(dependencies["dummy_action"]())
            for robot in env.robots:
                robot.controller.use_delta = False
            if relay is not None:
                url = relay.open(f"{case['case_id']}/{arm}",
                                 replay=base_exchanges if arm == "intervention" else None)
            worker = worker_factory(arm_root/"adapter-io", url)
            if capture is not None:
                capture.start_arm(arm)
            diagnostics = None
            if diagnostics_factory is not None:
                diagnostics = diagnostics_factory(wrapped, arm_root, case)
            elif capture is not None:
                diagnostics = load_module(
                    HERE/"diagnostics.py", "libero_adapter_dev_diagnostics").DevEvidence(arm_root)
            row, recorded = run_arm(env=wrapped,worker=worker,initial_observation=raw,
                instruction=instruction,horizon=horizon,query_policy=query,
                public_observation=agent_public_observation(),
                replay=history if arm == "intervention" else None,capture=capture,policy_seed=seed,
                diagnostics=diagnostics, time_limit_s=time_limit_s,
                set_replaying=(lambda on, u=url: relay.set_replaying(u, on)) if url else None)
            require_render_qa(wrapped)
            if url is not None:
                exchanges = relay.close(url)
                url = None
                row["policy_model"] = relay.stats(exchanges)
                if arm == "control":
                    base_exchanges = exchanges["record"]
            rows[arm] = row
            if arm == "control":
                history = recorded
            trace.write_text(json.dumps(row,sort_keys=True)+"\n")
        finally:
            try:
                if worker is not None:
                    worker.close()
            finally:
                if url is not None:
                    relay.close(url)
                env.close()
    summary = load_module(HERE/"paired_summary.py","libero_adapter_pairing").summarize_pair(
        case,rows["control"],rows["intervention"])
    for arm, prefix in (("control", "base"), ("intervention", "dynamic")):
        summary[f"{prefix}_policy_model"] = rows[arm].get("policy_model")
        summary[f"{prefix}_time_limit_reached"] = rows[arm].get("episode_time_limit_reached", False)
    (case_root/"paired_summary.json").write_text(json.dumps(summary,indent=2))
    (case_root/"status.json").write_text(json.dumps({"case_id":case["case_id"],
        "terminal_status":summary["terminal_status"],"execution_complete":True},indent=2))
    return summary


def run_pro_episode(native, case, policy_parts, dependencies, worker_factory, output_root,
                    capture=None, relay=None, time_limit_s=None):
    """Run one official PRO episode through the same public controller as MAX."""
    seed = int(case["policy_seed"])
    dependencies["reseed"](seed)
    task, states = native.load_case_task(case, dependencies["benchmark"], dependencies["suite_cache"])
    env, instruction = native.create_libero_env_with_retry(
        lambda: native._create_libero_env(task, 360, seed,
            bddl_root=dependencies["get_libero_path"]("bddl_files"), env_type=dependencies["env_type"]),
        policy_seed=seed, reseed=dependencies["reseed"])
    env = native.wrap_case_env(env, case)
    case_root = output_root / "cases" / case["case_id"] / "pro"
    case_root.mkdir(parents=True, exist_ok=True)
    worker, url = None, None
    started = time.monotonic()
    if capture is not None:
        capture.start_case(case["case_id"], "pro")
    try:
        env.seed(seed)
        env.reset()
        render_stats = dependencies["install_stable_camera_render"](env)
        raw = env.set_init_state(states[case["init_state_index"]])
        render_qa = dependencies["prime_offscreen_renderer"](env)
        if render_qa["status"] != "passed":
            raise RuntimeError("PRO offscreen renderer did not pass initialization")
        render_qa["stable_camera_render"] = render_stats
        render_qa["fallbacks_before_policy"] = int(render_stats.get("fallbacks", 0))
        for _ in range(10):
            raw, _, _, _ = env.step(dependencies["dummy_action"]())
        for robot in env.robots:
            robot.controller.use_delta = False
        if relay is not None:
            url = relay.open(f"{case['case_id']}/pro")
        worker = worker_factory(case_root / "adapter-io", url)
        diagnostics = (load_module(HERE / "diagnostics.py", "libero_pro_dev_diagnostics").DevEvidence(case_root)
                       if capture is not None else None)
        row, _ = run_arm(env=env, worker=worker, initial_observation=raw,
            instruction=instruction, horizon=native.MAX_STEPS[case["task_suite_name"]],
            query_policy=native_query(native, policy_parts, dependencies["action_key"], policy_seed=seed),
            public_observation=agent_public_observation(), capture=capture, policy_seed=seed,
            diagnostics=diagnostics, paired=False, time_limit_s=time_limit_s)
        if int(render_stats.get("fallbacks", 0)) != render_qa["fallbacks_before_policy"]:
            raise RuntimeError("PRO policy observation used an unstable render fallback")
        if url is not None:
            row["policy_model"] = relay.stats(relay.close(url))
            url = None
        failure = (None if row["success"] else
                   "episode_time_limit" if row.get("episode_time_limit_reached") else "horizon_exhausted")
        row.update(case_id=case["case_id"], failure_category=failure,
            max_policy_steps=native.MAX_STEPS[case["task_suite_name"]],
            policy_chunk_steps=30, warmup_steps=10, policy_seed=seed,
            init_state_index=case["init_state_index"], instruction=instruction,
            substrate=getattr(env, "substrate_info", {}), render_initialization_qa=render_qa,
            wall_s=time.monotonic() - started)
        (case_root / "trace.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n")
        return row
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if url is not None:
                relay.close(url)
            env.close()


def load_resident_policy(runtime):
    """Load the same frozen checkpoint and processors once per source partition."""
    import torch
    sys.path.insert(0,str(Path(runtime["lerobot_root"])/"src"))
    from lerobot.envs.libero import get_libero_dummy_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors
    from lerobot.utils.constants import ACTION
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_max.env_factory import install_stable_camera_render, prime_offscreen_renderer

    def reseed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    reseed(195)
    device = torch.device("cuda")
    policy = XVLAPolicy.from_pretrained(runtime["checkpoint"]).to(device).eval()
    if int(policy.config.n_action_steps) < 30:
        raise ValueError("fixed X-VLA checkpoint must support native 30-step chunks")
    preprocessor,postprocessor = make_pre_post_processors(policy.config,runtime["checkpoint"],
        preprocessor_overrides={"device_processor":{"device":str(device)}})
    env_preprocessor,env_postprocessor = make_xvla_libero_pre_post_processors()
    parts = dict(policy=policy,preprocessor=preprocessor,postprocessor=postprocessor,
                 env_preprocessor=env_preprocessor,env_postprocessor=env_postprocessor)
    dependencies = dict(benchmark=benchmark,get_libero_path=get_libero_path,
        env_type=OffScreenRenderEnv,suite_cache={},reseed=reseed,
        dummy_action=get_libero_dummy_action,action_key=ACTION,
        install_stable_camera_render=install_stable_camera_render,
        prime_offscreen_renderer=prime_offscreen_renderer)
    return parts,dependencies


def main(default_benchmark="max"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",choices=("pro","max"),default=default_benchmark)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--capture-frames",action="store_true")
    parser.add_argument("--episode-model-calls",type=int,required=True)
    parser.add_argument("--episode-time-limit-s",type=float,required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    cases = json.loads(args.manifest.read_text())["cases"]
    old = load_module(HERE/"native_runner.py","libero_adapter_native_helpers")
    native = old.native_runner(runtime)
    policy_parts,dependencies = load_resident_policy(runtime)
    adapter = load_module(HERE/"adapter.py","libero_adapter_cpu_parent")
    agent = load_module(HERE/"agent.py","libero_agent_relay")
    # The evaluation's gateway grant stays in this trusted process; workers get episode URLs.
    relay = agent.Relay(os.environ.get("RSIB_GATEWAY_URL"), os.environ.get("RSIB_GATEWAY_TOKEN"),
                        args.episode_model_calls)
    args.output_root.mkdir(parents=True,exist_ok=True)
    capture = old.FrameCapture(args.output_root/"frames") if args.capture_frames else None
    rows = []
    limit = args.episode_time_limit_s
    try:
        for index,case in enumerate(cases,1):
            factory = lambda io, url: adapter.AdapterWorker(runtime["python"],args.workspace,io,url)
            if args.benchmark == "pro":
                rows.append(run_pro_episode(native,case,policy_parts,dependencies,factory,args.output_root,
                                            capture,relay=relay,time_limit_s=limit))
                (args.output_root/"pro_results.json").write_text(json.dumps({"episodes":rows},indent=2))
                outcome = rows[-1]["success"]
            else:
                rows.append(run_pair(native,case,policy_parts,dependencies,factory,args.output_root,capture,
                                     relay=relay,time_limit_s=limit))
                (args.output_root/"paired_results.json").write_text(json.dumps(rows,indent=2))
                outcome = rows[-1]["intervention_correct"]
            print(f"[{index}/{len(cases)}] {case['case_id']} success={outcome}",flush=True)
    finally:
        if capture is not None:
            capture.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
